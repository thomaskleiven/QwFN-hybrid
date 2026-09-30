#include "qwfn_graph.h"
#include "qwfn_check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "ggml-backend.h"

namespace qwfn {

ggml_tensor * graph_builder::W(const std::string & name) const {
    assert_that(w_ != nullptr, "graph_builder has a primary weight set");
    assert_that(!name.empty(), "weight name is non-empty");
    ggml_tensor * t = w_->get(name);
    if (!t && alt_) t = alt_->get(name);
    if (!t) fprintf(stderr, "[qwfn] missing weight: %s\n", name.c_str());
    return t;
}

ggml_tensor * graph_builder::Wl(int il, const char * suffix) const {
    return W("blk." + std::to_string(il) + "." + suffix);
}

hc_mixed graph_builder::hc_mix(ggml_tensor * x, int il, bool ffn, bool with_inject) {
    assert_that(x != nullptr && hp_ != nullptr, "hc_mix has an input and hparams");
    assert_that(il < 0 || (uint32_t) il < hp_->n_layer, "hc_mix layer index in range");
    ggml_tensor * w_norm;
    ggml_tensor * w_down;
    ggml_tensor * w_up;
    ggml_tensor * w_inject = nullptr;
    if (il < 0) {
        w_norm = W("output_hc_norm.weight");
        w_down = W("output_hc_down.weight");
        w_up   = W("output_hc_up.weight");
    } else {
        const char * p = ffn ? "hc_ffn" : "hc_attn";
        w_norm   = Wl(il, (std::string(p) + "_norm.weight").c_str());
        w_down   = Wl(il, (std::string(p) + "_down.weight").c_str());
        w_up     = Wl(il, (std::string(p) + "_up.weight").c_str());
        w_inject = Wl(il, (std::string(p) + "_inject.weight").c_str());
    }
    return hc_mix_w(x, w_norm, w_down, w_up, w_inject, with_inject);
}

hc_mixed graph_builder::hc_mix_w(ggml_tensor * x, ggml_tensor * w_norm, ggml_tensor * w_down,
                                 ggml_tensor * w_up, ggml_tensor * w_inject, bool with_inject) {
    assert_that(x != nullptr && ctx0 != nullptr && hp_ != nullptr, "hc_mix_w has an input, a context and hparams");
    assert_that(!with_inject || w_inject != nullptr, "an inject output needs an inject weight");
    const int64_t hc      = hp_->hc_count;
    const int64_t n_embd  = hp_->n_embd;
    const int64_t hc_dim  = hc * n_embd;
    const int64_t nt      = x->ne[2];

    // RMSNorm reduces over ne0 = n_embd, i.e. within one stream, but the learned
    // gamma spans all hc*n_embd. The converter folded gamma to (1 + w). The
    // engine shapes the gamma [n_embd, hc] at load so the norm and the multiply
    // are adjacent nodes, which ggml-cuda runs as one fused kernel; a gamma
    // still shaped [hc*n_embd] (a tool without the engine) takes the old form.
    ggml_tensor * xn;
    if (w_norm->ne[0] == n_embd && w_norm->ne[1] == hc) {
        xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hp_->rms_eps), w_norm);   // [n_embd, hc, T]
        xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    } else {
        xn = ggml_rms_norm(ctx0, x, hp_->rms_eps);
        xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
        xn = ggml_mul(ctx0, xn, w_norm);
    }

    ggml_tensor * lo = ggml_mul_mat(ctx0, w_down, xn);                 // [hc_lr, T]
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, w_up, lo));  // [hc_dim, T]

    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

    // Collapse the streams by their mean.
    ggml_tensor * mixed;
    if (gpu_fuse_ && hc_mean_ && nt == 1) {
        // One matmul with the (1/hc) vector: two kernels instead of five. A
        // different summation order from the adds below, so GPU graphs only.
        ggml_tensor * gt = ggml_cont(ctx0, ggml_permute(ctx0, gated, 1, 0, 2, 3));   // [hc, n_embd, T]
        gt = ggml_reshape_2d(ctx0, gt, hc, n_embd * nt);
        mixed = ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, hc_mean_, gt), n_embd, nt);
    } else {
        mixed = ggml_cont(ctx0,
                ggml_view_2d(ctx0, gated, n_embd, nt, ggml_row_size(gated->type, n_embd) * hc, 0));
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }

    hc_mixed r;
    r.out = mixed;
    if (with_inject) {
        r.inject = ggml_mul_mat(ctx0, w_inject, xn);                   // [hc, T]
    }
    return r;
}

ggml_tensor * graph_builder::hc_combine(ggml_tensor * residual, ggml_tensor * block_out,
                                        ggml_tensor * inject) {
    assert_that(residual != nullptr && block_out != nullptr && inject != nullptr, "hc_combine has all three inputs");
    assert_that(ctx0 != nullptr && hp_ != nullptr, "hc_combine has a context and hparams");
    const int64_t hc     = hp_->hc_count;
    const int64_t n_embd = hp_->n_embd;
    const int64_t nt     = residual->ne[2];

    ggml_tensor * w = ggml_sigmoid(ctx0, hp_->hc_inject_prescaled ? inject : ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);                                       // [hc, T]

    // block_out[i] * w[c] for every stream is an outer product per token: one op
    // instead of a repeat and a multiply, and exact (a single product per element).
    ggml_tensor * b3 = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    ggml_tensor * w3 = ggml_reshape_3d(ctx0, w, hc, 1, nt);
    return ggml_add(ctx0, residual, ggml_out_prod(ctx0, b3, w3));       // [n_embd, hc, T]
}


ggml_tensor * graph_builder::rms(ggml_tensor * x, ggml_tensor * w) const {
    return ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hp_->rms_eps), w);
}

ggml_tensor * graph_builder::rope(ggml_tensor * x, ggml_tensor * pos, int * secs) const {
    return ggml_rope_multi(ctx0, x, pos, nullptr, hp_->rope_dim, secs, GGML_ROPE_TYPE_IMROPE,
                           hp_->n_ctx_train, hp_->rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

ggml_tensor * graph_builder::conv_with_history(ggml_tensor * state_row, ggml_tensor * x,
                                               int64_t hist, int64_t channels, ggml_tensor * rb_row) {
    assert_that(state_row != nullptr && x != nullptr, "conv_with_history has a state row and an input");
    assert_that(!persist_ || gf_ != nullptr, "persistent conv history writes need a bound graph");
    // state_row is [hist, channels]; x is [channels, T]. ggml_ssm_conv wants the
    // token axis first, so the history is concatenated ahead of x transposed.
    ggml_tensor * st = ggml_reshape_3d(ctx0, state_row, hist, channels, 1);
    // x transposed made contiguous first: the concat of two contiguous inputs is the fast kernel
    // (the non-contiguous one took ~29 us per layer at T=4). Same bytes either way.
    ggml_tensor * padded = ggml_concat(ctx0, st, ggml_cont(ctx0, ggml_transpose(ctx0, x)), 0);  // [hist+T, channels, 1]

    // Keep the trailing `hist` positions for the next ubatch.
    ggml_tensor * tail = ggml_view_2d(ctx0, padded, hist, channels,
            padded->nb[1], ggml_row_size(padded->type, padded->ne[0] - hist));
    if (persist_) ggml_build_forward_expand(gf_, ggml_cpy(ctx0, tail, state_row));   // cpy handles the strided source
    // Rollback: the history as it stood s tokens back, for s = 1..rb_n_ (slot s-1 of rb_row).
    if (persist_ && rb_row) {
        const int64_t T = padded->ne[0] - hist;
        const int64_t n_snap = std::min<int64_t>(T - 1, rb_row->ne[2] > 0 ? std::min<int64_t>(rb_n_, rb_row->ne[2]) : rb_n_);
        for (int64_t s = 1; s <= n_snap; s++) {
            ggml_tensor * tails = ggml_view_2d(ctx0, padded, hist, channels,
                    padded->nb[1], ggml_row_size(padded->type, padded->ne[0] - hist - s));
            ggml_tensor * dst = ggml_view_2d(ctx0, rb_row, hist, channels, rb_row->nb[1], (size_t) (s - 1) * rb_row->nb[2]);
            ggml_build_forward_expand(gf_, ggml_cpy(ctx0, tails, dst));
        }
    }

    return padded;
}

// -A_log.exp() * softplus(alpha + dt_bias); ssm_a already holds the negated exp.
ggml_tensor * graph_builder::deltanet_decay(ggml_tensor * cur, int il, int64_t n_v_heads, int64_t T) {
    assert_that(cur != nullptr, "deltanet_decay has an input");
    assert_that(T == cur->ne[1], "deltanet_decay input is [n_embd, T]");
    ggml_tensor * alpha = ggml_mul_mat(ctx0, Wl(il, "ssm_alpha.weight"), cur); // [48, T]
    alpha = ggml_reshape_3d(ctx0, alpha, n_v_heads, T, 1);
    ggml_tensor * g = ggml_softplus(ctx0, ggml_add(ctx0, alpha, Wl(il, "ssm_dt.bias")));
    g = ggml_mul(ctx0, g, Wl(il, "ssm_a"));
    return ggml_reshape_4d(ctx0, g, 1, n_v_heads, T, 1);
}

// Short causal conv over all 10240 channels, then SiLU; q, k, v are views of the result.
graph_builder::dn_qkv graph_builder::deltanet_conv_qkv(ggml_tensor * qkv, int il, int64_t T) {
    assert_that(qkv != nullptr && T == qkv->ne[1], "deltanet_conv_qkv input is [conv_dim, T]");
    assert_that(st_ != nullptr, "deltanet_conv_qkv needs a bound state");
    const int64_t head_k    = hp_->ssm_d_state;     // 128
    const int64_t head_v    = hp_->ssm_d_state;     // 128
    const int64_t n_k_heads = hp_->ssm_n_group;     // 16
    const int64_t n_v_heads = hp_->ssm_dt_rank;     // 48
    const int64_t key_dim   = head_k * n_k_heads;   // 2048
    const int64_t value_dim = head_v * n_v_heads;   // 6144
    const int64_t conv_dim  = key_dim * 2 + value_dim;  // 10240

    ggml_tensor * padded  = conv_with_history(st_->rs_conv(il), qkv,
                                              hp_->ssm_d_conv - 1, conv_dim, rb_conv_);
    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, padded, Wl(il, "ssm_conv1d.weight")));
    // conv_out is [conv_dim, T, 1]

    const size_t es  = ggml_element_size(conv_out);
    const size_t nb1 = ggml_row_size(conv_out->type, conv_dim);

    dn_qkv r;
    r.q = ggml_view_4d(ctx0, conv_out, head_k, n_k_heads, T, 1,
            ggml_row_size(conv_out->type, head_k), nb1, nb1 * T, 0);
    r.k = ggml_view_4d(ctx0, conv_out, head_k, n_k_heads, T, 1,
            ggml_row_size(conv_out->type, head_k), nb1, nb1 * T, key_dim * es);
    r.v = ggml_view_4d(ctx0, conv_out, head_v, n_v_heads, T, 1,
            ggml_row_size(conv_out->type, head_v), nb1, nb1 * T, 2 * key_dim * es);

    r.q = ggml_l2_norm(ctx0, r.q, hp_->rms_eps);
    r.k = ggml_l2_norm(ctx0, r.k, hp_->rms_eps);
    return r;
}

// The kernel's final state into the recurrent cache; with K > 1, the K-1 older
// snapshots into the rollback slots (slot s-1 holds the state s tokens back).
void graph_builder::deltanet_write_state(ggml_tensor * result, int il, int64_t T, int K) {
    assert_that(result != nullptr && st_ != nullptr, "deltanet_write_state has a result and a bound state");
    assert_that(K == 1 || (persist_ && rb_rs_ != nullptr), "rollback snapshots need persistence and a target");
    const int64_t head_v    = hp_->ssm_d_state;     // 128
    const int64_t n_v_heads = hp_->ssm_dt_rank;     // 48
    ggml_tensor * s1 = ggml_view_4d(ctx0, result, head_v, head_v, n_v_heads, 1,
            ggml_row_size(result->type, head_v),
            ggml_row_size(result->type, head_v * head_v),
            ggml_row_size(result->type, head_v * head_v * n_v_heads),
            ggml_row_size(result->type, head_v * n_v_heads * T));
    if (persist_) {
        // The destination as a [D, 1, 1] view: ggml-cuda recognises this copy of
        // the kernel's state snapshot and has the gated-delta-net kernel write the
        // state itself, skipping a 3 MB copy per layer. Same bytes either way.
        const int64_t D = head_v * head_v * n_v_heads;
        ggml_tensor * dst = ggml_view_3d(ctx0, st_->rs_state(il), D, 1, 1,
                                         ggml_row_size(GGML_TYPE_F32, D), ggml_row_size(GGML_TYPE_F32, D), 0);
        ggml_build_forward_expand(gf_, ggml_cpy(ctx0, s1, dst));
        for (int s = 1; s < K; s++) {
            ggml_tensor * ss = ggml_view_4d(ctx0, result, head_v, head_v, n_v_heads, 1,
                    ggml_row_size(result->type, head_v),
                    ggml_row_size(result->type, head_v * head_v),
                    ggml_row_size(result->type, head_v * head_v * n_v_heads),
                    ggml_row_size(result->type, head_v * n_v_heads * T) + (size_t) s * ggml_row_size(result->type, D));
            ggml_tensor * rdst = ggml_view_3d(ctx0, rb_rs_, head_v, head_v, n_v_heads, rb_rs_->nb[1], rb_rs_->nb[2], (size_t) (s - 1) * rb_rs_->nb[3]);
            ggml_build_forward_expand(gf_, ggml_cpy(ctx0, ss, rdst));
        }
    }
}

ggml_tensor * graph_builder::deltanet(ggml_tensor * cur, int il) {
    assert_that(cur != nullptr && st_ != nullptr && gf_ != nullptr, "deltanet needs an input and a bound state/graph");
    assert_that(il >= 0 && (uint32_t) il < hp_->n_layer, "deltanet layer index in range");
    const int64_t head_v    = hp_->ssm_d_state;     // 128
    const int64_t n_v_heads = hp_->ssm_dt_rank;     // 48
    const int64_t value_dim = head_v * n_v_heads;   // 6144
    const int64_t T         = cur->ne[1];

    ggml_tensor * qkv = ggml_mul_mat(ctx0, Wl(il, "attn_qkv.weight"),  cur);   // [10240, T]
    ggml_tensor * z   = ggml_mul_mat(ctx0, Wl(il, "attn_gate.weight"), cur);   // [ 6144, T]

    ggml_tensor * beta = ggml_mul_mat(ctx0, Wl(il, "ssm_beta.weight"), cur);   // [48, T]
    beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, beta, 1, n_v_heads, T, 1));

    ggml_tensor * g = deltanet_decay(cur, il, n_v_heads, T);
    const dn_qkv x  = deltanet_conv_qkv(qkv, il, T);

    ggml_tensor * s0 = ggml_reshape_4d(ctx0, st_->rs_state(il), head_v, head_v, n_v_heads, 1);

    // The fused op broadcasts the 16 k-heads across the 48 v-heads itself. K
    // state snapshots follow the scores in the kernel's output, most recent
    // first: slot s is the state s tokens back, which is what a rollback to any of
    // the step's first T-1 positions needs.
    const int K = (persist_ && rb_rs_ && T >= 2) ? (int) std::min<int64_t>(T, rb_n_ + 1) : 1;
    ggml_tensor * result = ggml_gated_delta_net(ctx0, x.q, x.k, x.v, g, beta, s0, K);

    ggml_tensor * out = ggml_view_4d(ctx0, result, head_v, n_v_heads, T, 1,
            ggml_row_size(result->type, head_v),
            ggml_row_size(result->type, head_v * n_v_heads),
            ggml_row_size(result->type, head_v * n_v_heads * T), 0);
    deltanet_write_state(result, il, T, K);

    // Gated RMSNorm; sigmoid gate here, unlike Qwen3.5's GDN which uses silu.
    ggml_tensor * zg = ggml_reshape_4d(ctx0, z, head_v, n_v_heads, T, 1);
    ggml_tensor * o  = ggml_mul(ctx0, rms(out, Wl(il, "ssm_norm.weight")), ggml_sigmoid(ctx0, zg));

    o = ggml_reshape_2d(ctx0, o, value_dim, T);
    return ggml_mul_mat(ctx0, Wl(il, "ssm_out.weight"), o);
}

// Pool r raw indexer keys per block (mean), norm, rope at the block positions.
// members is [idx_dim, r*n_blk] in block-major order; returns [idx_dim, n_blk].
static ggml_tensor * qsa_pool_blocks(ggml_context * ctx0, const hparams * hp, ggml_tensor * members,
                                     int64_t idx_dim, int64_t r, int64_t n_blk, ggml_tensor * k_norm_w,
                                     ggml_tensor * blk_pos, const int secs[4]) {
    assert_that(ctx0 != nullptr && hp != nullptr && members != nullptr, "qsa_pool_blocks has a context, hparams and keys");
    assert_that(r > 0 && n_blk > 0, "qsa_pool_blocks pools at least one block of at least one key");
    members = ggml_reshape_3d(ctx0, members, idx_dim, r, n_blk);
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; i++) {
        ggml_tensor * slice = ggml_cont(ctx0,
                ggml_view_2d(ctx0, members, idx_dim, n_blk, members->nb[2], i * members->nb[1]));
        pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f / (float) r);
    pooled = ggml_mul(ctx0, ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, pooled, idx_dim, n_blk, 1), hp->rms_eps), k_norm_w);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blk);
    pooled = ggml_rope_multi(ctx0, pooled, blk_pos, nullptr, hp->rope_dim, (int *) secs,
                             GGML_ROPE_TYPE_IMROPE, hp->n_ctx_train, hp->rope_freq_base,
                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return ggml_reshape_2d(ctx0, pooled, idx_dim, n_blk);
}

// The indexer query: projected, per-head normed and roped, [idx_dim, n_idx_h, T].
ggml_tensor * graph_builder::idx_query(ggml_tensor * cur, ggml_tensor * inp_pos, int * secs, int il, int64_t T) {
    assert_that(cur != nullptr && inp_pos != nullptr, "idx_query has an input and positions");
    assert_that(secs != nullptr, "idx_query has rope sections");
    ggml_tensor * q = ggml_mul_mat(ctx0, Wl(il, "indexer.q_proj.weight"), cur);      // [512, T]
    q = ggml_reshape_3d(ctx0, q, hp_->idx_key_len, hp_->idx_n_head, T);
    q = rms(q, Wl(il, "indexer.q_norm.weight"));
    return rope(q, inp_pos, secs);
}

ggml_tensor * graph_builder::qsa_top_k(ggml_tensor * cur, ggml_tensor * inp_pos,
                                       const int sections[4], int il,
                                       const qsa_inputs & qsa) {
    assert_that(cur != nullptr && st_ != nullptr && gf_ != nullptr, "qsa_top_k needs an input and a bound state/graph");
    assert_that(qsa.ratio > 0 && qsa.n_blocks > 0 && qsa.blk_cells != nullptr, "qsa_top_k has block inputs");
    const int64_t idx_dim  = hp_->idx_key_len;    // 128
    const int64_t n_idx_h  = hp_->idx_n_head;     // 4
    const int64_t r        = qsa.ratio;
    const int64_t n_blocks = qsa.n_blocks;
    const int64_t T        = cur->ne[1];
    const int64_t n_kv     = n_past_ + T;

    int secs[4] = { sections[0], sections[1], sections[2], sections[3] };

    // Cached indexer keys are stored RAW: pooling happens before norm and rope,
    // so neither may be applied on the way in.
    ggml_tensor * k_raw = ggml_mul_mat(ctx0, Wl(il, "indexer.k_proj.weight"), cur);  // [128, T]
    ggml_tensor * ic = st_->idx_cache(il);
    ggml_build_forward_expand(gf_, ggml_cpy(ctx0,
            ggml_reshape_2d(ctx0, k_raw, idx_dim, T),
            ggml_view_2d(ctx0, ic, idx_dim, T, ggml_row_size(ic->type, idx_dim),
                         ggml_row_size(ic->type, idx_dim) * n_past_)));

    ggml_tensor * k_all = ggml_view_2d(ctx0, ic, idx_dim, n_kv,
                                       ggml_row_size(ic->type, idx_dim), 0);

    // Mean-pool each block's member keys ([128, r*n_blk]), norm, rope.
    ggml_tensor * pooled = qsa_pool_blocks(ctx0, hp_, ggml_get_rows(ctx0, k_all, qsa.blk_cells), idx_dim, r, n_blocks,
                                           Wl(il, "indexer.k_norm.weight"), qsa.blk_pos, secs);
    ggml_tensor * q = idx_query(cur, inp_pos, secs, il, T);

    // Rectify each head's dot product before summing, as in the DeepSeek lightning indexer.
    ggml_tensor * score = ggml_mul_mat(ctx0, pooled,
            ggml_reshape_2d(ctx0, ggml_cont(ctx0, q), idx_dim, n_idx_h * T));        // [n_blk, 4*T]
    score = ggml_reshape_3d(ctx0, score, n_blocks, n_idx_h, T);
    score = ggml_relu(ctx0, score);
    score = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));                  // [4, n_blk, T]
    score = ggml_sum_rows(ctx0, score);                                              // [1, n_blk, T]
    score = ggml_reshape_2d(ctx0, score, n_blocks, T);

    score = ggml_add(ctx0, score, qsa.bias);

    // Select whole BLOCKS -- ceil(width / r) of them, 513 here -- and expand to
    // their cells. Every cell of a block inherits its block's score, so this is
    // the cell-level top-k up to the one partial block it would cut; it is also
    // what the decode path selects, and it removes the [n_kv, T] F32 score
    // expansion that made the prefill graph's memory scale with n_kv * T. The
    // causal mask, added later, still removes future and empty cells.
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hp_->idx_top_k + r - 1);
    const int64_t kb    = std::min<int64_t>(n_blocks, (width + r - 1) / r);
    ggml_tensor * top   = ggml_top_k(ctx0, score, kb);                                // [kb, T] I32
    ggml_tensor * table = ggml_reshape_3d(ctx0, qsa.blk_cells, r, n_blocks, 1);       // (k, b) -> b*r + k
    ggml_tensor * cells = ggml_get_rows(ctx0, table, ggml_reshape_1d(ctx0, ggml_cont(ctx0, top), kb * T)); // [r, kb*T]
    return ggml_cont(ctx0, ggml_reshape_2d(ctx0, cells, r * kb, T));                  // [r*kb, T]
}

// Append this ubatch's K and V rows to layer il's caches at n_past.
void graph_builder::kv_append(ggml_tensor * K, ggml_tensor * V, int il, int64_t T) {
    assert_that(K != nullptr && V != nullptr, "kv_append has K and V");
    assert_that(st_ != nullptr && gf_ != nullptr, "kv_append needs a bound state and graph");
    const int64_t kv_dim = (int64_t) hp_->n_embd_head_k * hp_->n_head_kv;
    ggml_tensor * kc = st_->k_cache(il);
    ggml_tensor * vc = st_->v_cache(il);
    ggml_build_forward_expand(gf_, ggml_cpy(ctx0,
            ggml_reshape_2d(ctx0, K, kv_dim, T),
            ggml_view_2d(ctx0, kc, kv_dim, T, ggml_row_size(kc->type, kv_dim),
                         ggml_row_size(kc->type, kv_dim) * n_past_)));
    ggml_build_forward_expand(gf_, ggml_cpy(ctx0,
            ggml_reshape_2d(ctx0, V, kv_dim, T),
            ggml_view_2d(ctx0, vc, kv_dim, T, ggml_row_size(vc->type, kv_dim),
                         ggml_row_size(vc->type, kv_dim) * n_past_)));
}

// Restrict the mask to the cells the indexer chose: start from all -inf,
// write 0 at the selected indices, then add the causal mask back.
ggml_tensor * graph_builder::qsa_mask(ggml_tensor * top_k, ggml_tensor * kq_mask, int64_t n_kv, int64_t T) {
    assert_that(top_k != nullptr && kq_mask != nullptr, "qsa_mask has a selection and a causal mask");
    assert_that(top_k->ne[1] == T, "qsa_mask selection is [width, T]");
    // Start from all -inf, write 0 at the chosen cells, then add the causal
    // mask back so future and empty cells stay masked. Built at exactly
    // [n_kv, T]: ggml_set_rows requires a->ne[2] == b->ne[2], and the
    // attention mask may carry padding rows that top_k does not.
    // Built in the mask's own type (F16): half the bytes of the F32 build
    // and no cast, which at n_kv * T = 96M is 400 MB less arena.
    ggml_tensor * base = ggml_fill(ctx0,
            ggml_new_tensor_2d(ctx0, kq_mask->type, n_kv, T), -INFINITY);
    ggml_tensor * a4 = ggml_view_4d(ctx0, base, 1, n_kv, T, 1,
                                    base->nb[0], base->nb[1], base->nb[2], 0);

    ggml_tensor * idx = ggml_view_3d(ctx0, top_k, top_k->ne[0], T, 1,
                                     top_k->nb[1], top_k->nb[1] * T, 0);
    ggml_tensor * zeros = ggml_fill(ctx0,
            ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k->ne[0], T, 1), 0.0f);

    ggml_tensor * sel = ggml_set_rows(ctx0, a4, zeros, idx);
    sel = ggml_view_2d(ctx0, sel, n_kv, T, base->nb[1], 0);

    // In place: the causal mask is added into the selection tensor rather
    // than into a second [n_kv, T] buffer. At the chunk cap that is one
    // 96 MB tensor fewer in the arena that made a 131K prefill abort.
    ggml_tensor * causal = ggml_view_2d(ctx0, kq_mask, n_kv, T, kq_mask->nb[1], 0);
    return ggml_add_inplace(ctx0, sel, causal);
}

ggml_tensor * graph_builder::sparse_attn(ggml_tensor * cur, ggml_tensor * inp_pos,
                                         ggml_tensor * kq_mask, const int sections[4], int il,
                                         const qsa_inputs * qsa) {
    assert_that(cur != nullptr && st_ != nullptr && gf_ != nullptr, "sparse_attn needs an input and a bound state/graph");
    assert_that(il >= 0 && (uint32_t) il < hp_->n_layer, "sparse_attn layer index in range");
    const int64_t hd    = hp_->n_embd_head_k;   // 256
    const int64_t nh    = hp_->n_head;          // 24
    const int64_t nh_kv = hp_->n_head_kv;       // 2
    const int64_t T     = cur->ne[1];
    const int64_t n_kv  = n_past_ + T;

    // The indexer reads the same block input as q/k/v; no ratio means dense.
    ggml_tensor * top_k = (qsa && qsa->ratio > 0)
        ? qsa_top_k(cur, inp_pos, sections, il, *qsa) : nullptr;

    // One projection emits query and gate INTERLEAVED PER HEAD: for head h, the
    // query occupies [h*2*hd, h*2*hd+hd) and the gate the next hd.
    ggml_tensor * qg = ggml_mul_mat(ctx0, Wl(il, "attn_q.weight"), cur);   // [2*hd*nh, T]
    const size_t es = ggml_element_size(qg);

    ggml_tensor * Q = ggml_view_3d(ctx0, qg, hd, nh, T, es * hd * 2, es * hd * 2 * nh, 0);
    Q = rms(Q, Wl(il, "attn_q_norm.weight"));
    ggml_tensor * gate = ggml_view_3d(ctx0, qg, hd, nh, T, es * hd * 2, es * hd * 2 * nh, es * hd);
    gate = ggml_cont_2d(ctx0, gate, hd * nh, T);
    ggml_tensor * K = ggml_mul_mat(ctx0, Wl(il, "attn_k.weight"), cur);
    K = ggml_reshape_3d(ctx0, K, hd, nh_kv, T);
    K = rms(K, Wl(il, "attn_k_norm.weight"));

    ggml_tensor * V = ggml_mul_mat(ctx0, Wl(il, "attn_v.weight"), cur);
    V = ggml_reshape_3d(ctx0, V, hd, nh_kv, T);

    // Interleaved multimodal RoPE: sections [11,11,10,0] over t/h/w.
    int secs[4] = { sections[0], sections[1], sections[2], sections[3] };
    Q = rope(Q, inp_pos, secs);
    K = rope(K, inp_pos, secs);

    kv_append(K, V, il, T);
    ggml_tensor * kc = st_->k_cache(il);
    ggml_tensor * vc = st_->v_cache(il);
    const int64_t kv_dim = hd * nh_kv;

    // ggml_flash_attn_ext wants [head_dim, n_kv, n_head_kv]. The cache is written
    // token-major (kv_dim contiguous per token), so the token stride is the outer
    // one and the head stride the inner one -- nb2 < nb1, which a view handles.
    ggml_tensor * k_all = ggml_view_3d(ctx0, kc, hd, n_kv, nh_kv,
            ggml_row_size(kc->type, kv_dim), ggml_row_size(kc->type, hd), 0);
    ggml_tensor * v_all = ggml_view_3d(ctx0, vc, hd, n_kv, nh_kv,
            ggml_row_size(vc->type, kv_dim), ggml_row_size(vc->type, hd), 0);

    ggml_tensor * mask = top_k ? qsa_mask(top_k, kq_mask, n_kv, T) : kq_mask;

    ggml_tensor * q = ggml_permute(ctx0, Q, 0, 2, 1, 3);                 // [hd, T, nh]
    ggml_tensor * out = ggml_flash_attn_ext(ctx0, q, k_all, v_all, mask,
                                            1.0f / sqrtf((float) hd), 0.0f, 0.0f);
    assert_that(ggml_prec_set_acc(out, GGML_PREC_F32), "flash attention accepts F32 accumulation");
    out = ggml_reshape_2d(ctx0, out, hd * nh, T);

    out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, gate));
    return ggml_mul_mat(ctx0, Wl(il, "attn_output.weight"), out);
}

void graph_builder::qsa_pool_rebuild(int il, const qsa_decode_inputs & qd, int64_t n_whole, ggml_tensor * blk_pos_all) {
    assert_that(st_ != nullptr && gf_ != nullptr, "qsa_pool_rebuild needs a bound state and graph");
    assert_that(il >= 0 && (uint32_t) il < hp_->n_layer, "qsa_pool_rebuild layer index in range");
    if (n_whole <= 0) return;
    const int64_t idx_dim = hp_->idx_key_len;
    const int64_t r       = qd.ratio;
    int secs[4] = { hp_->mrope_sections[0], hp_->mrope_sections[1], hp_->mrope_sections[2], hp_->mrope_sections[3] };
    // The caches are declared 1D; address them as [row, n_ctx].
    ggml_tensor * ic = st_->idx_cache(il);
    ic = ggml_reshape_2d(ctx0, ic, idx_dim, ic->ne[0] / idx_dim);
    // blk_cells is [r, n_blk_max] with entry (k, b) = b*r + k, so its first
    // r*n_whole entries are exactly cells 0..r*n_whole-1 in block-major order.
    ggml_tensor * cells = ggml_view_1d(ctx0, qd.blk_cells, r * n_whole, 0);
    ggml_tensor * k_all = ggml_view_2d(ctx0, ic, idx_dim, r * n_whole, ggml_row_size(ic->type, idx_dim), 0);
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, cells);                       // [idx_dim, r*n_whole]
    ggml_tensor * pooled = qsa_pool_blocks(ctx0, hp_, members, idx_dim, r, n_whole,
                                           Wl(il, "indexer.k_norm.weight"), blk_pos_all, secs);
    ggml_build_forward_expand(gf_, ggml_cpy(ctx0, pooled,
            ggml_view_2d(ctx0, qd.pool_cache, idx_dim, n_whole, qd.pool_cache->nb[1], 0)));
}

graph_builder::qsa_proj graph_builder::qsa_decode_proj(ggml_tensor * cur, ggml_tensor * inp_pos, const int sections[4], int il) {
    assert_that(cur != nullptr && inp_pos != nullptr, "qsa_decode_proj has an input and positions");
    assert_that(il >= 0 && (uint32_t) il < hp_->n_layer, "qsa_decode_proj layer index in range");
    const int64_t hd = hp_->n_embd_head_k, nh = hp_->n_head, nh_kv = hp_->n_head_kv;
    const int64_t idx_dim = hp_->idx_key_len, n_idx_h = hp_->idx_n_head, T = cur->ne[1];
    int secs[4] = { sections[0], sections[1], sections[2], sections[3] };
    qsa_proj p;
    p.k_raw = ggml_mul_mat(ctx0, Wl(il, "indexer.k_proj.weight"), cur);                   // [128, T]
    ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, Wl(il, "indexer.q_proj.weight"), cur), idx_dim, n_idx_h, T);
    q = rms(q, Wl(il, "indexer.q_norm.weight"));
    p.q_idx = rope(q, inp_pos, secs);
    ggml_tensor * qg = ggml_mul_mat(ctx0, Wl(il, "attn_q.weight"), cur);                   // [2*hd*nh, T]
    const size_t es = ggml_element_size(qg);
    ggml_tensor * Q = ggml_view_3d(ctx0, qg, hd, nh, T, es * hd * 2, qg->nb[1], 0);
    Q = rms(Q, Wl(il, "attn_q_norm.weight"));
    p.Q = rope(Q, inp_pos, secs);
    p.gate = ggml_cont_2d(ctx0, ggml_view_3d(ctx0, qg, hd, nh, T, es * hd * 2, qg->nb[1], es * hd), hd * nh, T);
    ggml_tensor * K = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, Wl(il, "attn_k.weight"), cur), hd, nh_kv, T);
    K = rms(K, Wl(il, "attn_k_norm.weight"));
    p.K = rope(K, inp_pos, secs);
    p.V = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, Wl(il, "attn_v.weight"), cur), hd, nh_kv, T);
    return p;
}

ggml_tensor * graph_builder::qsa_out_proj(ggml_tensor * out, int il) {
    return ggml_mul_mat(ctx0, Wl(il, "attn_output.weight"), out);
}

// Decode indexer: the raw key into the index cache at write_idx, then the block holding
// this token pooled into the pool cache. Returns the written caches as {ic, pc}.
graph_builder::qsa_chain graph_builder::qsa_dec_index(ggml_tensor * cur, int il, const qsa_decode_inputs & qd,
                                                      const qsa_chain * chain, const qsa_proj * pre, int64_t kpos,
                                                      int * secs) {
    assert_that(pre != nullptr || cur != nullptr, "qsa_dec_index has an input or projections");
    assert_that(qd.pool_cache != nullptr && qd.write_idx != nullptr && qd.member_idx != nullptr,
                "qsa_dec_index has its decode inputs");
    const int64_t idx_dim = hp_->idx_key_len;     // 128
    const int64_t r       = qd.ratio;
    ggml_tensor * k_raw = pre ? ggml_view_2d(ctx0, pre->k_raw, idx_dim, 1, pre->k_raw->nb[1], (size_t) kpos * pre->k_raw->nb[1])
                              : ggml_mul_mat(ctx0, Wl(il, "indexer.k_proj.weight"), cur);     // [128, 1]
    ggml_tensor * ic    = st_->idx_cache(il);                                          // declared 1D
    ic = ggml_reshape_2d(ctx0, ic, idx_dim, ic->ne[0] / idx_dim);
    if (chain && chain->ic) ic = chain->ic;
    qsa_chain w;
    w.ic = ggml_set_rows(ctx0, ic, k_raw, qd.write_idx);
    ggml_build_forward_expand(gf_, w.ic);
    // Reading through ic_w orders the gather after the write.
    ggml_tensor * members = ggml_get_rows(ctx0, w.ic, qd.member_idx);                 // [128, r]
    ggml_tensor * pooled  = qsa_pool_blocks(ctx0, hp_, members, idx_dim, r, 1,
                                            Wl(il, "indexer.k_norm.weight"), qd.blk_pos, secs);   // [128, 1]
    w.pc = ggml_set_rows(ctx0, (chain && chain->pc) ? chain->pc : qd.pool_cache, pooled, qd.blk_idx);
    ggml_build_forward_expand(gf_, w.pc);
    return w;
}

// Decode block scores over the bucket (read through the pool-cache write), the top
// k_blocks blocks expanded to their cell ids, I32 [k_blocks * ratio].
ggml_tensor * graph_builder::qsa_dec_cells(ggml_tensor * cur, ggml_tensor * inp_pos, int il, const qsa_decode_inputs & qd,
                                           const qsa_proj * pre, int64_t kpos, int * secs, ggml_tensor * pc_w) {
    assert_that(pc_w != nullptr && qd.bias != nullptr && qd.blk_cells != nullptr, "qsa_dec_cells has its inputs");
    assert_that(qd.k_blocks <= qd.n_bucket, "decode QSA keeps at most the blocks it scores");
    const int64_t idx_dim = hp_->idx_key_len;     // 128
    const int64_t n_idx_h = hp_->idx_n_head;      // 4
    const int64_t NB      = qd.n_bucket;
    const int64_t KB      = qd.k_blocks;
    const int64_t NC      = KB * (int64_t) qd.ratio;
    ggml_tensor * q = pre ? ggml_view_3d(ctx0, pre->q_idx, idx_dim, n_idx_h, 1, pre->q_idx->nb[1], pre->q_idx->nb[2],
                                         (size_t) kpos * pre->q_idx->nb[2])
                          : idx_query(cur, inp_pos, secs, il, 1);
    ggml_tensor * pool_v = ggml_view_2d(ctx0, pc_w, idx_dim, NB, pc_w->nb[1], 0);      // after the write
    ggml_tensor * score = ggml_mul_mat(ctx0, pool_v,
            ggml_reshape_2d(ctx0, ggml_cont(ctx0, q), idx_dim, n_idx_h));              // [NB, 4]
    score = ggml_relu(ctx0, score);
    score = ggml_cont(ctx0, ggml_transpose(ctx0, score));                             // [4, NB]
    score = ggml_sum_rows(ctx0, score);                                               // [1, NB]
    score = ggml_reshape_2d(ctx0, score, NB, 1);
    score = ggml_add(ctx0, score, ggml_reshape_2d(ctx0, ggml_view_1d(ctx0, qd.bias, NB, 0), NB, 1));
    ggml_tensor * top = ggml_top_k(ctx0, score, KB);                                   // [KB, 1] I32
    ggml_tensor * cells = ggml_get_rows(ctx0, qd.blk_cells, ggml_reshape_1d(ctx0, top, KB));   // [r, KB] I32
    return ggml_reshape_1d(ctx0, cells, NC);
}

// Decode q, gate, k, v for one position: views of `pre`'s position kpos, or projected here.
graph_builder::qsa_proj graph_builder::qsa_dec_qkv(ggml_tensor * cur, ggml_tensor * inp_pos, int il,
                                                   const qsa_proj * pre, int64_t kpos, int * secs) {
    assert_that(pre != nullptr || (cur != nullptr && inp_pos != nullptr), "qsa_dec_qkv has projections or an input");
    assert_that(kpos >= 0, "qsa_dec_qkv position is non-negative");
    const int64_t hd = hp_->n_embd_head_k, nh = hp_->n_head, nh_kv = hp_->n_head_kv;
    qsa_proj p;
    if (pre) {
        p.Q    = ggml_view_3d(ctx0, pre->Q, hd, nh, 1, pre->Q->nb[1], pre->Q->nb[2], (size_t) kpos * pre->Q->nb[2]);
        p.gate = ggml_view_2d(ctx0, pre->gate, hd * nh, 1, pre->gate->nb[1], (size_t) kpos * pre->gate->nb[1]);
        p.K    = ggml_view_3d(ctx0, pre->K, hd, nh_kv, 1, pre->K->nb[1], pre->K->nb[2], (size_t) kpos * pre->K->nb[2]);
        p.V    = ggml_view_3d(ctx0, pre->V, hd, nh_kv, 1, pre->V->nb[1], pre->V->nb[2], (size_t) kpos * pre->V->nb[2]);
        return p;
    }
    ggml_tensor * qg = ggml_mul_mat(ctx0, Wl(il, "attn_q.weight"), cur);              // [2*hd*nh, 1]
    const size_t es = ggml_element_size(qg);
    p.Q = ggml_view_3d(ctx0, qg, hd, nh, 1, es * hd * 2, es * hd * 2 * nh, 0);
    p.Q = rms(p.Q, Wl(il, "attn_q_norm.weight"));
    p.gate = ggml_view_3d(ctx0, qg, hd, nh, 1, es * hd * 2, es * hd * 2 * nh, es * hd);
    p.gate = ggml_cont_2d(ctx0, p.gate, hd * nh, 1);

    p.K = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, Wl(il, "attn_k.weight"), cur), hd, nh_kv, 1);
    p.K = rms(p.K, Wl(il, "attn_k_norm.weight"));
    p.V = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, Wl(il, "attn_v.weight"), cur), hd, nh_kv, 1);
    p.Q = rope(p.Q, inp_pos, secs);
    p.K = rope(p.K, inp_pos, secs);
    return p;
}

// Decode K/V rows into the caches at write_idx (through the chain's earlier writes).
// Returns the written caches as {kc, vc}.
graph_builder::qsa_chain graph_builder::qsa_dec_kv_write(const qsa_proj & p, int il, const qsa_decode_inputs & qd,
                                                         const qsa_chain * chain, bool pre) {
    assert_that(p.K != nullptr && p.V != nullptr, "qsa_dec_kv_write has K and V");
    assert_that(qd.write_idx != nullptr && st_ != nullptr && gf_ != nullptr, "qsa_dec_kv_write has a row and a bound state/graph");
    const int64_t kv_dim = (int64_t) hp_->n_embd_head_k * hp_->n_head_kv;
    ggml_tensor * kc   = st_->k_cache(il);
    ggml_tensor * vc   = st_->v_cache(il);
    kc = ggml_reshape_2d(ctx0, kc, kv_dim, kc->ne[0] / kv_dim);
    vc = ggml_reshape_2d(ctx0, vc, kv_dim, vc->ne[0] / kv_dim);
    if (chain && chain->kc) { kc = chain->kc; vc = chain->vc; }
    qsa_chain w;
    w.kc = ggml_set_rows(ctx0, kc, ggml_reshape_2d(ctx0, pre ? ggml_cont(ctx0, p.K) : p.K, kv_dim, 1), qd.write_idx);
    w.vc = ggml_set_rows(ctx0, vc, ggml_reshape_2d(ctx0, pre ? ggml_cont(ctx0, p.V) : p.V, kv_dim, 1), qd.write_idx);
    ggml_build_forward_expand(gf_, w.kc);
    ggml_build_forward_expand(gf_, w.vc);
    return w;
}

// The selected cells' K or V rows, [hd, ncp, nh_kv]: raw cache words when `raw`, else
// dequantised to F32 and rounded to F16.
ggml_tensor * graph_builder::qsa_gather(ggml_tensor * cache_w, ggml_tensor * cells, ggml_tensor * gcells, bool raw,
                                        size_t row, int64_t ncp) {
    assert_that(cache_w != nullptr && cells != nullptr && gcells != nullptr, "qsa_gather has a cache and cell ids");
    assert_that(!raw || row % 4 == 0, "a raw gather copies whole words");
    const int64_t hd = hp_->n_embd_head_k, nh_kv = hp_->n_head_kv;
    if (raw) {
        ggml_tensor * words = ggml_view_2d(ctx0, cache_w, cache_w->ne[0], cache_w->ne[1], cache_w->nb[1], 0);
        words->type = GGML_TYPE_I32; words->ne[0] = (int64_t) row / 4; words->nb[0] = 4;
        ggml_tensor * g = ggml_get_rows(ctx0, words, gcells);                        // raw [row/4, NCP]
        ggml_tensor * t = ggml_view_1d(ctx0, g, ggml_nelements(g), 0);
        t->type = cache_w->type;                                                     // [hd, NCP, nh_kv]
        t->ne[0] = hd; t->ne[1] = ncp; t->ne[2] = nh_kv; t->ne[3] = 1;
        t->nb[0] = ggml_type_size(t->type); t->nb[1] = row; t->nb[2] = ggml_row_size(t->type, hd); t->nb[3] = row * ncp;
        return t;
    }
    ggml_tensor * g = ggml_get_rows(ctx0, cache_w, cells);                        // F32 [kv_dim, NC]
    g = ggml_reshape_3d(ctx0, g, hd, nh_kv, ncp);
    return ggml_cpy(ctx0, ggml_permute(ctx0, g, 0, 2, 1, 3),
                    ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, hd, ncp, nh_kv));   // [hd, NC, nh_kv]
}

ggml_tensor * graph_builder::sparse_attn_decode(ggml_tensor * cur, ggml_tensor * inp_pos, const int sections[4],
                                                int il, const qsa_decode_inputs & qd, qsa_chain * chain,
                                                const qsa_proj * pre, int64_t kpos) {
    assert_that(st_ != nullptr && gf_ != nullptr, "sparse_attn_decode needs a bound state and graph");
    assert_that(qd.ratio > 0 && qd.k_blocks > 0 && qd.k_blocks <= qd.n_bucket, "decode QSA keeps k_blocks of n_bucket blocks");
    assert_that(pre == nullptr || (kpos >= 0 && kpos < pre->Q->ne[2]), "position k lies within the projected step");
    const int64_t hd      = hp_->n_embd_head_k;   // 256
    const int64_t nh      = hp_->n_head;          // 24
    const int64_t kv_dim  = hd * (int64_t) hp_->n_head_kv;
    const int64_t NC      = qd.k_blocks * (int64_t) qd.ratio;   // cells attended
    int secs[4] = { sections[0], sections[1], sections[2], sections[3] };

    // ---- indexer: raw key in, this token's block pooled -------------------
    const qsa_chain ix = qsa_dec_index(cur, il, qd, chain, pre, kpos, secs);
    // ---- scores over the bucket, top blocks -> cells ----------------------
    ggml_tensor * cells = qsa_dec_cells(cur, inp_pos, il, qd, pre, kpos, secs, ix.pc);

    // ---- cell mask: 0 for cells <= n_past, -inf beyond ---------------------
    ggml_tensor * cp   = ggml_get_rows(ctx0, qd.cell_pos, cells);                     // [1, NC]
    cp = ggml_reshape_2d(ctx0, cp, NC, 1);
    ggml_tensor * dpos = ggml_sub(ctx0, cp, qd.npast_f);                              // cell - n_past
    ggml_tensor * ok   = ggml_step(ctx0, ggml_scale_bias(ctx0, dpos, -1.0f, 0.5f));   // 1 if cell <= n_past
    ggml_tensor * mask = ggml_cast(ctx0, ggml_scale_bias(ctx0, ok, 1e30f, -1e30f), GGML_TYPE_F16);  // 0 / -inf

    // ---- q, k, v; cache writes; gather the selected cells -----------------
    const qsa_proj  p  = qsa_dec_qkv(cur, inp_pos, il, pre, kpos, secs);
    const qsa_chain kv = qsa_dec_kv_write(p, il, qd, chain, pre != nullptr);

    // Gather the chosen cells' raw cache bytes -- an I32 row copy, no dequantisation -- padded
    // to a multiple of 256 cells with masked duplicates, so flash attention reads the cache
    // type directly with its vector kernel, as llama.cpp does over its own cache. The
    // fallback, for rows that are not whole words or K/V caches of different row sizes
    // (dequantise to F32, round to F16, mma kernel), moves ~6x the bytes.
    const size_t  row = ggml_row_size(kv.kc->type, kv_dim);
    const bool    raw = row % 4 == 0 && row == ggml_row_size(kv.vc->type, kv_dim);
    const int64_t NCP = raw ? GGML_PAD(NC, 256) : NC;
    ggml_tensor * gcells = NCP > NC ? ggml_concat(ctx0, cells, ggml_view_1d(ctx0, cells, NCP - NC, 0), 0) : cells;
    if (NCP > NC) {
        ggml_tensor * pad = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, NCP - NC, 1), 0.0f);
        mask = ggml_cast(ctx0, ggml_scale_bias(ctx0, ggml_concat(ctx0, ok, pad, 0), 1e30f, -1e30f), GGML_TYPE_F16);
    }
    ggml_tensor * Kg = qsa_gather(kv.kc, cells, gcells, raw, row, NCP);
    ggml_tensor * Vg = qsa_gather(kv.vc, cells, gcells, raw, row, NCP);

    ggml_tensor * qp  = ggml_permute(ctx0, p.Q, 0, 2, 1, 3);                          // [hd, 1, nh]
    ggml_tensor * out = ggml_flash_attn_ext(ctx0, qp, Kg, Vg, mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
    assert_that(ggml_prec_set_acc(out, GGML_PREC_F32), "flash attention accepts F32 accumulation");
    out = ggml_reshape_2d(ctx0, out, hd * nh, 1);
    out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, p.gate));
    if (chain) { chain->ic = ix.ic; chain->pc = ix.pc; chain->kc = kv.kc; chain->vc = kv.vc; }
    if (pre) return out;   // projected by the caller for all positions at once
    return ggml_mul_mat(ctx0, Wl(il, "attn_output.weight"), out);
}

// Depthwise causal conv, dilated by the n-gram size, as a sum of shifted taps; then SiLU.
ggml_tensor * graph_builder::ple_conv(ggml_tensor * normalized, int il, int64_t T) {
    assert_that(normalized != nullptr && T == normalized->ne[1], "ple_conv input is [hc_dim, T]");
    assert_that(st_ != nullptr, "ple_conv needs a bound state");
    const int64_t hc     = hp_->hc_count;
    const int64_t n_embd = hp_->n_embd;
    const int64_t hc_dim = hc * n_embd;
    const int64_t kern = hp_->ple_conv_kernel;
    const int64_t dil  = hp_->ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    ggml_tensor * padded = conv_with_history(st_->ple_conv(), normalized, hist, hc_dim, rb_ple_conv_);
    ggml_tensor * w1d    = Wl(il, "ple_conv1d.weight");   // [kern, hc_dim]

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; k++) {
        const int64_t start = hist - (kern - 1 - k) * dil;
        ggml_tensor * shifted = ggml_cont(ctx0, ggml_transpose(ctx0,
                ggml_view_2d(ctx0, padded, T, hc_dim, padded->nb[1],
                             ggml_row_size(padded->type, start))));
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, w1d, 1, hc_dim, w1d->nb[1], k * w1d->nb[0]));
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }
    conv_out = ggml_silu(ctx0, conv_out);
    return ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, T);
}

ggml_tensor * graph_builder::ple(ggml_tensor * emb, ggml_tensor * hidden, int il) {
    assert_that(emb != nullptr && hidden != nullptr, "ple has embeddings and a residual");
    assert_that(st_ != nullptr && il >= 0 && (uint32_t) il < hp_->n_layer, "ple needs a bound state and a valid layer");
    const int64_t hc     = hp_->hc_count;
    const int64_t n_embd = hp_->n_embd;
    const int64_t hc_dim = hc * n_embd;
    const int64_t T      = hidden->ne[2];

    ggml_tensor * key   = ggml_mul_mat(ctx0, Wl(il, "ple_key.weight"),   emb);  // [hc_dim, T]
    ggml_tensor * value = ggml_mul_mat(ctx0, Wl(il, "ple_value.weight"), emb);  // [n_embd, T]

    // Both norms reduce within one hc stream but scale with a full hc_dim gamma.
    auto grouped_norm = [&](ggml_tensor * t, ggml_tensor * w) {
        t = ggml_reshape_3d(ctx0, t, n_embd, hc, T);
        t = ggml_rms_norm(ctx0, t, hp_->rms_eps);
        if (w->ne[0] == n_embd && w->ne[1] == hc) return ggml_mul(ctx0, t, w);   // adjacent norm+mul, fused on CUDA
        t = ggml_reshape_2d(ctx0, t, hc_dim, T);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, T);
    };

    key = grouped_norm(key, Wl(il, "ple_norm_key.weight"));
    ggml_tensor * query = grouped_norm(hidden, Wl(il, "ple_norm_query.weight"));

    // Per-stream dot product, then a signed square root before the sigmoid.
    ggml_tensor * sdot = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));       // [1, hc, T]
    sdot = ggml_scale(ctx0, sdot, 1.0f / sqrtf((float) n_embd));
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, sdot), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, sdot), mag));

    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, T);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, T, 1);
    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);

    ggml_tensor * normalized = grouped_norm(ggml_reshape_2d(ctx0, gated, hc_dim, T),
                                            Wl(il, "ple_norm_conv.weight"));
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, T);

    ggml_tensor * conv_out = ple_conv(normalized, il, T);
    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}


moe_routing graph_builder::moe_route(ggml_tensor * cur, int il) {
    assert_that(cur != nullptr && ctx0 != nullptr, "moe_route has an input and a context");
    assert_that(il >= 0 && (uint32_t) il < hp_->n_layer, "moe_route layer index in range");
    const int64_t n_expert      = hp_->n_expert;
    const int64_t n_expert_used = hp_->n_expert_used;
    const int64_t T             = cur->ne[1];

    ggml_tensor * logits = ggml_mul_mat(ctx0, Wl(il, "ffn_gate_inp.weight"), cur);   // [n_expert, T]
    ggml_tensor * probs  = ggml_soft_max(ctx0, logits);

    // llama.cpp's own node sequence -- argsort_top_k on the probabilities, the
    // probabilities reshaped for the gather, the gathered weights normalised by a
    // clamped sum -- is what ggml-cuda recognises and runs as ONE fused kernel
    // (softmax, sort, gather, normalise). The clamp is part of the pattern and a
    // no-op here: ten softmax probabilities of 512 sum far above 6e-5. Expanding
    // from the weights pins that order in the graph whatever is built next.
    ggml_tensor * selected = ggml_argsort_top_k(ctx0, probs, n_expert_used);            // [U, T], a view
    ggml_tensor * probs3   = ggml_reshape_3d(ctx0, probs, 1, n_expert, T);
    ggml_tensor * weights_ = ggml_get_rows(ctx0, probs3, selected);                      // [1, U, T]
    weights_ = ggml_reshape_2d(ctx0, weights_, n_expert_used, T);
    ggml_tensor * denom = ggml_clamp(ctx0, ggml_sum_rows(ctx0, weights_), 6.103515625e-5f, INFINITY);
    weights_ = ggml_div(ctx0, weights_, denom);
    if (gate_drop > 0.0f) {
        // Gate-threshold dropping: zero every normalised gate below the threshold
        // and renormalise what is left. The top gate of ten that sum to one is at
        // least 0.1, so a threshold below that always keeps at least one expert.
        ggml_tensor * keep = ggml_step(ctx0, ggml_scale_bias(ctx0, weights_, 1.0f, -gate_drop));   // 1 where w >= thr
        weights_ = ggml_mul(ctx0, weights_, keep);
        ggml_tensor * denom2 = ggml_clamp(ctx0, ggml_sum_rows(ctx0, weights_), 6.103515625e-5f, INFINITY);
        weights_ = ggml_div(ctx0, weights_, denom2);
    }
    weights_ = ggml_reshape_3d(ctx0, weights_, 1, n_expert_used, T);
    if (gf_) ggml_build_forward_expand(gf_, weights_);

    return { selected, weights_ };   // weights [1, U, T]
}

route_prediction graph_builder::moe_route_predict(ggml_tensor * res_hc, int il_next, int k, bool with_scores) {
    assert_that(res_hc != nullptr && k > 0, "moe_route_predict has a residual and ranks at least one expert");
    assert_that(il_next >= 0 && (uint32_t) il_next < hp_->n_layer, "moe_route_predict layer index in range");
    ggml_tensor * cur = hc_mix(res_hc, il_next, /*ffn=*/true, false).out;
    ggml_tensor * logits = ggml_mul_mat(ctx0, Wl(il_next, "ffn_gate_inp.weight"), cur);
    // Only the ranking matters, so the softmax and the renormalisation are
    // skipped. argsort rather than top_k because the result must be ordered:
    // the first n_expert_used entries are scored against the true routing.
    route_prediction p;
    p.ids = ggml_argsort_top_k(ctx0, logits, k);
    ggml_tensor * ids = p.ids;
    if (with_scores) {
        // The candidates' own logits, in the same order. Differences between
        // them are softmax-invariant, which is what the confidence gate uses.
        const int64_t n_expert = logits->ne[0], T = logits->ne[1];
        ggml_tensor * idc = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx0, ids);
        ggml_tensor * s = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, logits, 1, n_expert, T), idc);   // [1, k, T]
        p.scores = ggml_reshape_2d(ctx0, s, k, T);
    }
    return p;
}

mtp_post_out graph_builder::mtp_head_post(ggml_tensor * res, ggml_tensor * moe_out, ggml_tensor * inject, int il) {
    assert_that(res != nullptr && moe_out != nullptr && inject != nullptr, "mtp_head_post has its inputs");
    assert_that(il >= 0 && (uint32_t) il < hp_->n_layer, "mtp_head_post block index in range");
    mtp_post_out out;
    out.hres = hc_combine(res, moe_out, inject);
    // The head's own mixer collapses the streams and doubles as the output norm.
    ggml_tensor * o = hc_mix_w(out.hres, Wl(il, "nextn.hc_head_norm.weight"), Wl(il, "nextn.hc_head_down.weight"),
                               Wl(il, "nextn.hc_head_up.weight"), nullptr, false).out;
    out.logits = ggml_mul_mat(ctx0, W("output.weight"), o);                                  // the trunk's LM head, via alt
    return out;
}

mtp_pre_out graph_builder::mtp_head_pre(ggml_tensor * h, ggml_tensor * emb, ggml_tensor * inp_pos, ggml_tensor * kq_mask,
                                        const int sections[4], int il) {
    assert_that(h != nullptr && emb != nullptr, "mtp_head_pre has a residual and an embedding");
    assert_that(il >= 0 && (uint32_t) il < hp_->n_layer, "mtp_head_pre block index in range");
    const int64_t hc = hp_->hc_count, n_embd = hp_->n_embd, hc_dim = hc * n_embd;
    const int64_t T  = emb->ne[1];
    ggml_tensor * hnorm = Wl(il, "nextn.hnorm.weight");
    ggml_tensor * hn;
    if (hnorm->ne[0] == n_embd && hnorm->ne[1] == hc) {
        hn = ggml_mul(ctx0, ggml_rms_norm(ctx0, h, hp_->rms_eps), hnorm);
    } else {
        hn = ggml_reshape_2d(ctx0, ggml_rms_norm(ctx0, h, hp_->rms_eps), hc_dim, T);
        hn = ggml_reshape_3d(ctx0, ggml_mul(ctx0, hn, hnorm), n_embd, hc, T);
    }
    ggml_tensor * en = rms(emb, Wl(il, "nextn.enorm.weight"));
    en = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, en, n_embd, 1, T), n_embd, hc, T, 1);
    ggml_tensor * cat = ggml_concat(ctx0, en, hn, 0);
    ggml_tensor * res = ggml_mul_mat(ctx0, Wl(il, "nextn.eh_proj.weight"), ggml_reshape_2d(ctx0, cat, 2 * n_embd, hc * T));
    res = ggml_reshape_3d(ctx0, res, n_embd, hc, T);
    const hc_mixed ma = hc_mix(res, il, /*ffn=*/false, true);
    ggml_tensor * cur = sparse_attn(ma.out, inp_pos, kq_mask, sections, il, nullptr);
    res = hc_combine(res, cur, ma.inject);
    const hc_mixed mf = hc_mix(res, il, /*ffn=*/true, true);
    mtp_pre_out o;
    const moe_routing rt = moe_route(mf.out, il);
    o.sel = rt.sel; o.w = rt.w;
    o.sh = shared_expert(mf.out, il);
    o.res = res; o.cur = mf.out; o.inject = mf.inject;
    return o;
}

void graph_builder::mtp_head_kv(ggml_tensor * h, ggml_tensor * emb, ggml_tensor * inp_pos, const int sections[4], int il) {
    assert_that(h != nullptr && emb != nullptr && st_ != nullptr && gf_ != nullptr, "mtp_head_kv has inputs and a bound state/graph");
    assert_that(il >= 0 && (uint32_t) il < hp_->n_layer, "mtp_head_kv block index in range");
    const int64_t hc = hp_->hc_count, n_embd = hp_->n_embd, hc_dim = hc * n_embd;
    const int64_t T  = emb->ne[1];
    ggml_tensor * hnorm = Wl(il, "nextn.hnorm.weight");
    ggml_tensor * hn;
    if (hnorm->ne[0] == n_embd && hnorm->ne[1] == hc) {
        hn = ggml_mul(ctx0, ggml_rms_norm(ctx0, h, hp_->rms_eps), hnorm);
    } else {
        hn = ggml_reshape_2d(ctx0, ggml_rms_norm(ctx0, h, hp_->rms_eps), hc_dim, T);
        hn = ggml_reshape_3d(ctx0, ggml_mul(ctx0, hn, hnorm), n_embd, hc, T);
    }
    ggml_tensor * en = rms(emb, Wl(il, "nextn.enorm.weight"));
    en = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, en, n_embd, 1, T), n_embd, hc, T, 1);
    ggml_tensor * cat = ggml_concat(ctx0, en, hn, 0);
    ggml_tensor * res = ggml_mul_mat(ctx0, Wl(il, "nextn.eh_proj.weight"), ggml_reshape_2d(ctx0, cat, 2 * n_embd, hc * T));
    res = ggml_reshape_3d(ctx0, res, n_embd, hc, T);
    // The inject is built (and unused) as before: the context's tensor sequence stays the same.
    ggml_tensor * cur = hc_mix(res, il, /*ffn=*/false, true).out;
    // The same K/V path as sparse_attn, same ops in the same order.
    const int64_t hd = hp_->n_embd_head_k, nh_kv = hp_->n_head_kv, kv_dim = hd * nh_kv;
    int secs[4] = { sections[0], sections[1], sections[2], sections[3] };
    ggml_tensor * K = ggml_mul_mat(ctx0, Wl(il, "attn_k.weight"), cur);
    K = ggml_reshape_3d(ctx0, K, hd, nh_kv, T);
    K = rms(K, Wl(il, "attn_k_norm.weight"));
    ggml_tensor * V = ggml_mul_mat(ctx0, Wl(il, "attn_v.weight"), cur);
    V = ggml_reshape_3d(ctx0, V, hd, nh_kv, T);
    K = ggml_rope_multi(ctx0, K, inp_pos, nullptr, hp_->rope_dim, secs,
                        GGML_ROPE_TYPE_IMROPE, hp_->n_ctx_train, hp_->rope_freq_base,
                        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * kc = st_->k_cache(il);
    ggml_tensor * vc = st_->v_cache(il);
    ggml_build_forward_expand(gf_, ggml_cpy(ctx0,
            ggml_reshape_2d(ctx0, K, kv_dim, T),
            ggml_view_2d(ctx0, kc, kv_dim, T, ggml_row_size(kc->type, kv_dim), ggml_row_size(kc->type, kv_dim) * n_past_)));
    ggml_build_forward_expand(gf_, ggml_cpy(ctx0,
            ggml_reshape_2d(ctx0, V, kv_dim, T),
            ggml_view_2d(ctx0, vc, kv_dim, T, ggml_row_size(vc->type, kv_dim), ggml_row_size(vc->type, kv_dim) * n_past_)));
}

ggml_tensor * graph_builder::moe_resident(ggml_tensor * x, ggml_tensor * ids, ggml_tensor * w, int il, const weights * src) {
    assert_that(x != nullptr && ids != nullptr && w != nullptr, "moe_resident has an input, ids and weights");
    assert_that(src != nullptr, "moe_resident has an expert weight set");
    const int64_t n_embd = x->ne[0], T = x->ne[1], U = ids->ne[0];
    const std::string b = "blk." + std::to_string(il) + ".";
    ggml_tensor * gate_w = src->get(b + "ffn_gate_exps.weight");
    ggml_tensor * up_w   = src->get(b + "ffn_up_exps.weight");
    ggml_tensor * down_w = src->get(b + "ffn_down_exps.weight");
    ggml_tensor * x3   = ggml_reshape_3d(ctx0, x, n_embd, 1, T);
    ggml_tensor * gate = ggml_mul_mat_id(ctx0, gate_w, x3, ids);                 // [n_ff, U, T]
    ggml_tensor * up   = ggml_mul_mat_id(ctx0, up_w,   x3, ids);
    ggml_tensor * act  = ggml_swiglu_split(ctx0, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(ctx0, down_w, act, ids);                // [n_embd, U, T]
    ggml_tensor * wd   = ggml_mul(ctx0, down, w);
    ggml_tensor * acc  = ggml_cont(ctx0, ggml_view_2d(ctx0, wd, n_embd, T, wd->nb[2], 0));
    for (int64_t e = 1; e < U; e++)
        acc = ggml_add(ctx0, acc, ggml_view_2d(ctx0, wd, n_embd, T, wd->nb[2], (size_t) e * wd->nb[1]));
    return acc;
}

ggml_tensor * graph_builder::moe_apply(ggml_tensor * cur,
                                       ggml_tensor * const * gate, ggml_tensor * const * up,
                                       ggml_tensor * const * down, const float * w, int n_used) {
    assert_that(cur != nullptr && n_used >= 0, "moe_apply has an input and a non-negative expert count");
    assert_that(n_used == 0 || (gate && up && down && w), "moe_apply has expert tensors and gate weights");
    ggml_tensor * acc = nullptr;
    for (int e = 0; e < n_used; e++) {
        ggml_tensor * g = ggml_mul_mat(ctx0, gate[e], cur);
        ggml_tensor * u = ggml_mul_mat(ctx0, up[e],   cur);
        ggml_tensor * y = ggml_mul_mat(ctx0, down[e], ggml_swiglu_split(ctx0, g, u));

        y = ggml_scale(ctx0, y, w[e]);
        acc = acc ? ggml_add(ctx0, acc, y) : y;
    }

    return acc;
}

ggml_tensor * graph_builder::shared_expert(ggml_tensor * cur, int il) {
    assert_that(cur != nullptr, "shared_expert has an input");
    assert_that(il >= 0 && (uint32_t) il < hp_->n_layer, "shared_expert layer index in range");
    ggml_tensor * sg = ggml_mul_mat(ctx0, Wl(il, "ffn_gate_shexp.weight"), cur);
    ggml_tensor * su = ggml_mul_mat(ctx0, Wl(il, "ffn_up_shexp.weight"),   cur);
    ggml_tensor * sh = ggml_mul_mat(ctx0, Wl(il, "ffn_down_shexp.weight"), ggml_swiglu_split(ctx0, sg, su));
    // One scalar sigmoid gate per token, distinct from the routed gates.
    ggml_tensor * g = ggml_sigmoid(ctx0,
            ggml_mul_mat(ctx0, Wl(il, "ffn_gate_inp_shexp.weight"), cur));
    return ggml_mul(ctx0, sh, g);
}

} // namespace qwfn
