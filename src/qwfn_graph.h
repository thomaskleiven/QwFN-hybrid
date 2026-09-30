#pragma once
// qwen4exp graph construction.
//
// The residual stream is not a vector but hc=4 parallel streams of n_embd, carried
// as [n_embd, hc, T]. There is no output norm anywhere in the model: the final
// hyper-connection mixer serves as one. Each layer is
//
//     cur, inject = hc_mix(res, hc_attn_*)
//     cur         = deltanet(cur) or sparse_attn(cur)
//     res         = hc_combine(res, cur, inject)
//     cur, inject = hc_mix(res, hc_ffn_*)
//     cur         = moe(cur)
//     res         = hc_combine(res, cur, inject)
//
// with PLE injected once, before layer 1's first mix.

#include "qwfn_expert_cache.h"
#include "qwfn_hparams.h"
#include "qwfn_state.h"
#include "qwfn_weights.h"

#include <string>
#include <vector>

#include "ggml.h"

namespace qwfn {

// Host-built inputs for Qwen Sparse Attention. QSA scores whole blocks of
// `ratio` cells with one mean-pooled indexer key each, keeps the top
// indexer_top_k, and always keeps the incomplete tail -- hence a selection
// width of indexer_top_k + ratio - 1 (2051 for this model).
//
// Below that width every cell is selected and QSA is exactly dense attention,
// which is what makes it safe to validate against the dense path first.
// Results of the graph_builder pieces that produce more than one tensor (rule 9: structs, not
// T ** out-parameters).
struct hc_mixed {
    ggml_tensor * out    = nullptr;   // the streams collapsed, [n_embd, T]
    ggml_tensor * inject = nullptr;   // [hc, T] scatter weights for hc_combine (when asked for)
};
struct moe_routing {
    ggml_tensor * sel = nullptr;      // I32 expert ids [U, T]
    ggml_tensor * w   = nullptr;      // normalised gate weights [1, U, T]
};
struct route_prediction {
    ggml_tensor * ids    = nullptr;   // the k predicted expert ids, best first
    ggml_tensor * scores = nullptr;   // their logits, F32 [k, T] (when asked for)
};
struct mtp_pre_out {
    ggml_tensor * res = nullptr, * cur = nullptr, * inject = nullptr;
    ggml_tensor * sel = nullptr, * w = nullptr, * sh = nullptr;
};
struct mtp_post_out {
    ggml_tensor * logits = nullptr;
    ggml_tensor * hres   = nullptr;   // the head's wide residual after its MoE fold
};

struct qsa_inputs {
    ggml_tensor * cell_blk  = nullptr;  // I32 [n_kv]        cell -> block
    ggml_tensor * blk_cells = nullptr;  // I32 [ratio*n_blk] block -> member cells
    ggml_tensor * blk_pos   = nullptr;  // I32 [4*n_blk]     M-RoPE pos per block, section-major
    ggml_tensor * bias      = nullptr;  // F32 [n_blk, T]    per-query block visibility
    uint32_t      ratio     = 0;
    int64_t       n_blocks  = 0;
};

// Decode-time (T == 1) sparse attention with per-token work that does not grow
// with the context. Persistent tensors (engine-owned) plus five tiny per-token
// inputs; see graph_builder::sparse_attn_decode.
struct qsa_decode_inputs {
    ggml_tensor * pool_cache = nullptr;  // F16 [idx_dim, n_ctx/ratio], this layer: pooled+normed+roped block keys
    ggml_tensor * bias       = nullptr;  // F32 [n_ctx/ratio]: 0 whole block, 1e9 the tail block, -inf beyond
    ggml_tensor * blk_cells  = nullptr;  // I32 [ratio, n_ctx/ratio]: cell ids of each block (static)
    ggml_tensor * cell_pos   = nullptr;  // F32 [1, n_ctx]: cell index as a float (static)
    ggml_tensor * write_idx  = nullptr;  // I32 [1]: n_past, the row K/V/raw key go to
    ggml_tensor * member_idx = nullptr;  // I32 [ratio]: the cells of the block holding n_past
    ggml_tensor * blk_pos    = nullptr;  // I32 [4]: that block's M-RoPE positions, section-major
    ggml_tensor * blk_idx    = nullptr;  // I32 [1]: that block's index
    ggml_tensor * npast_f    = nullptr;  // F32 [1]: n_past, for the cell mask
    uint32_t ratio     = 0;
    int64_t  n_bucket  = 0;              // blocks scored (a multiple of 256; the graph shape)
    int64_t  k_blocks  = 0;              // blocks kept: ceil((indexer_top_k + ratio - 1) / ratio)
};

class graph_builder {
public:
    // Router gate threshold (see engine_config::gate_drop); applied by moe_route when > 0.
    float gate_drop = 0.0f;
    // `alt` is consulted for names `w` does not hold. The dense core lives in a
    // CUDA buffer while the PLE table stays mmap'd on the host, so a graph needs
    // to resolve tensors across both; ggml_backend_sched then places each op on
    // the device its tensors are on.
    graph_builder(ggml_context * ctx, const hparams * hp, const weights * w,
                  const weights * alt = nullptr)
        : ctx0(ctx), hp_(hp), w_(w), alt_(alt) {}

    // Optional; required by the layer builders that touch cache state.
    void bind(const state * st, ggml_cgraph * gf, int64_t n_past) {
        st_ = st; gf_ = gf; n_past_ = n_past;
    }

    // --- token mixers ----------------------------------------------------
    // 36 of 48 layers. Short causal conv, then the gated delta rule with a
    // constant-size recurrent state. Writes the new conv and recurrent state back.
    ggml_tensor * deltanet(ggml_tensor * cur, int il);

    // 12 of 48 layers. Lightning indexer picks 2051 cells, then GQA over them.
    // `qsa` may be null, in which case attention is dense over the whole cache.
    ggml_tensor * sparse_attn(ggml_tensor * cur, ggml_tensor * inp_pos,
                              ggml_tensor * kq_mask, const int sections[4], int il,
                              const qsa_inputs * qsa = nullptr);

    // The indexer: scores every block against the query and returns the cell
    // indices to keep, [width, T].
    ggml_tensor * qsa_top_k(ggml_tensor * cur, ggml_tensor * inp_pos,
                            const int sections[4], int il, const qsa_inputs & qsa);

    // Decode only. The raw key, K and V are written with ggml_set_rows at
    // write_idx; the block holding this token is pooled and written into
    // pool_cache (exact once the block is complete, forced visible by the bias
    // until then); the scores run over n_bucket cached block keys; the top
    // k_blocks blocks become cell ids; K/V rows for those cells are gathered
    // out of the caches and attention runs over them with a cell mask. Nothing
    // here touches all n_kv cells, and no shape depends on n_past, so the graph
    // replays within a bucket.
    // Two positions in one graph (the MTP verify step) run this twice; the
    // chain carries the caches as written by the first call so the second reads
    // through those writes (its block may contain the first position's key).
    struct qsa_chain { ggml_tensor * ic = nullptr, * kc = nullptr, * vc = nullptr, * pc = nullptr; };
    // The per-position projections of a T-position decode step (indexer k/q, attention q/gate/k/v
    // with their norms and rope), computed once for all T instead of once per chained call.
    // Only the cache writes, pooling, block scores, gather and attention depend on the positions
    // before them; these do not.
    struct qsa_proj { ggml_tensor * k_raw = nullptr, * q_idx = nullptr, * Q = nullptr, * gate = nullptr, * K = nullptr, * V = nullptr; };
    qsa_proj qsa_decode_proj(ggml_tensor * cur, ggml_tensor * inp_pos, const int sections[4], int il);
    // With `pre`, position k's projections come from it and the result is the gated attention
    // output before attn_output (the caller projects all positions at once, qsa_out_proj).
    ggml_tensor * sparse_attn_decode(ggml_tensor * cur, ggml_tensor * inp_pos, const int sections[4],
                                     int il, const qsa_decode_inputs & qd, qsa_chain * chain = nullptr,
                                     const qsa_proj * pre = nullptr, int64_t k = 0);
    ggml_tensor * qsa_out_proj(ggml_tensor * out, int il);

    // Rollback snapshots for a two-token graph: the DeltaNet block keeps K=2
    // state snapshots and copies the one-token-back state into `rs`, the conv
    // history one token back into `conv`, and the PLE conv likewise. The engine
    // restores them when a verified draft is rejected.
    // n_snap snapshots: slot s-1 holds the state s tokens back (s = 1..n_snap), so a
    // step of T tokens can be rolled back to any of its first T-1 positions.
    void set_rollback(ggml_tensor * rs, ggml_tensor * conv, int n_snap = 1) { rb_rs_ = rs; rb_conv_ = conv; rb_n_ = n_snap; }
    void set_rollback_ple(ggml_tensor * conv) { rb_ple_conv_ = conv; }

    // Fill pool_cache for blocks [0, n_whole) from the raw indexer cache -- after
    // a prefill, which does not maintain it. `blk_pos_all` is I32 [4*n_whole].
    void qsa_pool_rebuild(int il, const qsa_decode_inputs & qd, int64_t n_whole, ggml_tensor * blk_pos_all);

    // --- feed-forward ----------------------------------------------------
    // Router only: softmax over all 512, top-k, renormalise. `sel` is the
    // I32 expert ids and `w` the normalised gate weights, both graph outputs so
    // the host can read the selection back and fetch those experts.
    moe_routing moe_route(ggml_tensor * cur, int il);

    // Predict the NEXT layer's expert selection from this layer's
    // post-attention residual -- i.e. before this layer's MoE output exists, so
    // the reads can be issued while that MoE is still computing. It is layer
    // il_next's own ffn mixer and router applied to a residual that is missing
    // one MoE contribution: an approximation, and the accuracy is measured.
    // `k` may exceed n_expert_used: the extra, lower-ranked candidates are the
    // experts most likely to be swapped in by the true routing, and reading
    // them costs bandwidth the decode loop is not using. Sorted, best first.
    // with_scores: also the k logits, best first: F32 [k, T].
    route_prediction moe_route_predict(ggml_tensor * res_hc, int il_next, int k, bool with_scores = false);

    // When false, the layer builders leave the persistent state alone: no
    // recurrent-state, conv-history or PLE-conv write-back. The engine uses it
    // to run a layer's block speculatively on an approximate residual, purely to
    // predict that layer's routing; the exact pass that follows does the writes.
    void set_persist(bool p) { persist_ = p; }

    // GPU-only fusions that change the summation order (so the CPU path, which
    // is kept bit-exact against llama.cpp, does not take them): the mean over
    // the hc streams in hc_mix as one matmul with `hc_mean` (F32 [hc], every
    // entry 1/hc). Only applied to single-token graphs.
    void set_gpu_fusion(bool on, ggml_tensor * hc_mean) { gpu_fuse_ = on; hc_mean_ = hc_mean; }

    // MoE with the selected experts supplied explicitly, one mul_mat each,
    // pointing straight at cached blocks. Decode path: with a single token the
    // per-layer selection is exactly n_expert_used experts, so no gather or
    // remapping is needed and nothing has to be copied.
    // `w` are the normalised gate weights, already read back to the host with
    // the expert ids. Passing them as scalars rather than as views into a
    // device tensor keeps graph B free of strided cross-device reads, which is
    // both faster and avoids a scheduler edge case.
    // The always-on shared expert. Dense and small, so it belongs with the rest
    // of the dense core on the GPU rather than in the routed-expert graph.
    ggml_tensor * shared_expert(ggml_tensor * cur, int il);

    ggml_tensor * moe_apply(ggml_tensor * cur,
                            ggml_tensor * const * gate, ggml_tensor * const * up,
                            ggml_tensor * const * down, const float * w, int n_used);

    // --- PLE -------------------------------------------------------------
    // Injected once, before layer ple_layers[0]. `emb` is the gathered rows,
    // [ple_head_dim * ple_n_head, T] == [n_embd, T].
    ggml_tensor * ple(ggml_tensor * emb, ggml_tensor * hidden, int il);

    // Grouped RMSNorm over one stream, a low-rank gate over all of them, then the
    // streams are collapsed by their mean. With `with_inject` the [hc, T] scatter
    // weights used by hc_combine come back too; not for the final head mixer.
    //
    // il >= 0 selects blk.<il>.hc_{attn,ffn}_*; il < 0 selects the model-level
    // hc_head_* mixer. `ffn` picks which of the two per-layer modules to use.
    hc_mixed hc_mix(ggml_tensor * x, int il, bool ffn, bool with_inject);
    // The same mixer with its tensors given explicitly (the MTP head's own mixer).
    hc_mixed hc_mix_w(ggml_tensor * x, ggml_tensor * w_norm, ggml_tensor * w_down, ggml_tensor * w_up,
                      ggml_tensor * w_inject, bool with_inject);

    // The nextn/MTP draft head, wired as llama.cpp's graph_mtp: the trunk's wide
    // residual of token p (`h`, [n_embd, hc, T]) and the embedding of token p+1
    // (`emb`, [n_embd, T]), each normed with the head's gamma, concatenated per
    // stream and projected by eh_proj into a new wide residual; then one
    // trunk-style block at position p (attention over the head's own KV, dense
    // as in the reference; the head's MoE), the head's mixer, the trunk's LM
    // head, giving logits [n_vocab, T] for token p+2. `il` is the block's index
    // in the MTP file; its tensors come from `w`, the LM head through `alt`.
    // Built in two halves around its routed MoE, whose experts live in host
    // memory: the first half returns the wide residual after the attention
    // block, the FFN input, its inject, the routing and the shared expert; the
    // second folds a routed partial in and produces the logits.
    mtp_pre_out mtp_head_pre(ggml_tensor * h, ggml_tensor * emb, ggml_tensor * inp_pos, ggml_tensor * kq_mask,
                             const int sections[4], int il);
    // hres: the head's wide residual after its MoE fold -- what stands in for the
    // trunk's residual when the head drafts again from its own draft.
    mtp_post_out mtp_head_post(ggml_tensor * res, ggml_tensor * moe_out, ggml_tensor * inject, int il);
    // The head's KV rows for n prompt positions and nothing else: eh_proj, the
    // attention mixer and the K/V projections, written into the head's cache at
    // n_past. No mask, no attention output, no MoE, no LM head -- what a streamed
    // prefill needs so the head can draft after it.
    void mtp_head_kv(ggml_tensor * h, ggml_tensor * emb, ggml_tensor * inp_pos, const int sections[4], int il);
    // The head's routed experts as three resident [.., .., n_expert] tensors
    // (host or device), applied to x [n_embd, T] with ids [U, T] and weights
    // [1, U, T]: the trunk's summation order.
    ggml_tensor * moe_resident(ggml_tensor * x, ggml_tensor * ids, ggml_tensor * w, int il, const weights * src);

    // residual + broadcast(block_out) * 2*sigmoid(inject/hc).
    // The 2*sigmoid centres the scatter weights on 1, so a zero injection leaves
    // a plain residual add.
    ggml_tensor * hc_combine(ggml_tensor * residual, ggml_tensor * block_out, ggml_tensor * inject);

    ggml_context * ctx0 = nullptr;

    // Read a conv history out of its state row, append the new input, and write
    // the tail back. Returns the padded input for ggml_ssm_conv.
    ggml_tensor * conv_with_history(ggml_tensor * state_row, ggml_tensor * x,
                                    int64_t hist, int64_t channels, ggml_tensor * rb_row = nullptr);

private:
    ggml_tensor * W(const std::string & name) const;
    ggml_tensor * Wl(int il, const char * suffix) const;
    ggml_tensor * rms(ggml_tensor * x, ggml_tensor * w) const;
    // The model's interleaved M-RoPE with its fixed extension parameters.
    ggml_tensor * rope(ggml_tensor * x, ggml_tensor * pos, int * secs) const;

    // deltanet pieces, in graph order: the decay gate, the conv + q/k/v split, the state write-back.
    struct dn_qkv { ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr; };
    ggml_tensor * deltanet_decay(ggml_tensor * cur, int il, int64_t n_v_heads, int64_t T);
    dn_qkv deltanet_conv_qkv(ggml_tensor * qkv, int il, int64_t T);
    void deltanet_write_state(ggml_tensor * result, int il, int64_t T, int K);

    // sparse_attn / qsa_top_k pieces.
    ggml_tensor * idx_query(ggml_tensor * cur, ggml_tensor * inp_pos, int * secs, int il, int64_t T);
    void kv_append(ggml_tensor * K, ggml_tensor * V, int il, int64_t T);
    ggml_tensor * qsa_mask(ggml_tensor * top_k, ggml_tensor * kq_mask, int64_t n_kv, int64_t T);

    // sparse_attn_decode pieces, in graph order. The returned chain holds ic/pc (index) or kc/vc (kv).
    qsa_chain qsa_dec_index(ggml_tensor * cur, int il, const qsa_decode_inputs & qd, const qsa_chain * chain,
                            const qsa_proj * pre, int64_t kpos, int * secs);
    ggml_tensor * qsa_dec_cells(ggml_tensor * cur, ggml_tensor * inp_pos, int il, const qsa_decode_inputs & qd,
                                const qsa_proj * pre, int64_t kpos, int * secs, ggml_tensor * pc_w);
    qsa_proj qsa_dec_qkv(ggml_tensor * cur, ggml_tensor * inp_pos, int il, const qsa_proj * pre, int64_t kpos, int * secs);
    qsa_chain qsa_dec_kv_write(const qsa_proj & p, int il, const qsa_decode_inputs & qd, const qsa_chain * chain,
                               bool pre);
    ggml_tensor * qsa_gather(ggml_tensor * cache_w, ggml_tensor * cells, ggml_tensor * gcells, bool raw,
                             size_t row, int64_t ncp);

    // ple's depthwise causal conv over the normalised gated values.
    ggml_tensor * ple_conv(ggml_tensor * normalized, int il, int64_t T);

    const hparams * hp_  = nullptr;
    const weights * w_   = nullptr;
    const weights * alt_ = nullptr;
    const state *   st_  = nullptr;
    ggml_cgraph *   gf_ = nullptr;
    int64_t         n_past_ = 0;
    bool            persist_ = true;
    bool            gpu_fuse_ = false;
    ggml_tensor *   hc_mean_ = nullptr;
    ggml_tensor *   rb_rs_ = nullptr, * rb_conv_ = nullptr, * rb_ple_conv_ = nullptr;
    int             rb_n_ = 1;
};

} // namespace qwfn
