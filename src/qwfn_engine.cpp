#include "qwfn_engine.h"
#include "qwfn_check.h"
#include <new>
#include "ggml-impl.h"   // struct ggml_cgraph, for its uid (the cached decode graphs)
#include "qwfn_ple.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <numeric>
#include <span>
#include <sstream>

namespace qwfn {


static const char * EXP_SUFFIX[EXPERT_NPARTS] = {
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"
};

static uint16_t f16_of(float f) {
    ggml_fp16_t h = ggml_fp32_to_fp16(f);
    uint16_t o; memcpy(&o, &h, 2); return o;
}

static double secs_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// A no_alloc context holding a graph of up to n nodes, its own memory.
// rule 3 deviation: ggml builds each graph in a context it allocates (per step on the decode path
// for the graphs not cached), see docs/CODING_RULES.md; allocation failure is asserted.
engine::graph_ctx engine::new_graph_ctx(size_t n) {
    assert_that(n > 0, "a graph has room for nodes");
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * n + ggml_graph_overhead_custom(n, false);
    p.no_alloc = true;
    engine::graph_ctx gc;
    gc.ctx = ggml_init(p); assert_that(gc.ctx != nullptr, "ggml_init: a graph context");
    gc.gf  = ggml_new_graph_custom(gc.ctx, n, false);
    assert_that(gc.ctx != nullptr && gc.gf != nullptr, "graph context allocated");
    return gc;
}

// Rows [off, off + n) of a row-major tensor, and of the wide residual.
static ggml_tensor * view_rows(ggml_context * c, ggml_tensor * t, int64_t off, int64_t n) {
    return ggml_view_2d(c, t, t->ne[0], n, t->nb[1], (size_t) off * t->nb[1]);
}
static ggml_tensor * view_res(ggml_context * c, ggml_tensor * res, int64_t off, int64_t n) {
    return ggml_view_3d(c, res, res->ne[0], res->ne[1], n, res->nb[1], res->nb[2], (size_t) off * res->nb[2]);
}

// M-RoPE positions are section-major: all t, then all h, then all w, then 0.
// `pos` is the caller's reusable host scratch.
static void set_positions(ggml_tensor * dst, int64_t first, int64_t n, std::vector<int32_t> & pos) {
    assert_that(dst != nullptr && n >= 1, "positions for at least one token");
    assert_that((size_t) n * 4 * sizeof(int32_t) <= ggml_nbytes(dst), "position tensor holds n tokens");
    pos.assign((size_t) 4 * n, 0);
    for (int64_t i = 0; i < n; i++) pos[i] = pos[n + i] = pos[2 * n + i] = (int32_t) (first + i);
    ggml_backend_tensor_set(dst, pos.data(), 0, pos.size() * 4);
}

// Decode MoE state of one layer.
struct engine::moe_layer {
    uint32_t il = 0;
    bool     packed = false, two_pass = false, prefetched = false;
    int64_t  n_u = 0;                 // distinct experts of the step, in ids_
    std::vector<uint8_t> rdy;
    std::vector<int> on_gpu, late_gpu, ready_cpu, late_cpu, all_cpu;
    std::vector<int> pos_in_all;      // index into all_cpu (selection order), per expert
    size_t nfl = 0;
    std::vector<float> acc;           // the CPU experts' sum, [n_embd, T]

    // A fresh layer, keeping the vectors' capacity (engine::ml_ is reused every layer).
    void reset(uint32_t layer, bool pk) {
        il = layer; packed = pk; two_pass = false; prefetched = false; n_u = 0; nfl = 0;
        rdy.clear(); on_gpu.clear(); late_gpu.clear(); ready_cpu.clear(); late_cpu.clear(); all_cpu.clear();
        pos_in_all.clear(); acc.clear();
    }
    void reserve(size_t n_ids, size_t n_acc) {
        rdy.reserve(n_ids); on_gpu.reserve(n_ids); late_gpu.reserve(n_ids); ready_cpu.reserve(n_ids);
        late_cpu.reserve(n_ids); all_cpu.reserve(n_ids); pos_in_all.reserve(n_ids); acc.reserve(n_acc);
    }
};

// Per-term id matrices of the cache-served batch. ggml's MoE gather assumes an
// expert appears at most once in a token's list (a single shared "dummy" for
// every absent slot faulted inside MMQ), so each token's list is its real
// experts of this term followed by DISTINCT unused ones at weight 0, and the
// list width is the widest any token needs.
struct engine::cb_term { std::vector<int32_t> slots;  /* per chunk entry: slot or -1 */
                         std::vector<int32_t> ids; std::vector<float> w; int width = 0;
                         void reset() { slots.clear(); ids.clear(); w.clear(); width = 0; } };

// Cache-served batch state of one layer.
struct engine::cbatch_layer {
    uint32_t il = 0;
    std::vector<uint32_t> uniq;       // distinct experts of the batch, ascending
    size_t   bounce_bytes = 0;
    uint8_t * bounce = nullptr;
    uint32_t bounce_slots = 0, chunk = 1;
    bool     pipe = false;
    size_t   half_bytes = 0;
    size_t   slice[EXPERT_NPARTS] = { 0, 0, 0 };
    tier_view gt, rt;
    std::vector<expert_handle> chs[2];
    std::vector<int32_t> uniq_pos;
    ggml_context * pend_ctx = nullptr;   // the previous chunk's graph, running while the next reads land
    cb_term tt, ts;                      // tier term, scratch term
    // Scratch of the setup and of each chunk.
    std::vector<uint32_t> cnt;           // routed pairs per expert
    std::vector<int32_t>  mem_t, mem_s;  // the chunk's tier / scratch members
    std::vector<int32_t>  per_tok;       // [T, U]: chunk indices per token, per_tok_n[t] of them
    std::vector<int32_t>  per_tok_n;

    // A fresh layer, keeping the vectors' capacity (engine::cl_ is reused every layer).
    void reset(uint32_t layer) {
        assert_that(layer < (1u << 16), "cbatch_layer: a layer index");
        il = layer; uniq.clear(); bounce_bytes = 0; bounce = nullptr; bounce_slots = 0; chunk = 1;
        pipe = false; half_bytes = 0;
        for (size_t & b : slice) b = 0;
        gt = tier_view{}; rt = tier_view{};
        chs[0].clear(); chs[1].clear(); uniq_pos.clear(); pend_ctx = nullptr;
        tt.reset(); ts.reset();
        cnt.clear(); mem_t.clear(); mem_s.clear(); per_tok.clear(); per_tok_n.clear();
        assert_that(uniq.empty() && chs[0].empty() && tt.width == 0, "cbatch_layer: fresh");
    }
};

// Every per-step host vector at its largest (rule 3), so a step only clears and refills.
// Capacity is all that is set here: what a step computes is unchanged.
void engine::init_step_scratch() {
    const size_t U = hp_.n_expert_used, E = hp_.n_expert, n_embd = hp_.n_embd, B = cfg_.n_batch;
    const size_t MAXT = 1 + MTP_MAX_DRAFTS;              // positions of the longest decode (verify) step
    const size_t UD = U * MAXT;                          // distinct experts of a decode step, at most
    const size_t CB_T = std::min<size_t>(std::max<uint32_t>(cfg_.prefill_decode_max, 1), B);   // a cache-served batch
    const size_t CHUNK = std::max<size_t>(1, std::min<size_t>(scr_slots_, cfg_.cache_batch_chunk));
    assert_that(U >= 1 && E >= U && B >= 1, "init_step_scratch: routing and batch sizes");
    assert_that(!ml_ && !cl_, "init_step_scratch: sized once");
    ml_ = std::make_unique<moe_layer>(); ml_->reserve(UD, n_embd * MAXT);
    cl_ = std::make_unique<cbatch_layer>();
    cl_->uniq.reserve(E); cl_->uniq_pos.reserve(E); cl_->cnt.reserve(E);
    for (auto & ch : cl_->chs) ch.reserve(CHUNK);
    cl_->mem_t.reserve(CHUNK); cl_->mem_s.reserve(CHUNK); cl_->tt.slots.reserve(CHUNK); cl_->ts.slots.reserve(CHUNK);
    for (cb_term * tm : { &cl_->tt, &cl_->ts }) { tm->ids.reserve(U * CB_T); tm->w.reserve(U * CB_T); }
    cl_->per_tok.reserve(U * CB_T); cl_->per_tok_n.reserve(CB_T);
    eh_.reserve(UD);
    pos_host_.reserve(4 * B);
    ple_rows_host_.reserve((size_t) hp_.ple_n_head() * B);
    vt_slot_.reserve(E); vt_mask_.reserve(E);
    for (auto * v : { &sc_sid_, &sc_pk_, &sc_pj_, &sc_gids_ }) v->reserve(std::max(UD * MAXT, (size_t) late_w_ * MAXT));
    for (auto * v : { &sc_sw_, &sc_gw_, &sc_ww_ }) v->reserve(std::max(UD * MAXT, (size_t) late_w_ * MAXT));
    sc_hot_.reserve(UD); sc_cold_.reserve(UD);
    sc_tg_.reserve(UD); sc_tu_.reserve(UD); sc_td_.reserve(UD);
    pf1_.reserve(QWFN_SPEC_MAX * MAXT); pf2_.reserve(QWFN_SPEC_MAX);
    pred_.reserve(QWFN_SPEC_MAX * MAXT); spec_scores_.reserve(QWFN_SPEC_MAX * MAXT); pred_margin_.reserve(QWFN_SPEC_MAX * MAXT);
    pred2_a_.reserve(QWFN_SPEC_MAX); pred2_b_.reserve(QWFN_SPEC_MAX);
    if (mtp_on_) {
        mtp_mask_host_.reserve(t_mtp_mask_ ? (size_t) ggml_nelements(t_mtp_mask_) : 0);
        mtp_ids_host_.reserve(U * B); mtp_w_host_.reserve(U * B); mtp_top_.reserve(B);
        mtp_keep_.reserve((size_t) n_vocab_);
    }
}


engine::engine() = default;   // here, not in the header: ml_/cl_ point at types private to this file

// rule 5 deviation: a destructor also runs after a failed init, so it must not abort.
engine::~engine() {
    if (moe_ctx_) {   // an async MoE graph may still be in flight
        ggml_backend_synchronize(w_.backend());
        ggml_free(moe_ctx_);
    }
    for (auto & lg : gA_) {
        if (lg.ga)  ggml_gallocr_free(lg.ga);
        if (lg.ctx) ggml_free(lg.ctx);
    }
    for (auto & mg : gM_) {
        if (mg.ga)  ggml_gallocr_free(mg.ga);
        if (mg.ctx) ggml_free(mg.ctx);
    }
    if (galloc_gpu_) ggml_gallocr_free(galloc_gpu_);
    if (galloc_cpu_) ggml_gallocr_free(galloc_cpu_);
    if (galloc_moe_) ggml_gallocr_free(galloc_moe_);
    if (wbuf_) ggml_backend_buffer_free(wbuf_);
    if (hbuf_) ggml_backend_buffer_free(hbuf_);
    if (galloc_pf_) ggml_gallocr_free(galloc_pf_);
    if (galloc_pf_dyn_) ggml_gallocr_free(galloc_pf_dyn_);
    if (vbuf_) ggml_backend_buffer_free(vbuf_);
    if (vctx_) ggml_free(vctx_);
    if (pwbuf_) ggml_backend_buffer_free(pwbuf_);
    if (pwctx_) ggml_free(pwctx_);
    if (qbuf_) ggml_backend_buffer_free(qbuf_);
    if (qctx_) ggml_free(qctx_);
    if (pbuf_) ggml_backend_buffer_free(pbuf_);
    if (pctx_) ggml_free(pctx_);
    if (scr_buf_) ggml_backend_buffer_free(scr_buf_);
    if (scr_ctx_) ggml_free(scr_ctx_);
    if (wctx_) ggml_free(wctx_);
    if (hctx_) ggml_free(hctx_);
}

// ggml's default logger prints its DEBUG lines ("CUDA graph warmup complete",
// "CUDA Graph id N reused" on every replay of a cached graph): thousands a
// request in a server log. Keep INFO and above.
static void qwfn_ggml_log(enum ggml_log_level level, const char * text, void * /*user*/) {
    assert_that(text != nullptr, "ggml log line");
    assert_that(level >= GGML_LOG_LEVEL_NONE && level <= GGML_LOG_LEVEL_CONT, "ggml log level in range");
    if (level == GGML_LOG_LEVEL_DEBUG) return;
    fputs(text, stderr);
}

void engine::set_n_threads(int n) {
    assert_that(mi_ != nullptr, "set_n_threads: engine initialised");
    n_threads_ = std::max(1, n);
    assert_that(n_threads_ >= 1, "at least one CPU thread");
    wh_.set_n_threads(n_threads_);
    if (w_.on_gpu()) w_.set_n_threads(n_threads_);   // --cpu: wh_ and w_ share cores; one pool (wh_'s) is enough
    else { w_.set_n_threads_nopool(n_threads_); }
}

// Every expert slice of `mi` starts on a page and strides by page multiples.
static bool page_strides(const model_index * mi) {
    assert_that(QWFN_DIO_PAGE % 512 == 0, "the page is a whole number of sectors");
    if (!mi) return true;
    assert_that(mi->hp().n_layer > 0, "a model index has layers");
    assert_that(mi->hp().n_layer < 65536, "the layer count fits the prefill progress word");
    for (uint32_t il = 0; il < mi->hp().n_layer; il++)
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const byte_range a = mi->expert_range(il, 0, (expert_part) q);
            const byte_range b = mi->expert_range(il, 1, (expert_part) q);
            // The first slice's own offset counts too: a page-multiple stride off a
            // 32-byte GGUF tensor offset would make every in-place O_DIRECT read fail.
            if (a.valid() && (a.offset % QWFN_DIO_PAGE) != 0) return false;
            if (a.valid() && b.valid() && ((b.offset - a.offset) % QWFN_DIO_PAGE) != 0) return false;
        }
    return true;
}

bool engine::init(const model_index * hot, const model_index * cold,
                  const engine_config & cfg, const std::string & backend_dir, std::string & err) {
    assert_that(hot != nullptr, "init: a model index");
    assert_that(mi_ == nullptr, "init: called once per engine");
    ggml_log_set(qwfn_ggml_log, nullptr);
    mi_  = hot;
    cfg_ = cfg;
    hp_  = hot->hp();
    n_vocab_ = hp_.n_vocab;
    if (cfg.indexer_top_k) {
        fprintf(stderr, "[qwfn] indexer top-k %u -> %u\n", hp_.idx_top_k, cfg.indexer_top_k);
        hp_.idx_top_k = cfg.indexer_top_k;
    }
    // The direct I/O layout (see qwfn_io.h): page-aligned slots when every expert
    // slice stride in the file is a page multiple, so reads land in place; else
    // 512-byte slots behind page-aligned bounce reads. Decided before any tier
    // or staging is laid out.
    {
        const bool page = page_strides(hot) && page_strides(cold);
        set_dio_align(page ? QWFN_DIO_PAGE : 512);
        fprintf(stderr, "[qwfn] direct I/O: %s\n", page ? "page-aligned slots, reads land in place"
                                                     : "512-byte slots, page-aligned reads through a bounce buffer (slice strides are not page multiples)");
    }

    if (!w_.init(hot, cfg.use_gpu, backend_dir, err)) return false;
    if (!w_.declare_dense_core(err)) return false;
    if (!w_.commit(err)) return false;

    if (!wh_.init(hot, /*prefer_gpu=*/false, backend_dir, err)) return false;

    if (!init_mtp(backend_dir, err)) return false;
    if (!init_host_maps(err)) return false;
    if (!init_state(err)) return false;
    if (!init_work_set(err)) return false;
    if (!init_host_work(err)) return false;
    init_pinned_staging();
    init_qsa_decode();
    init_cbatch_scratch();
    if (!init_expert_cache(hot, cold, err)) return false;
    return init_graphs(hot, err);
}

// ---- MTP draft head (experiment): the nextn block, resident ---------------
bool engine::init_mtp(const std::string & backend_dir, std::string & err) {
    const engine_config & cfg = cfg_;
    assert_that(mi_ != nullptr, "init_mtp: after the trunk's index");
    assert_that(!mtp_on_, "init_mtp: the head is loaded once");
    if (!cfg.mtp_path.empty() && cfg.use_gpu && cfg.skip_miss)
        fprintf(stderr, "[qwfn] mtp: draft head not loaded: a verified pair and --skip-miss do not combine (skip-miss "
                        "computes a token without the experts still on disk, one token at a time); drafts off\n");
    if (!cfg.mtp_path.empty() && cfg.use_gpu && !cfg.skip_miss) {
        if (!mi_mtp_.load(cfg.mtp_path, err)) return false;
        if (!wm_.init(&mi_mtp_, /*prefer_gpu=*/true, backend_dir, err)) return false;
        if (!wmh_.init(&mi_mtp_, /*prefer_gpu=*/false, backend_dir, err)) return false;
        for (const auto & kv : mi_mtp_.tensors()) {
            const bool exps = kv.first.find("_exps.weight") != std::string::npos;
            weights & dst = exps ? wmh_ : wm_;
            if (!dst.declare(kv.first)) { err = "mtp: failed to declare " + kv.first; return false; }
        }
        if (!wm_.commit(err)) return false;
        if (!wmh_.commit(err)) return false;
        hpm_ = mi_mtp_.hp();
        hpm_.full_attention_interval = hpm_.n_layer;   // in this index only the last block, the nextn block, is attention
        hpm_.ssm_dt_rank = 1;                           // the index's 48 trunk slots carry no state here; keep theirs tiny
        hpm_.hc_inject_prescaled = false;               // its inject weights are Q8_0 and are not folded
        state_config scm; scm.n_ctx = cfg.n_ctx; scm.type_k = cfg.type_k; scm.type_v = cfg.type_v;
        if (!st_mtp_.init(&hpm_, scm, w_.buft(), err)) return false;
        mtp_on_ = true;
        fprintf(stderr, "[qwfn] mtp: nextn block %u of %s: %.2f GB on %s%s\n",
                hpm_.n_layer - 1, cfg.mtp_path.c_str(), wm_.bytes() / 1e9, wm_.dev_name(),
                (", its " + std::to_string((long long) (wmh_.bytes() / 1e6)) + " MB of experts in host memory (computed on the CPU per draft)").c_str());
    }
    return true;
}

bool engine::init_host_maps(std::string & err) {
    assert_that(mi_ != nullptr, "init_host_maps: after the trunk's index");
    assert_that(!have_expert_map_, "init_host_maps: mapped once");
    if (!wh_.map_shards(err)) return false;
    if (!wh_.declare_mapped("per_layer_token_embd.weight")) {
        err = "per_layer_token_embd.weight missing"; return false;
    }
    // The token embedding table (0.68 GB at Q8_0) is only ever gathered one
    // row per token, so it stays in the host mapping like the PLE table and
    // the rows are uploaded; that is 0.68 GB more of VRAM expert tier.
    if (!wh_.declare_mapped("token_embd.weight")) { err = "token_embd.weight missing"; return false; }
    // Prefill reads experts straight from the mapping; the pages it touches are
    // whatever the routing asks for, and the cache is left alone for decode.
    for (uint32_t il = 0; il < hp_.n_layer; il++)
        for (int q = 0; q < EXPERT_NPARTS; q++)
            if (!wh_.declare_mapped("blk." + std::to_string(il) + "." + EXP_SUFFIX[q])) {
                err = "expert tensor missing on layer " + std::to_string(il); return false;
            }
    have_expert_map_ = true;
    set_n_threads(cfg_.n_threads);
    return true;
}

// State first: the KV cache, indexer keys and recurrent state are not
// optional, whereas the VRAM expert tier is. Allocating the tier first let
// it take the memory the state needed and fail the whole engine.
bool engine::init_state(std::string & err) {
    const engine_config & cfg = cfg_;
    assert_that(have_expert_map_, "init_state: after the host maps");
    assert_that(rbctx_ == nullptr, "init_state: rollback snapshots allocated once");
    state_config sc;
    sc.n_ctx  = cfg.n_ctx;
    sc.type_k = cfg.type_k;
    sc.type_v = cfg.type_v;
    sc.idx_host = cfg.idx_host;
    sc.kv_host  = cfg.kv_host;
    if (!st_.init(&hp_, sc, w_.buft(), err)) return false;
    if (mtp_on_ || cfg.rollback_snapshots) {
        // Rollback snapshots for the MTP verify: per DeltaNet layer the state and
        // conv history after the FIRST token of a pair, and the PLE conv (117 MB).
        ggml_init_params rp{}; rp.mem_size = ggml_tensor_overhead() * (2 * hp_.n_layer + 4); rp.no_alloc = true;
        rbctx_ = ggml_init(rp); assert_that(rbctx_ != nullptr, "ggml_init: rbctx_");
        rb_rs_.assign(hp_.n_layer, nullptr); rb_conv_.assign(hp_.n_layer, nullptr);
        const int64_t hv = hp_.ssm_d_state, nvh = hp_.ssm_dt_rank;
        const int64_t conv_dim = 2 * (int64_t) hp_.ssm_n_group * hp_.ssm_d_state + (int64_t) hp_.ssm_dt_rank * hp_.ssm_d_state;
        // One snapshot per draft the step may carry: slot s-1 is the state s tokens back.
        rb_nsnap_ = (int) std::max<uint32_t>(1, std::min<uint32_t>(cfg.mtp_drafts, (uint32_t) MTP_MAX_DRAFTS));
        for (uint32_t il = 0; il < hp_.n_layer; il++) {
            if (hp_.is_attn_layer(il)) continue;
            rb_rs_[il]   = ggml_new_tensor_4d(rbctx_, GGML_TYPE_F32, hv, hv, nvh, rb_nsnap_);
            rb_conv_[il] = ggml_new_tensor_3d(rbctx_, GGML_TYPE_F32, hp_.ssm_d_conv - 1, conv_dim, rb_nsnap_);
        }
        rb_ple_conv_ = ggml_new_tensor_3d(rbctx_, GGML_TYPE_F32, (int64_t) (hp_.ple_conv_kernel - 1) * hp_.ple_ngram_size, (int64_t) hp_.hc_count * hp_.n_embd, rb_nsnap_);
        rbbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(rbctx_, w_.buft());
        if (!rbbuf_) { err = "no device memory for the rollback snapshots"; return false; }
    }
    return true;
}

// Decode and the short-prompt batch never see more than prefill_decode_max
// tokens at once; the n_batch-sized set exists only during a prefill.
int64_t engine::decode_rows() const {
    const int64_t B = cfg_.n_batch;
    return std::max<int64_t>(1, std::min<int64_t>(B, std::max<uint32_t>(1, cfg_.prefill_decode_max)));
}

// The persistent work buffers are sized by n_batch and are not optional, so
// they go in before the VRAM expert tier -- same reasoning as the state
// above. At ubatch 2048 they are ~420 MB; leaving them until after the tier
// meant the tier had already taken that memory.
bool engine::init_work_set(std::string & err) {
    assert_that(wctx_ == nullptr, "init_work_set: allocated once");
    assert_that(hp_.n_embd > 0 && hp_.hc_count > 0, "init_work_set: model dimensions");
    init_work_tensors();
    // Shape the hyper-connection and PLE norm gammas [hc*n_embd] as [n_embd, hc]
    // -- metadata only, same bytes -- so hc_mix and ple can multiply right after
    // the RMSNorm without a reshape node between them: ggml-cuda fuses
    // (rms_norm, mul) only when the two are adjacent.
    init_gammas();
    init_pack_and_spec();
    init_route_tensors();
    wbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(wctx_, w_.buft());
    if (!wbuf_) { err = "failed to allocate engine work buffer"; return false; }
    { std::vector<float> m(hp_.hc_count, 1.0f / (float) hp_.hc_count); ggml_backend_tensor_set(t_hcmean_, m.data(), 0, m.size() * sizeof(float)); }
    if (mtp_on_ && !init_mtp_work(err)) return false;
    if (t_rscale_) { const float one = 1.0f; ggml_backend_tensor_set(t_rscale_, &one, 0, 4); }
    save_work_set(dec_ws_);
    return true;
}

void engine::init_work_tensors() {
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count, U = hp_.n_expert_used, Bd = decode_rows();
    assert_that(wctx_ == nullptr, "init_work_tensors: one work context");
    assert_that(Bd >= 1 && U >= 1, "init_work_tensors: decode rows and routed experts");
    ggml_init_params wp{}; wp.mem_size = ggml_tensor_overhead() * 64; wp.no_alloc = true;
    wctx_ = ggml_init(wp); assert_that(wctx_ != nullptr, "ggml_init: wctx_");
    res_[0]   = ggml_new_tensor_3d(wctx_, GGML_TYPE_F32, n_embd, hc, Bd);
    res_[1]   = ggml_new_tensor_3d(wctx_, GGML_TYPE_F32, n_embd, hc, Bd);
    t_cur_    = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
    t_emb_    = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
    t_sh_     = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
    t_pg_     = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
    t_pc_     = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
    t_ple_    = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
    t_inject_ = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, hc, Bd);
    t_sel_    = ggml_new_tensor_2d(wctx_, GGML_TYPE_I32, U, Bd);
    t_selnext_= ggml_new_tensor_2d(wctx_, GGML_TYPE_I32, QWFN_SPEC_MAX, Bd);
    t_selnext2_= ggml_new_tensor_2d(wctx_, GGML_TYPE_I32, QWFN_SPEC_MAX, Bd);
    t_specscore_= ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, QWFN_SPEC_MAX, Bd);
    t_hcmean_   = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, hc, 1);   // (1/hc, ...): the stream mean as a matmul
    if (mtp_on_) {
        t_hlast_   = ggml_new_tensor_3d(wctx_, GGML_TYPE_F32, n_embd, hc, Bd);
        t_mtp_pos_ = ggml_new_tensor_1d(wctx_, GGML_TYPE_I32, 4 * Bd);
        t_mtp_emb_ = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
        t_mtp_mask_ = ggml_new_tensor_1d(wctx_, GGML_TYPE_F16, 2 * ((int64_t) cfg_.n_ctx + 2));
    }
}

static void reshape_gamma_in(weights & wt, const std::string & name, int64_t n_embd, int64_t hc) {
    assert_that(n_embd > 0 && hc > 0, "reshape_gamma: model dimensions");
    assert_that(!name.empty(), "reshape_gamma: a tensor name");
    ggml_tensor * t = wt.get(name);
    if (!t || ggml_nelements(t) != (int64_t) hc * n_embd || t->ne[1] == (int64_t) hc) return;
    if (ggml_blck_size(t->type) != 1) return;
    t->ne[0] = n_embd; t->ne[1] = hc; t->ne[2] = 1; t->ne[3] = 1;
    t->nb[1] = t->nb[0] * n_embd; t->nb[2] = t->nb[1] * hc; t->nb[3] = t->nb[2];
}

void engine::init_gammas() {
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count;
    assert_that(n_embd > 0 && hc > 0, "init_gammas: model dimensions");
    assert_that(!hp_.hc_inject_prescaled, "init_gammas: the inject weights are folded once");
    if (mtp_on_) {
        const std::string b = "blk." + std::to_string(hpm_.n_layer - 1) + ".";
        for (const char * n : { "hc_attn_norm", "hc_ffn_norm", "nextn.hnorm", "nextn.hc_head_norm" })
            reshape_gamma_in(wm_, b + n + ".weight", n_embd, hc);
    }
    reshape_gamma_in(w_, "output_hc_norm.weight", n_embd, hc);
    for (uint32_t l = 0; l < hp_.n_layer; l++) {
        const std::string b = "blk." + std::to_string(l) + ".";
        for (const char * n : { "hc_attn_norm", "hc_ffn_norm", "ple_norm_key", "ple_norm_query", "ple_norm_conv" })
            reshape_gamma_in(w_, b + n + ".weight", n_embd, hc);
    }
    fold_inject_scale();
}

// Fold the 1/hc of hc_combine's gate into the F32 inject weights: 0.25 is a
// power of two, so every product and partial sum rounds exactly as before
// and the scale kernel disappears. Only if every inject weight is F32.
void engine::fold_inject_scale() {
    const int64_t hc = hp_.hc_count;
    assert_that(hc > 0, "fold_inject_scale: hyper-connection streams");
    assert_that(!hp_.hc_inject_prescaled, "fold_inject_scale: folded once");
    bool all_f32 = true; std::vector<ggml_tensor *> inj;
    for (uint32_t l = 0; l < hp_.n_layer && all_f32; l++)
        for (const char * n : { "hc_attn_inject", "hc_ffn_inject" }) {
            ggml_tensor * t = w_.get("blk." + std::to_string(l) + "." + n + ".weight");
            if (!t) continue;
            if (t->type != GGML_TYPE_F32) { all_f32 = false; break; }
            inj.push_back(t);
        }
    if (all_f32 && !inj.empty()) {
        std::vector<float> buf;
        for (ggml_tensor * t : inj) {
            buf.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
            for (float & v : buf) v *= 1.0f / (float) hc;
            ggml_backend_tensor_set(t, buf.data(), 0, buf.size() * sizeof(float));
        }
        hp_.hc_inject_prescaled = true;
    }
}

void engine::init_pack_and_spec() {
    const int64_t n_embd = hp_.n_embd, U = hp_.n_expert_used, Bd = decode_rows();
    const engine_config & cfg = cfg_;
    assert_that(t_cur_ != nullptr, "init_pack_and_spec: after the work tensors");
    assert_that(t_pack_ == nullptr, "init_pack_and_spec: one pack");
    constexpr int64_t MAXT = 1 + MTP_MAX_DRAFTS;   // positions of the longest verify step
    pack_n_ = MAXT * (n_embd + 2 * (int64_t) U + 2 * (int64_t) QWFN_SPEC_MAX);   // room for a multi-token step
    if ((int64_t) Bd * n_embd >= pack_n_ && cfg.use_gpu) {
        t_pack_ = ggml_view_1d(wctx_, t_cur_, pack_n_, 0);
        pack_host_.resize(pack_n_); pred_next_.resize(MAXT * QWFN_SPEC_MAX); scores_next_.resize(MAXT * QWFN_SPEC_MAX);
    }
    gA_pack_.assign(hp_.n_layer, 0);
    gA_T_.assign(hp_.n_layer, 0);
    // Speculative-block mask by predicted layer, and the per-layer counters.
    spec_block_mask_.assign(hp_.n_layer, cfg.spec_block_layers.empty() ? 1 : 0);
    if (!cfg.spec_block_layers.empty()) parse_spec_block_layers();
    pred_hits_layer.assign(hp_.n_layer, 0); pred_total_layer.assign(hp_.n_layer, 0);
    if (cfg.spec_block || cfg.spec_margin > 0.0f) {
        int nb = 0; for (uint32_t l = 1; l < hp_.n_layer; l++) nb += cfg.spec_block ? spec_block_mask_[l] : 0;
        fprintf(stderr, "[qwfn] prefetch: speculative block %s (%d of %u predicted layers), margin gate %s\n",
                cfg.spec_block ? "on" : "off", nb, hp_.n_layer - 1,
                cfg.spec_margin > 0.0f ? (std::to_string(cfg.spec_margin) + (cfg.spec_gate_inflight ? " when >= " + std::to_string(cfg.spec_gate_inflight) + " reads issued by the last two layers" : "")).c_str() : "off");
    }
}

void engine::parse_spec_block_layers() {
    const std::string & sl = cfg_.spec_block_layers; size_t p = 0;
    assert_that(!sl.empty(), "parse_spec_block_layers: a layer list");
    assert_that(spec_block_mask_.size() == hp_.n_layer, "parse_spec_block_layers: one entry per layer");
    while (p < sl.size()) {
        size_t cpos = sl.find(',', p); if (cpos == std::string::npos) cpos = sl.size();
        const std::string tok = sl.substr(p, cpos - p); p = cpos + 1;
        if (tok.empty()) continue;
        const size_t d = tok.find('-');
        long long a = 0, b = 0;
        const bool ok = parse_int(tok.substr(0, d).c_str(), 0, 65535, a) &&
                        (d == std::string::npos ? (b = a, true) : parse_int(tok.c_str() + d + 1, 0, 65535, b));
        if (!ok) { fprintf(stderr, "[qwfn] ignoring spec-block layer range '%s'\n", tok.c_str()); continue; }
        for (long long l = a; l <= b && l < (long long) hp_.n_layer; l++) spec_block_mask_[l] = 1;
    }
}

void engine::init_route_tensors() {
    const int64_t U = hp_.n_expert_used, PH = hp_.ple_n_head(), Bd = decode_rows();
    const engine_config & cfg = cfg_;
    assert_that(wctx_ != nullptr, "init_route_tensors: the work context");
    assert_that(t_w_ == nullptr, "init_route_tensors: created once");
    t_w_      = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, U, Bd);
    // Rows of the late fold's ids/weights: wide enough for U and for the fold's
    // n_late_ slots (a promotion budget above U/(1+drafts) overflowed a U-wide row
    // into the next token's).
    late_w_   = std::max<int64_t>(U, (1 + MTP_MAX_DRAFTS) * (int64_t) std::max<uint32_t>(1, cfg.promote_per_layer));
    t_gids_   = ggml_new_tensor_2d(wctx_, GGML_TYPE_I32, late_w_, 1 + MTP_MAX_DRAFTS);
    t_gw_     = ggml_new_tensor_3d(wctx_, GGML_TYPE_F32, 1, late_w_, 1 + MTP_MAX_DRAFTS);
    inp_tok_  = ggml_new_tensor_1d(wctx_, GGML_TYPE_I32, Bd);
    inp_pos_  = ggml_new_tensor_1d(wctx_, GGML_TYPE_I32, Bd * 4);
    inp_pos_one_ = ggml_new_tensor_1d(wctx_, GGML_TYPE_I32, Bd * 4);
    inp_ple_  = ggml_new_tensor_1d(wctx_, GGML_TYPE_I32, PH * Bd);
    if (cfg.skip_miss) t_rscale_ = ggml_new_tensor_1d(wctx_, GGML_TYPE_F32, 1);
}

bool engine::init_mtp_work(std::string & err) {
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count, U = hp_.n_expert_used, Bd = decode_rows();
    assert_that(mtp_on_, "init_mtp_work: the head is loaded");
    assert_that(mctx_ == nullptr && mhctx_ == nullptr, "init_mtp_work: allocated once");
    ggml_init_params mp{}; mp.mem_size = ggml_tensor_overhead() * 8; mp.no_alloc = true;
    mctx_ = ggml_init(mp); assert_that(mctx_ != nullptr, "ggml_init: mctx_");
    t_m_res_    = ggml_new_tensor_3d(mctx_, GGML_TYPE_F32, n_embd, hc, Bd);
    t_m_cur_    = ggml_new_tensor_2d(mctx_, GGML_TYPE_F32, n_embd, Bd);
    t_m_inject_ = ggml_new_tensor_2d(mctx_, GGML_TYPE_F32, hc, Bd);
    t_m_sel_    = ggml_new_tensor_2d(mctx_, GGML_TYPE_I32, U, Bd);
    t_m_w_      = ggml_new_tensor_2d(mctx_, GGML_TYPE_F32, U, Bd);
    t_m_sh_     = ggml_new_tensor_2d(mctx_, GGML_TYPE_F32, n_embd, Bd);
    t_m_pc_     = ggml_new_tensor_2d(mctx_, GGML_TYPE_F32, n_embd, Bd);
    t_m_hres_   = ggml_new_tensor_3d(mctx_, GGML_TYPE_F32, n_embd, hc, Bd);
    mbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(mctx_, w_.buft());
    if (!mbuf_) { err = "no device memory for the head's work set"; return false; }
    ggml_init_params hp2{}; hp2.mem_size = ggml_tensor_overhead() * 8; hp2.no_alloc = true;
    mhctx_ = ggml_init(hp2); assert_that(mhctx_ != nullptr, "ggml_init: mhctx_");
    h_m_cur_     = ggml_new_tensor_2d(mhctx_, GGML_TYPE_F32, n_embd, Bd);
    h_m_ids_     = ggml_new_tensor_2d(mhctx_, GGML_TYPE_I32, U, Bd);
    h_m_w_       = ggml_new_tensor_3d(mhctx_, GGML_TYPE_F32, 1, U, Bd);
    h_m_partial_ = ggml_new_tensor_2d(mhctx_, GGML_TYPE_F32, n_embd, Bd);
    mhbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(mhctx_, wh_.buft());
    if (!mhbuf_) { err = "no host memory for the head's work set"; return false; }
    return true;
}

bool engine::init_host_work(std::string & err) {
    const int64_t n_embd = hp_.n_embd, B = cfg_.n_batch, U = hp_.n_expert_used, PH = hp_.ple_n_head();
    assert_that(hctx_ == nullptr, "init_host_work: allocated once");
    assert_that(B >= 1, "init_host_work: a batch of at least one token");
    ggml_init_params hpar{}; hpar.mem_size = ggml_tensor_overhead() * 34; hpar.no_alloc = true;
    hctx_ = ggml_init(hpar); assert_that(hctx_ != nullptr, "ggml_init: hctx_");
    h_cur_     = ggml_new_tensor_2d(hctx_, GGML_TYPE_F32, n_embd, B);
    h_partial_ = ggml_new_tensor_2d(hctx_, GGML_TYPE_F32, n_embd, B);
    h_ple_     = ggml_new_tensor_2d(hctx_, GGML_TYPE_F32, n_embd, B);
    h_ple_idx_ = ggml_new_tensor_1d(hctx_, GGML_TYPE_I32, PH * B);
    h_tok_     = ggml_new_tensor_1d(hctx_, GGML_TYPE_I32, B);
    h_emb_     = ggml_new_tensor_2d(hctx_, GGML_TYPE_F32, n_embd, B);
    if (mtp_on_) { h_mtp_tok_ = ggml_new_tensor_1d(hctx_, GGML_TYPE_I32, B); h_mtp_emb_ = ggml_new_tensor_2d(hctx_, GGML_TYPE_F32, n_embd, B); }
    h_wd_ = ggml_new_tensor_3d(hctx_, GGML_TYPE_F32, n_embd, (1 + MTP_MAX_DRAFTS) * U, 1 + MTP_MAX_DRAFTS);
    hbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(hctx_, wh_.buft());
    if (!hbuf_) { err = "failed to allocate engine host buffer"; return false; }
    return true;
}

// ---- pinned staging for the async id/weight uploads ----------------------
void engine::init_pinned_staging() {
    assert_that(pctx_ == nullptr, "init_pinned_staging: allocated once");
    assert_that(late_w_ > 0, "init_pinned_staging: after the late-fold width");
    if (!w_.on_gpu()) return;
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(w_.buft());
    ggml_backend_buffer_type_t hb = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    if (!hb) return;
    ggml_init_params pp{}; pp.mem_size = ggml_tensor_overhead() * 8; pp.no_alloc = true;
    pctx_    = ggml_init(pp); assert_that(pctx_ != nullptr, "ggml_init: pctx_");
    p_gids_  = ggml_new_tensor_1d(pctx_, GGML_TYPE_I32, (1 + MTP_MAX_DRAFTS) * late_w_);
    p_gw_    = ggml_new_tensor_1d(pctx_, GGML_TYPE_F32, (1 + MTP_MAX_DRAFTS) * late_w_);
    p_vslot_ = ggml_new_tensor_1d(pctx_, GGML_TYPE_I32, hp_.n_expert);
    p_vmask_ = ggml_new_tensor_1d(pctx_, GGML_TYPE_F32, hp_.n_expert);
    p_pc_    = ggml_new_tensor_1d(pctx_, GGML_TYPE_F32, 2 * hp_.n_embd);
    pbuf_    = ggml_backend_alloc_ctx_tensors_from_buft(pctx_, hb);
    if (!pbuf_ || ggml_backend_buffer_get_type(pbuf_) != hb) {
        if (pbuf_) ggml_backend_buffer_free(pbuf_);
        ggml_free(pctx_); pbuf_ = nullptr; pctx_ = nullptr; p_gids_ = nullptr; p_gw_ = nullptr;
        p_vslot_ = nullptr; p_vmask_ = nullptr; p_pc_ = nullptr;
    }
}

// ---- decode-time sparse attention state --------------------------------
void engine::init_qsa_decode() {
    assert_that(qctx_ == nullptr, "init_qsa_decode: allocated once");
    assert_that(hp_.n_layer > 0, "init_qsa_decode: model layers");
    qsa_ratio_ = 0;
    if (cfg_.use_qsa)
        for (uint32_t il = 0; il < hp_.n_layer; il++)
            if (hp_.is_attn_layer(il) && il < hp_.compress_ratios.size() && hp_.compress_ratios[il] > 0)
                { qsa_ratio_ = (uint32_t) hp_.compress_ratios[il]; break; }
    if (!qsa_ratio_) return;
    const int64_t r = qsa_ratio_, NBmax = (cfg_.n_ctx + r - 1) / r;
    init_qsa_tensors(NBmax);
    qbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(qctx_, w_.buft());
    if (!qbuf_) {
        fprintf(stderr, "[qwfn] no device memory for the decode QSA state; decode attention scans the whole context\n");
        ggml_free(qctx_); qctx_ = nullptr; pool_cache_.clear();
    } else {
        init_qsa_tables(NBmax);
    }
}

void engine::init_qsa_tensors(int64_t NBmax) {
    const int64_t r = qsa_ratio_, idx_dim = hp_.idx_key_len;
    assert_that(r > 0 && NBmax > 0, "init_qsa_tensors: a block ratio and blocks");
    assert_that(qctx_ == nullptr, "init_qsa_tensors: one context");
    ggml_init_params qp{}; qp.mem_size = ggml_tensor_overhead() * (hp_.n_layer + 32); qp.no_alloc = true;
    qctx_ = ggml_init(qp); assert_that(qctx_ != nullptr, "ggml_init: qctx_");
    pool_cache_.assign(hp_.n_layer, nullptr);
    for (uint32_t il = 0; il < hp_.n_layer; il++)
        if (hp_.is_attn_layer(il)) pool_cache_[il] = ggml_new_tensor_2d(qctx_, GGML_TYPE_F16, idx_dim, NBmax);
    qd_.bias       = ggml_new_tensor_1d(qctx_, GGML_TYPE_F32, NBmax);
    qd_.blk_cells  = ggml_new_tensor_2d(qctx_, GGML_TYPE_I32, r, NBmax);
    qd_.cell_pos   = ggml_new_tensor_2d(qctx_, GGML_TYPE_F32, 1, cfg_.n_ctx);
    qd_.write_idx  = ggml_new_tensor_1d(qctx_, GGML_TYPE_I32, 1);
    qd_.member_idx = ggml_new_tensor_1d(qctx_, GGML_TYPE_I32, r);
    qd_.blk_pos    = ggml_new_tensor_1d(qctx_, GGML_TYPE_I32, 4);
    qd_.blk_idx    = ggml_new_tensor_1d(qctx_, GGML_TYPE_I32, 1);
    qd_.npast_f    = ggml_new_tensor_1d(qctx_, GGML_TYPE_F32, 1);
    // The later positions of a multi-token decode step: their own per-token
    // inputs and bias; the block tables and pooled keys are shared.
    for (int k = 0; k < MTP_MAX_DRAFTS; k++) {
        qdk_[k] = qd_;
        qdk_[k].bias       = ggml_new_tensor_1d(qctx_, GGML_TYPE_F32, NBmax);
        qdk_[k].write_idx  = ggml_new_tensor_1d(qctx_, GGML_TYPE_I32, 1);
        qdk_[k].member_idx = ggml_new_tensor_1d(qctx_, GGML_TYPE_I32, r);
        qdk_[k].blk_pos    = ggml_new_tensor_1d(qctx_, GGML_TYPE_I32, 4);
        qdk_[k].blk_idx    = ggml_new_tensor_1d(qctx_, GGML_TYPE_I32, 1);
        qdk_[k].npast_f    = ggml_new_tensor_1d(qctx_, GGML_TYPE_F32, 1);
    }
}

void engine::init_qsa_tables(int64_t NBmax) {
    const int64_t r = qsa_ratio_;
    assert_that(qbuf_ != nullptr, "init_qsa_tables: the QSA buffer");
    assert_that(r > 0 && NBmax > 0, "init_qsa_tables: a block ratio and blocks");
    ggml_backend_buffer_clear(qbuf_, 0);
    std::vector<int32_t> bc((size_t) r * NBmax);
    for (int64_t b = 0; b < NBmax; b++) for (int64_t k = 0; k < r; k++) bc[b * r + k] = (int32_t) (b * r + k);
    ggml_backend_tensor_set(qd_.blk_cells, bc.data(), 0, bc.size() * 4);
    std::vector<float> cp(cfg_.n_ctx);
    for (uint32_t i = 0; i < cfg_.n_ctx; i++) cp[i] = (float) i;
    ggml_backend_tensor_set(qd_.cell_pos, cp.data(), 0, cp.size() * 4);
    std::vector<float> ninf(NBmax, -INFINITY);
    ggml_backend_tensor_set(qd_.bias, ninf.data(), 0, ninf.size() * 4);
    for (int k = 0; k < MTP_MAX_DRAFTS; k++) ggml_backend_tensor_set(qdk_[k].bias, ninf.data(), 0, ninf.size() * 4);
    qd_.ratio    = qsa_ratio_;
    qd_.k_blocks = (int64_t) ((hp_.idx_top_k + r - 1) + r - 1) / r;   // ceil(width / r)
    for (int k = 0; k < MTP_MAX_DRAFTS; k++) { qdk_[k].ratio = qd_.ratio; qdk_[k].k_blocks = qd_.k_blocks; }
    fprintf(stderr, "[qwfn] decode QSA state: %.1f MB (pooled block keys for %lld blocks, %lld kept)\n",
            ggml_backend_buffer_get_size(qbuf_) / 1e6, (long long) NBmax, (long long) qd_.k_blocks);
}

// Scratch for the cache-served batched prefill, taken before the VRAM
// tier sizes itself (it is small: 48 slots of the largest expert, ~170 MB).
void engine::init_cbatch_scratch() {
    const engine_config & cfg = cfg_;
    assert_that(scr_buf_ == nullptr, "init_cbatch_scratch: allocated once");
    assert_that(mi_ != nullptr, "init_cbatch_scratch: the model index");
    if (!(cfg.use_gpu && w_.on_gpu() && cfg.cache_batched && cfg.vram_bytes > 0)) return;
    scr_slots_ = std::max<uint32_t>(1, std::min<uint32_t>(64, cfg.cache_batch_chunk));
    size_t total = 0;
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        size_t mx = 0;
        for (uint32_t il = 0; il < hp_.n_layer; il++)
            mx = std::max<size_t>(mx, mi_->expert_range(il, 0, (expert_part) q).nbytes);
        scr_part_off_[q] = total;
        total += mx * scr_slots_ + (16u << 10);   // room for the zeroed tail
    }
    scr_buf_ = ggml_backend_buft_alloc_buffer(w_.buft(), total);
    if (scr_buf_) {
        scr_base_ = (uint8_t *) ggml_backend_buffer_get_base(scr_buf_);
        ggml_backend_buffer_clear(scr_buf_, 0);
        ggml_init_params sp{}; sp.mem_size = ggml_tensor_overhead() * 4; sp.no_alloc = true;
        scr_ctx_  = ggml_init(sp); assert_that(scr_ctx_ != nullptr, "ggml_init: scr_ctx_");
        scr_xfer_ = ggml_new_tensor_1d(scr_ctx_, GGML_TYPE_I8, (int64_t) total);
        scr_xfer_->buffer = scr_buf_;
        scr_xfer_->data   = scr_base_;
    } else {
        fprintf(stderr, "[qwfn] no device memory for the batched-prefill scratch; short prompts go token by token\n");
        scr_slots_ = 0;
    }
}

expert_cache::config engine::expert_cache_config() const {
    const engine_config & cfg = cfg_;
    assert_that(mi_ != nullptr, "expert_cache_config: the model index");
    assert_that(w_.buft() != nullptr, "expert_cache_config: the dense core's buffer type");
    expert_cache::config ec_cfg;
    ec_cfg.ram_bytes     = cfg.ram_bytes;
    // With --cpu the "VRAM" buffer type is host memory: a tier there would just
    // duplicate the arena outside the MemAvailable clamp. GPU only.
    ec_cfg.vram_bytes    = cfg.use_gpu ? cfg.vram_bytes : 0;
    ec_cfg.vram_buft     = w_.buft();
    if (w_.on_gpu()) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(w_.buft());
        ec_cfg.host_buft    = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        ec_cfg.vram_backend = w_.backend();
    }
    ec_cfg.use_cold_tier = cfg.use_cold_tier;
    ec_cfg.max_promotions_per_layer = cfg.promote_per_layer;
    ec_cfg.ram_frac      = cfg.ram_frac;
    ec_cfg.ram_headroom  = cfg.ram_headroom;
    ec_cfg.io_backend  = cfg.io_threads ? io_engine::backend::threads : io_engine::backend::uring;
    if (cfg.io_threads) ec_cfg.queue_depth = cfg.io_workers;
    ec_cfg.policy = cfg.evict_policy == 1 ? expert_cache::config::evict_policy::lfu
                  : cfg.evict_policy == 2 ? expert_cache::config::evict_policy::hybrid
                                          : expert_cache::config::evict_policy::lru;
    if (!(cfg.use_gpu && cfg.vram_bytes > 0)) ec_cfg.lend_bytes = 0;
    return ec_cfg;
}

// What still has to fit on the device after the tier has taken its share:
//
//   - the prefill MoE graph arena. Its xp/yp/yt intermediates are each
//     [n_embd, n_batch * n_expert_used] F32, ~0.31 MB per token of ubatch,
//     so a 2048 ubatch wants ~600 MB;
//   - the per-call input arena, dominated by kq_mask F16 [n_kv, T] and the
//     QSA bias F32 [n_kv/ratio, T], hence bounded by ubatch_kv_product;
//   - the 36 replay allocators, created lazily on the first decode token.
//
// A fixed 768 MB covered ubatch 256 and failed outright at 2048 --
// "prefill galloc failed", 15 minutes into a long prompt.
// What a PREFILL needs on the device, all of it only while one runs: the
// staging, the n_batch work set, the MoE chunk arena, the attention
// chunk's inputs and its arena. During decode this memory holds expert
// slots: it is the expert tier's dynamic buffer (expert_cache lend_bytes),
// freed when a prefill starts and taken back when it ends. What decode
// itself allocates after the tier -- the 48 replay allocators, the
// short-prompt chunk graphs, the head -- is the only reserve left.
void engine::set_lend_budget(expert_cache::config & ec_cfg, const model_index * hot) const {
    const engine_config & cfg = cfg_;
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count;
    const int64_t B = cfg.n_batch, U = hp_.n_expert_used, PH = hp_.ple_n_head();
    assert_that(hot != nullptr, "set_lend_budget: the model index");
    assert_that(B >= 1, "set_lend_budget: a batch of at least one token");
    const size_t Tm = std::min<size_t>(B, std::max<uint32_t>(64, cfg.prefill_chunk));
    const size_t pf_graph = (cfg.prefill_on_gpu && cfg.use_gpu)
        ? (size_t) 4 * n_embd * U * Tm * sizeof(float) : 0;
    const size_t kvp = cfg.ubatch_kv_product ? (size_t) cfg.ubatch_kv_product : (size_t) 32 << 20;
    const size_t inputs    = std::min<size_t>(kvp * 3, 512ull << 20);
    const size_t qsa_graph = cfg.use_qsa ? std::min<size_t>(kvp * 14, 1536ull << 20) : 0;
    const size_t work_bytes = (size_t) (2 * n_embd * hc + 6 * n_embd + hc + 3 * U + 5 + PH) * B * 4 + (64ull << 20);
    const size_t staging   = (cfg.prefill_on_gpu && cfg.use_gpu) ? prefill_streamer::device_bytes_for(hot) : 0;
    ec_cfg.lend_bytes   = staging + work_bytes + pf_graph + inputs + qsa_graph;
    // What decode allocates after the tier grows with the context: the attention
    // graphs are shaped by the block bucket (flat up to ~48K tokens, then per
    // 256 blocks), the CUDA pool with their temporaries, the head's graphs, and
    // the CUDA graph instantiations. Measured at 101K: 805 MB ran out at the
    // first token after the prefill (cudaGraphInstantiate), 1024 held; 768 held
    // at 43K and 10K. 768 MB + 8 MB per 1K tokens above 48K: 1.4 GB at 131K.
    const size_t ctx_k = (size_t) cfg.n_ctx / 1024;
    const size_t auto_reserve = (768ull << 20) + (ctx_k > 48 ? (ctx_k - 48) * (8ull << 20) : 0);
    ec_cfg.vram_reserve = cfg.vram_reserve ? cfg.vram_reserve : auto_reserve;
    fprintf(stderr, "[qwfn] prefill VRAM (dynamic, lent by the expert tier): %.2f GB; decode reserve %.0f MB\n",
            ec_cfg.lend_bytes / 1e9, ec_cfg.vram_reserve / 1e6);
}

bool engine::init_expert_cache(const model_index * hot, const model_index * cold, std::string & err) {
    assert_that(hot != nullptr, "init_expert_cache: the model index");
    assert_that(vctx_ == nullptr, "init_expert_cache: residency tables allocated once");
    expert_cache::config ec_cfg = expert_cache_config();
    set_lend_budget(ec_cfg, hot);

    // Decode computes each layer's VRAM-resident routed experts inside that
    // layer's own graph, looking residency up on the device; see eval_batch.
    moe_in_graph_ = w_.on_gpu() && cfg_.vram_bytes > 0;   // cold-file blocks never reach VRAM, so the in-graph MoE holds with a cold tier
    // Experts promoted to VRAM during a layer's fetch were not resident when
    // that layer's graph ran: the next graph computes them from the tier by
    // slot (at most max_promotions_per_layer of them), the "late fold".
    n_late_ = moe_in_graph_ ? (1 + MTP_MAX_DRAFTS) * (int) std::max<uint32_t>(1, ec_cfg.max_promotions_per_layer) : 0;   // room for a multi-token step's budget
    if (!ec_.init(hot, cold, ec_cfg, err)) return false;
    tier_epoch_seen_ = ec_.tier_epoch();
    if (moe_in_graph_) {
        ggml_init_params vp{}; vp.mem_size = ggml_tensor_overhead() * 2 * hp_.n_layer + 1024; vp.no_alloc = true;
        vctx_ = ggml_init(vp); assert_that(vctx_ != nullptr, "ggml_init: vctx_");
        t_vslot_.assign(hp_.n_layer, nullptr); t_vmask_.assign(hp_.n_layer, nullptr);
        vslot_ver_.assign(hp_.n_layer, UINT64_MAX);
        for (uint32_t il = 0; il < hp_.n_layer; il++) {
            t_vslot_[il] = ggml_new_tensor_2d(vctx_, GGML_TYPE_I32, 1, hp_.n_expert);
            t_vmask_[il] = ggml_new_tensor_2d(vctx_, GGML_TYPE_F32, 1, hp_.n_expert);
        }
        vbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(vctx_, w_.buft());
        if (!vbuf_) { err = "failed to allocate the VRAM residency tables"; return false; }
        ggml_backend_buffer_clear(vbuf_, 0);
    }
    return true;
}

bool engine::init_graphs(const model_index * hot, std::string & err) {
    const engine_config & cfg = cfg_;
    const int64_t n_embd = hp_.n_embd, B = cfg.n_batch, U = hp_.n_expert_used;
    assert_that(hot != nullptr, "init_graphs: the model index");
    assert_that(gA_.empty() && gM_.empty(), "init_graphs: the graph tables are laid out once");
    if (!pf_.init(hot, cfg.io_workers, cfg.io_threads,
                  cfg.prefill_on_gpu ? w_.buft() : nullptr,
                  cfg.prefill_on_gpu ? w_.backend() : nullptr, cfg.prefill_overlap, err)) return false;
    // The streamed sweep takes the slices the RAM tier holds from it instead of
    // the file.
    pf_.set_resident_source(&ec_);

    gA_.assign(hp_.n_layer, layer_graph{});
    gA_bucket_.assign(hp_.n_layer, -1);
    gM_.assign(hp_.n_layer, moe_graph{});
    galloc_gpu_ = ggml_gallocr_new(w_.buft());
    galloc_cpu_ = ggml_gallocr_new(wh_.buft());
    galloc_moe_ = ggml_gallocr_new(w_.buft());
    if (!galloc_gpu_ || !galloc_cpu_ || !galloc_moe_) { err = "failed to create the graph allocators"; return false; }

    // The prefill MoE allocator is created per prefill (prefill_enter) and
    // freed after.

    logits_.resize((size_t) n_vocab_ * (1 + MTP_MAX_DRAFTS));   // every position of a verify step
    xfer_.resize((size_t) n_embd * B);
    zeros_.assign((size_t) n_embd * B, 0.0f);
    sel_.resize((size_t) U * B);
    wgt_.resize((size_t) U * B);
    ids_.resize((size_t) U * B);
    init_step_scratch();
    return true;
}


// The mul_mat_id MoE over one tier: gate/up/down as [.., .., n_slots] tensors
// aliasing the tier's arrays, experts chosen by `ids` (slot indices, [n, 1]),
// gate weights `w` ([1, n, 1]), the n weighted outputs summed in order. Per
// expert this is the same mmvq / vec_dot kernel the per-expert path ran, the
// same scale-by-weight, and the same left-to-right sum, so it is bit-identical
// to that path -- an expert with weight 0 adds an exact 0.0 and changes
// nothing. Returns the [n_embd, 1] sum.
// The weighted output row of every selected expert, [n_embd, n, T]: the three
// mul_mat_id and the gate weight, nothing summed. moe_id_graph sums them; the
// decode path sums them itself after computing the rows in two passes.
static ggml_tensor * moe_id_wd(ggml_context * c, const tier_view & tv,
                               ggml_tensor * ids, ggml_tensor * w, ggml_tensor * x_in,
                               int64_t n_embd, int64_t n_ff, int n, int64_t T) {
    assert_that(c && ids && w && x_in, "moe_id_wd: graph inputs");
    assert_that(n >= 1 && T >= 1, "moe_id_wd: at least one expert and one token");
    ggml_tensor * as[EXPERT_NPARTS];
    as[EXPERT_GATE] = ggml_new_tensor_3d(c, tv.type[EXPERT_GATE], n_embd, n_ff, tv.n_slots);
    as[EXPERT_UP]   = ggml_new_tensor_3d(c, tv.type[EXPERT_UP],   n_embd, n_ff, tv.n_slots);
    as[EXPERT_DOWN] = ggml_new_tensor_3d(c, tv.type[EXPERT_DOWN], n_ff, n_embd, tv.n_slots);
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        as[q]->buffer = tv.buffer;
        as[q]->data   = tv.part[q];
        as[q]->nb[2]  = tv.stride[q];
        as[q]->nb[3]  = tv.stride[q] * tv.n_slots;
    }
    ggml_tensor * x    = ggml_view_3d(c, x_in, n_embd, 1, T, x_in->nb[1], x_in->nb[1], 0);
    ggml_tensor * gate = ggml_mul_mat_id(c, as[EXPERT_GATE], x, ids);          // [n_ff, n, T]
    ggml_tensor * up   = ggml_mul_mat_id(c, as[EXPERT_UP],   x, ids);
    ggml_tensor * act  = ggml_swiglu_split(c, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(c, as[EXPERT_DOWN], act, ids);        // [n_embd, n, T]
    GGML_UNUSED(n);
    return ggml_mul(c, down, w);
}

static ggml_tensor * moe_id_graph(ggml_context * c, const tier_view & tv,
                                  ggml_tensor * ids, ggml_tensor * w, ggml_tensor * x_in,
                                  int64_t n_embd, int64_t n_ff, int n, int64_t T = 1,
                                  bool fused_sum = false) {
    assert_that(c && ids && w && x_in, "moe_id_graph: graph inputs");
    assert_that(n >= 1 && T >= 1, "moe_id_graph: at least one expert and one token");
    ggml_tensor * as[EXPERT_NPARTS];
    as[EXPERT_GATE] = ggml_new_tensor_3d(c, tv.type[EXPERT_GATE], n_embd, n_ff, tv.n_slots);
    as[EXPERT_UP]   = ggml_new_tensor_3d(c, tv.type[EXPERT_UP],   n_embd, n_ff, tv.n_slots);
    as[EXPERT_DOWN] = ggml_new_tensor_3d(c, tv.type[EXPERT_DOWN], n_ff, n_embd, tv.n_slots);
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        as[q]->buffer = tv.buffer;
        as[q]->data   = tv.part[q];
        as[q]->nb[2]  = tv.stride[q];
        as[q]->nb[3]  = tv.stride[q] * tv.n_slots;
    }
    // x_in is [n_embd, T] token-major; as [n_embd, 1, T] each token's vector is
    // broadcast to its n experts. ids is [n, T], w is [1, n, T].
    ggml_tensor * x    = ggml_view_3d(c, x_in, n_embd, 1, T, x_in->nb[1], x_in->nb[1], 0);
    ggml_tensor * gate = ggml_mul_mat_id(c, as[EXPERT_GATE], x, ids);          // [n_ff, n, T]
    ggml_tensor * up   = ggml_mul_mat_id(c, as[EXPERT_UP],   x, ids);
    ggml_tensor * act  = ggml_swiglu_split(c, gate, up);                        // silu(gate) * up, one op; fused with the matmuls on CUDA
    ggml_tensor * down = ggml_mul_mat_id(c, as[EXPERT_DOWN], act, ids);        // [n_embd, n, T]
    if (fused_sum && T <= 2) {
        // The weighted sum of the n expert rows as one matmul over the transposed
        // rows: two kernels instead of a multiply and n-1 adds. A different
        // summation order, so only the GPU graphs ask for it; the CPU path keeps
        // the sequential sum it is validated with. Batched over the T positions.
        ggml_tensor * dt = ggml_reshape_3d(c, ggml_cont(c, ggml_permute(c, down, 1, 0, 2, 3)), n, n_embd, T);   // [n, n_embd, T]
        ggml_tensor * wv = ggml_reshape_3d(c, ggml_cont(c, w), n, 1, T);                                       // [n, 1, T]
        return ggml_reshape_2d(c, ggml_mul_mat(c, dt, wv), n_embd, T);                                          // [n_embd, T]
    }
    ggml_tensor * wd   = ggml_mul(c, down, w);
    // Sum the n expert rows of each token, in order.
    ggml_tensor * acc  = ggml_cont(c, ggml_view_2d(c, wd, n_embd, T, wd->nb[2], 0));
    for (int e = 1; e < n; e++)
        acc = ggml_add(c, acc, ggml_view_2d(c, wd, n_embd, T, wd->nb[2], (size_t) e * wd->nb[1]));
    return acc;
}

bool engine::build_moe_gpu_graph(uint32_t il) {
    assert_that(il < hp_.n_layer && il < gM_.size(), "build_moe_gpu_graph: layer in range");
    assert_that(gM_[il].gf == nullptr, "build_moe_gpu_graph: built once per tier epoch");
    const tier_view tv = ec_.gpu_tier(il);
    if (tv.n_slots == 0) return false;
    const int64_t n_embd = hp_.n_embd, n_ff = hp_.n_ff_exp;
    const int U = (int) hp_.n_expert_used;
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        const int64_t rows = q == EXPERT_DOWN ? n_embd : n_ff;
        const int64_t cols = q == EXPERT_DOWN ? n_ff : n_embd;
        if (tv.stride[q] != ggml_row_size(tv.type[q], cols) * (size_t) rows) {
            fprintf(stderr, "[qwfn] layer %u: VRAM tier stride is not the natural slice; mul_mat_id MoE disabled\n", il);
            return false;
        }
    }
    moe_graph & mg = gM_[il];
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false);
    p.no_alloc = true;
    mg.ctx = ggml_init(p); assert_that(mg.ctx != nullptr, "ggml_init: mg.ctx");
    mg.gf  = ggml_new_graph_custom(mg.ctx, 128, false);
    // One position: the first column of the id and weight tables (they are sized for a verify step).
    ggml_tensor * ids1 = ggml_view_2d(mg.ctx, t_gids_, U, 1, t_gids_->nb[1], 0);
    ggml_tensor * w1   = ggml_view_3d(mg.ctx, t_gw_, 1, U, 1, t_gw_->nb[1], t_gw_->nb[2], 0);
    ggml_tensor * acc = moe_id_graph(mg.ctx, tv, ids1, w1, t_cur_, n_embd, n_ff, U);
    ggml_build_forward_expand(mg.gf, ggml_cpy(mg.ctx, acc,
            ggml_view_2d(mg.ctx, t_pg_, n_embd, 1, t_pg_->nb[1], 0)));
    mg.ga = ggml_gallocr_new(w_.buft());
    if (!mg.ga || !ggml_gallocr_alloc_graph(mg.ga, mg.gf)) {
        if (mg.ga) ggml_gallocr_free(mg.ga);
        ggml_free(mg.ctx);
        mg = moe_graph{};
        static bool warned = false;
        if (!warned) { warned = true; fprintf(stderr, "[qwfn] mul_mat_id MoE graph: no device memory; using the per-expert path\n"); }
        return false;
    }
    return true;
}

void engine::set_embeddings(int32_t pos, const float * emb, int32_t n) {
    assert_that(n >= 0, "set_embeddings: a row count");
    assert_that(ov_.size() == ov_pos_.size() * (size_t) hp_.n_embd, "set_embeddings: one row per substituted position");
    const int64_t d = hp_.n_embd;
    const size_t base = ov_.size();
    ov_.resize(base + (size_t) n * d);
    memcpy(ov_.data() + base, emb, (size_t) n * d * sizeof(float));
    for (int32_t i = 0; i < n; i++) ov_pos_.push_back(pos + i);
}

// The tensors of a snapshot, in a fixed order, with how many bytes of each belong
// to the first n positions: positional caches by rows, recurrent state whole.
static void snapshot_parts(const state & st, const hparams & hp, int32_t n,
                           std::vector<std::pair<ggml_tensor *, size_t>> & out) {
    assert_that(n >= 0, "snapshot_parts: a position count");
    assert_that(st.n_ctx() > 0, "snapshot_parts: an initialised state");
    for (uint32_t il = 0; il < hp.n_layer; il++) {
        for (ggml_tensor * t : { st.k_cache(il), st.v_cache(il), st.idx_cache(il) })
            if (t) out.push_back({ t, std::min(ggml_nbytes(t), (size_t) n * ggml_row_size(t->type, t->ne[0] / st.n_ctx())) });
        for (ggml_tensor * t : { st.rs_state(il), st.rs_conv(il) })
            if (t) out.push_back({ t, ggml_nbytes(t) });
    }
    if (ggml_tensor * t = st.ple_conv()) out.push_back({ t, ggml_nbytes(t) });
}

bool engine::snapshot_save(state_snapshot & s) {
    assert_that(mi_ != nullptr, "snapshot_save: engine initialised");
    assert_that(n_past_ >= 0 && n_past_ <= (int32_t) cfg_.n_ctx, "snapshot_save: position within the context");
    ggml_backend_synchronize(w_.backend());
    std::vector<std::pair<ggml_tensor *, size_t>> parts;
    snapshot_parts(st_, hp_, n_past_, parts);
    if (mtp_on_) snapshot_parts(st_mtp_, hpm_, n_past_, parts);
    try {
        s.n_past = n_past_;
        s.blobs.assign(parts.size(), {});
        for (size_t k = 0; k < parts.size(); k++) {
            s.blobs[k].resize(parts[k].second);
            if (parts[k].second) ggml_backend_tensor_get(parts[k].first, s.blobs[k].data(), 0, parts[k].second);
        }
        s.mtp_have_h = mtp_on_ && mtp_have_h_ && mtp_h_rows_ >= 1 && t_hlast_;
        s.mtp_kv_valid = mtp_kv_valid_;
        s.hlast.clear();
        if (s.mtp_have_h) {   // the row of the last evaluated position
            s.hlast.resize(t_hlast_->nb[2]);
            ggml_backend_tensor_get(t_hlast_, s.hlast.data(), (size_t) (mtp_h_rows_ - 1) * t_hlast_->nb[2], s.hlast.size());
        }
    } catch (const std::bad_alloc &) {
        s = state_snapshot{};
        return false;
    }
    return true;
}

bool engine::snapshot_load(const state_snapshot & s) {
    assert_that(mi_ != nullptr, "snapshot_load: engine initialised");
    assert_that(n_past_ >= 0 && n_past_ <= (int32_t) cfg_.n_ctx, "snapshot_load: position within the context");
    ggml_backend_synchronize(w_.backend());   // nothing of the old sequence still in flight
    ec_.settle_promotions();
    reset();
    std::vector<std::pair<ggml_tensor *, size_t>> parts;
    snapshot_parts(st_, hp_, s.n_past, parts);
    if (mtp_on_) snapshot_parts(st_mtp_, hpm_, s.n_past, parts);
    if (parts.size() != s.blobs.size()) return false;
    for (size_t k = 0; k < parts.size(); k++) {
        if (parts[k].second != s.blobs[k].size()) return false;
        if (parts[k].second) ggml_backend_tensor_set(parts[k].first, s.blobs[k].data(), 0, parts[k].second);
    }
    n_past_ = s.n_past;
    pool_dirty_ = true;   // pooled block keys and the block bias are rebuilt from the caches
    if (mtp_on_) {
        mtp_kv_valid_ = s.mtp_kv_valid;
        if (s.mtp_have_h && t_hlast_ && s.hlast.size() == t_hlast_->nb[2]) {
            ggml_backend_tensor_set(t_hlast_, s.hlast.data(), 0, s.hlast.size());
            mtp_have_h_ = true; mtp_h_rows_ = 1;
        }
    }
    return true;
}

void engine::reset() {
    assert_that(n_past_ >= 0, "reset: a valid position");
    assert_that(qbuf_ == nullptr || qd_.bias != nullptr, "reset: the decode QSA bias exists with its buffer");
    mtp_have_h_ = false; mtp_kv_valid_ = true; mtp_draft_ = -1; rb_valid_ = false;
    st_.reset(); n_past_ = 0;
    pool_dirty_ = true;
    if (qbuf_) {
        const int64_t NBmax = qd_.bias->ne[0];
        std::vector<float> ninf(NBmax, -INFINITY);
        ggml_backend_tensor_set(qd_.bias, ninf.data(), 0, ninf.size() * 4);
    }
}

// Everything the decode attention graph needs for this token: the write row,
// the cells and positions of the block holding it, the bias window around it,
// and the block bucket the graph is shaped for. After a prefill the pooled
// keys of every complete block are rebuilt once from the raw cache.
void engine::qsa_decode_prepare(int32_t n_past) {
    assert_that(qsa_ratio_ > 0 && qd_.bias != nullptr, "qsa_decode_prepare: the decode QSA state");
    assert_that(n_past >= 0 && n_past < (int32_t) cfg_.n_ctx, "qsa_decode_prepare: position within the context");
    const int64_t r      = qsa_ratio_;
    const int64_t NBmax  = qd_.bias->ne[0];
    const int32_t b_last = n_past / (int32_t) r;
    const int64_t n_bid  = (n_past + 1) / r;                 // whole blocks once this token is in

    const int32_t wi = n_past;
    ggml_backend_tensor_set(qd_.write_idx, &wi, 0, 4);
    std::vector<int32_t> mi(r);
    for (int64_t m = 0; m < r; m++) mi[m] = (int32_t) (b_last * r + m);
    ggml_backend_tensor_set(qd_.member_idx, mi.data(), 0, mi.size() * 4);
    int32_t bp[4] = { (int32_t) (b_last * r), (int32_t) (b_last * r), (int32_t) (b_last * r), (int32_t) (b_last * r) };
    ggml_backend_tensor_set(qd_.blk_pos, bp, 0, sizeof bp);
    ggml_backend_tensor_set(qd_.blk_idx, &b_last, 0, 4);
    const float nf = (float) n_past;
    ggml_backend_tensor_set(qd_.npast_f, &nf, 0, 4);

    // Bias: 0 for whole blocks, 1e9 for the incomplete tail block (always
    // visible), -inf past it. Only the window around b_last can have changed.
    float win[3]; int64_t b0 = std::max<int64_t>(0, b_last - 1), n = 0;
    for (int64_t b = b0; b <= b_last + 1 && b < NBmax; b++, n++)
        win[n] = b < n_bid ? 0.0f : (b == b_last ? 1e9f : -INFINITY);
    ggml_backend_tensor_set(qd_.bias, win, (size_t) b0 * 4, (size_t) n * 4);

    int64_t NB = ((b_last + 1 + 255) / 256) * 256;
    NB = std::max<int64_t>(NB, 768);
    NB = std::min<int64_t>(NB, NBmax);
    qd_.n_bucket = NB;
    qd_.k_blocks = std::min<int64_t>(qd_.k_blocks > 0 ? qd_.k_blocks : NB, NB);

    if (pool_dirty_) qsa_pool_rebuild(n_past, b_last, n_bid);
}

void engine::qsa_pool_rebuild(int32_t n_past, int32_t b_last, int64_t n_bid) {
    const int64_t r      = qsa_ratio_;
    const int64_t NBmax  = qd_.bias->ne[0];
    assert_that(pool_dirty_, "qsa_pool_rebuild: only after a prefill or a reset");
    assert_that(r > 0 && pool_cache_.size() == hp_.n_layer, "qsa_pool_rebuild: one pooled cache slot per layer");
    // A prefill does not maintain the bias: every block it filled still
    // carries the -inf it was reset to, so the top-k would take the
    // window's blocks and then arbitrary ones -- fine below 513 blocks
    // (everything is selected anyway), a random 1.5% of the context at
    // 133K. Seen as ungrounded answers over a long document while the
    // mask-based path over the same caches was grounded. Rewrite it whole.
    {
        std::vector<float> full(NBmax, -INFINITY);
        for (int64_t b = 0; b < n_bid && b < NBmax; b++) full[b] = 0.0f;
        if (b_last < NBmax) full[b_last] = (b_last < n_bid) ? 0.0f : 1e9f;
        ggml_backend_tensor_set(qd_.bias, full.data(), 0, full.size() * 4);
    }
    const int64_t n_whole = n_past / r;
    if (n_whole > 0) {
        std::vector<int32_t> bpa((size_t) 4 * n_whole);
        for (int sec = 0; sec < 4; sec++)
            for (int64_t b = 0; b < n_whole; b++) bpa[sec * n_whole + b] = (int32_t) (b * r);
        for (uint32_t il = 0; il < hp_.n_layer; il++) {
            if (!pool_cache_[il]) continue;
            ggml_init_params p{};
            p.mem_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false); p.no_alloc = true;
            ggml_context * c = ggml_init(p); assert_that(c != nullptr, "ggml_init: c");
            ggml_cgraph * g = ggml_new_graph_custom(c, 256, false);
            graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, n_past);
            ggml_tensor * t_bp = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * n_whole); ggml_set_input(t_bp);
            qd_.pool_cache = pool_cache_[il];
            gb.qsa_pool_rebuild((int) il, qd_, n_whole, t_bp);
            if (!ggml_gallocr_alloc_graph(galloc_gpu_, g)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
            ggml_backend_tensor_set(t_bp, bpa.data(), 0, bpa.size() * 4);
            if (ggml_backend_graph_compute(w_.backend(), g) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[qwfn] compute failed\n"); abort();
            }
            ggml_free(c);
        }
    }
    pool_dirty_ = false;
}

std::string engine::memory_summary() const {
    assert_that(mi_ != nullptr, "memory_summary: engine initialised");
    assert_that(n_past_ >= 0, "memory_summary: a valid position");
    std::ostringstream o;
    o << "dense core " << w_.bytes() / 1e9 << " GB on " << w_.dev_name()
      << " | " << st_.summary()
      << " | expert cache " << ec_.capacity_experts() << " RAM blocks, "
      << ec_.capacity_experts_gpu() << " VRAM blocks";
    return o.str();
}

void engine::run_on(ggml_cgraph * gf, bool gpu) {
    assert_that(gf != nullptr, "run_on: a graph");
    assert_that((gpu ? galloc_gpu_ : galloc_cpu_) != nullptr, "run_on: the backend's allocator");
    ggml_gallocr_t ga = gpu ? galloc_gpu_ : galloc_cpu_;
    ggml_backend_t be = gpu ? w_.backend() : wh_.backend();
    if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[qwfn] compute failed\n"); abort();
    }
}

const float * engine::eval(const int32_t * hist, int32_t n_hist, int32_t n_new, std::string & err) {
    assert_that(mi_ != nullptr, "eval: engine initialised");
    assert_that(cfg_.n_batch >= 1 && cfg_.n_ctx >= 1, "eval: batch and context sizes");
    if (n_new <= 0 || n_new > (int32_t) cfg_.n_batch) {
        err = "n_new out of range (1.." + std::to_string(cfg_.n_batch) + ")"; return nullptr;
    }
    if (n_past_ + n_new > (int32_t) cfg_.n_ctx) { err = "context exhausted"; return nullptr; }

    // The per-call input arena is dominated by kq_mask F16 [n_kv, T] and the QSA
    // bias F32 [n_blocks, T], both linear in n_kv*T. A ubatch that fits at 16K
    // does not fit at 128K: measured 3452 MiB at n_kv=131072, T=2048, which is
    // more device memory than is left after the dense core and the staging. So
    // cap the product and let long contexts shrink the ubatch themselves rather
    // than failing 15 minutes into a prefill.
    // A short prefill is cheaper token by token: the streaming path's cost is
    // the whole expert set per layer regardless of T, so it only pays off once
    // T is large enough to amortise it.
    const uint64_t cbatch_prod = cfg_.cbatch_kv_product;
    const bool as_decode = n_new > 1 && n_new <= (int32_t) cfg_.prefill_decode_max &&
                           (cbatch_prod == 0 || (uint64_t) n_new * (uint64_t) std::max<int32_t>(n_past_, 1) <= cbatch_prod);
    // ... and, with a GPU, as ONE batch whose experts come through the cache.
    const bool as_cbatch = as_decode && cfg_.cache_batched && scr_buf_;   // cold-resident experts are re-read hot by fetch_batch

    const int32_t base = n_hist - n_new;
    // Long prompts: layer-major, one expert sweep per n_batch tokens, the
    // attention chunked to ubatch_kv_product inside.
    const bool streamed = n_new > 1 && !as_decode;
    // A streamed prefill fills the head's KV rows for its positions as it goes
    // (eval_prefill_big), so the head keeps drafting after a long prompt.
    // A client lend (the server staging the vision projector) that no prefill
    // followed: take the tier back before a decode, and rebuild whatever the
    // lend invalidated either way.
    if (!streamed && client_lent_) vram_lend_end();
    if (!streamed) sync_tier_epoch();
    if (streamed && !prefill_enter(err)) return nullptr;
    bool ok = true;
    if (streamed) {
        for (int32_t off = 0; off < n_new && ok; ) {
            const int32_t take = std::min<int32_t>(n_new - off, (int32_t) cfg_.n_batch);
            ok = eval_prefill_big(hist, base + off + take, take, err);
            off += take;
        }
    } else {
        for (int32_t off = 0; off < n_new && ok; ) {
            const int32_t take = as_cbatch ? n_new : 1;
            ok = eval_batch(hist, base + off + take, take, err, as_cbatch);
            off += take;
        }
    }
    if (streamed) prefill_leave();
    prefill_progress_.store(0, std::memory_order_relaxed);   // no batch in flight
    return ok ? logits_.data() : nullptr;
}


int32_t engine::max_ubatch(int32_t n_past) const {
    assert_that(n_past >= 0, "max_ubatch: a valid position");
    assert_that(cfg_.n_batch >= 1, "max_ubatch: a batch of at least one token");
    if (cfg_.ubatch_kv_product == 0) return (int32_t) cfg_.n_batch;
    const int64_t t = (int64_t) cfg_.ubatch_kv_product / std::max<int64_t>(n_past + 1, 1);
    int32_t take = (int32_t) std::min<int64_t>(t, (int64_t) cfg_.n_batch);
    take &= ~63;                                  // keep ubatches 64-aligned
    return std::max(take, 64);
}


bool engine::build_attn_inputs(int64_t n_past_c, int64_t Tc, attn_inputs & ai, std::string & err) {
    assert_that(n_past_c >= 0 && Tc >= 1, "build_attn_inputs: a chunk of at least one query");
    assert_that(ai.ctx == nullptr && ai.buf == nullptr, "build_attn_inputs: fresh inputs");
    const int64_t n_kv = n_past_c + Tc;
    ggml_init_params ip{}; ip.mem_size = ggml_tensor_overhead() * 16; ip.no_alloc = true;
    ai.ctx = ggml_init(ip); assert_that(ai.ctx != nullptr, "ggml_init: ai.ctx");
    ai.kq_mask = ggml_new_tensor_2d(ai.ctx, GGML_TYPE_F16, n_kv, Tc);
    ai.ratio = cfg_.use_qsa ? qsa_ratio_ : 0;
    const uint32_t ratio = ai.ratio;
    const int64_t n_blocks = ratio ? (n_kv + ratio - 1) / ratio : 0;
    if (ratio) {
        ai.qsa.cell_blk  = ggml_new_tensor_1d(ai.ctx, GGML_TYPE_I32, n_kv);
        ai.qsa.blk_cells = ggml_new_tensor_1d(ai.ctx, GGML_TYPE_I32, ratio * n_blocks);
        ai.qsa.blk_pos   = ggml_new_tensor_1d(ai.ctx, GGML_TYPE_I32, 4 * n_blocks);
        ai.qsa.bias      = ggml_new_tensor_2d(ai.ctx, GGML_TYPE_F32, n_blocks, Tc);
        ai.qsa.ratio     = ratio;
        ai.qsa.n_blocks  = n_blocks;
    }
    ai.buf = ggml_backend_alloc_ctx_tensors_from_buft(ai.ctx, w_.buft());
    if (!ai.buf) { ggml_free(ai.ctx); ai.ctx = nullptr; err = "failed to allocate per-chunk attention inputs"; return false; }
    // These are O(n_kv * Tc) per chunk and at 128K ran for a third of the
    // prefill when written element by element. Row-wise fills: a row is a
    // run of zeros then a run of -inf (F16 0x0000 / 0xFC00).
    {
        std::vector<uint16_t> m((size_t) n_kv * Tc);
        const uint16_t ninf = f16_of(-INFINITY);
        for (int64_t i = 0; i < Tc; i++) {
            uint16_t * row = m.data() + i * n_kv;
            const int64_t vis = std::min<int64_t>(n_kv, n_past_c + i + 1);
            memset(row, 0, (size_t) vis * 2);
            std::fill(row + vis, row + n_kv, ninf);
        }
        ggml_backend_tensor_set(ai.kq_mask, m.data(), 0, m.size() * 2);
    }
    if (ratio) fill_chunk_qsa(ai, n_past_c, Tc);
    return true;
}

void engine::fill_chunk_qsa(attn_inputs & ai, int64_t n_past_c, int64_t Tc) {
    const uint32_t ratio = ai.ratio;
    assert_that(ratio > 0 && ai.buf != nullptr, "fill_chunk_qsa: allocated block tables");
    assert_that(n_past_c >= 0 && Tc >= 1, "fill_chunk_qsa: a chunk of at least one query");
    const int64_t n_kv = n_past_c + Tc;
    const int64_t n_blocks = ai.qsa.n_blocks;
    const int64_t n_bid = n_kv / ratio;
    const bool have_dead = n_bid < n_blocks;
    const int64_t dead = have_dead ? n_bid : n_blocks - 1;
    std::vector<int32_t> cb(n_kv), bc((size_t) ratio * n_blocks, 0), bp((size_t) 4 * n_blocks, 0);
    std::vector<float> bi((size_t) n_blocks * Tc);
    for (int64_t j = 0; j < n_bid * (int64_t) ratio; j++) cb[j] = (int32_t) (j / ratio);
    for (int64_t j = n_bid * (int64_t) ratio; j < n_kv; j++) cb[j] = (int32_t) dead;
    for (int64_t b = 0; b < n_bid; b++) {
        for (uint32_t k = 0; k < ratio; k++) bc[b * ratio + k] = (int32_t) (b * ratio + k);
        for (int sec = 0; sec < 4; sec++) bp[sec * n_blocks + b] = (int32_t) (b * ratio);
    }
    // The partial tail block, if any: its real cells (the last one repeated) and position, as decode
    // maps it. Left at zero it pointed at cell 0, and a query inside it -- the last 1-3 tokens of a chunk
    // that ends mid-block -- could not attend to itself or the cells just before it.
    if (n_bid < n_blocks) {
        for (uint32_t k = 0; k < ratio; k++)
            bc[n_bid * ratio + k] = (int32_t) std::min<int64_t>(n_bid * (int64_t) ratio + k, n_kv - 1);
        for (int sec = 0; sec < 4; sec++) bp[sec * n_blocks + n_bid] = (int32_t) (n_bid * ratio);
    }
    // Row i (query q): 0 for whole blocks before the tail, 1e9 for the block holding q
    // (whole or the dead one), -1e9 for blocks wholly after q, -inf for blocks past
    // n_bid. A block after q must not be forced: in a prefill chunk it would take one
    // of the selection's slots from a past block, and the causal mask empties it.
    for (int64_t i = 0; i < Tc; i++) {
        float * row = bi.data() + i * n_blocks;
        const int64_t q = n_past_c + i;
        const int64_t tail_b = std::min<int64_t>(n_bid, ((q + 1) / ratio));   // first block at/after the tail
        std::fill(row, row + tail_b, 0.0f);
        std::fill(row + tail_b, row + n_bid, 1e9f);
        std::fill(row + n_bid, row + n_blocks, -INFINITY);
        if (have_dead) row[dead] = 1e9f;
        for (int64_t b = tail_b; b < n_blocks; b++)
            if (row[b] > 0.0f && b * (int64_t) ratio > q) row[b] = -1e9f;
    }
    ggml_backend_tensor_set(ai.qsa.cell_blk,  cb.data(), 0, cb.size() * 4);
    ggml_backend_tensor_set(ai.qsa.blk_cells, bc.data(), 0, bc.size() * 4);
    ggml_backend_tensor_set(ai.qsa.blk_pos,   bp.data(), 0, bp.size() * 4);
    ggml_backend_tensor_set(ai.qsa.bias,      bi.data(), 0, bi.size() * 4);
}

bool engine::spec_layer0(const int32_t * hist, int32_t n_hist, int32_t T, std::string & err) {
    if (!cfg_.speculate || !w_.on_gpu() || T < 1 || T > 2) return true;
    if (hp_.is_attn_layer(0)) return true;            // would need the attention inputs; not this model
    if (n_hist - T != n_past_) return true;           // only for the very next positions
    assert_that(hist != nullptr && n_past_ >= 0, "spec_layer0: the sequence up to the next positions");
    assert_that(h_tok_ != nullptr && t_emb_ != nullptr, "spec_layer0: the token inputs");
    const int64_t U = hp_.n_expert_used;
    const auto t0 = std::chrono::steady_clock::now();
    // The tokens' inputs, gathered as eval_batch gathers them (its own pass overwrites
    // these). The PLE rows only if layer 0 is a PLE layer: on this model it is layer 1,
    // and the rows are gathered from a 38 GB host mapping -- most of a millisecond.
    const bool is_ple = std::find(hp_.ple_layers.begin(), hp_.ple_layers.end(), (int32_t) 0) != hp_.ple_layers.end();
    spec_l0_inputs(hist, n_hist, T, is_ple);
    std::vector<int32_t> ids((size_t) U * T);
    spec_l0_route(T, is_ple, ids);
    std::vector<uint32_t> pf; pf.reserve(ids.size());
    for (int32_t id : ids) if (std::find(pf.begin(), pf.end(), (uint32_t) id) == pf.end()) pf.push_back((uint32_t) id);
    ec_.prefetch_layer_begin(0, pf.data(), (uint32_t) pf.size());
    t_spec_l0 += secs_since(t0);
    n_spec_l0++;
    (void) err;
    return true;
}

void engine::spec_l0_inputs(const int32_t * hist, int32_t n_hist, int64_t T, bool is_ple) {
    const int64_t n_embd = hp_.n_embd, PH = hp_.ple_n_head();
    assert_that(hist != nullptr && T >= 1 && T <= 2, "spec_l0_inputs: one or two tokens");
    assert_that(n_hist >= T, "spec_l0_inputs: the tokens are in the sequence");
    ggml_backend_tensor_set(h_tok_, hist + (n_hist - T), 0, (size_t) T * 4);
    if (is_ple) {
        std::vector<int32_t> & rows = ple_rows_host_;
        rows.assign((size_t) PH * T, 0);
        for (int64_t i = 0; i < T; i++) {
            const ple_rows r = ple_rows_for(hp_, hist, n_hist, (n_hist - T) + i);
            for (uint32_t h = 0; h < r.n; h++) rows[i * PH + h] = (int32_t) r.row[h];
        }
        ggml_backend_tensor_set(h_ple_idx_, rows.data(), 0, rows.size() * 4);
    }
    const graph_ctx gc = new_graph_ctx(4096);
    ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
    if (is_ple) {
        ggml_tensor * idx = ggml_view_1d(c, h_ple_idx_, PH * T, 0);
        ggml_tensor * e = ggml_get_rows(c, wh_.get("per_layer_token_embd.weight"), idx);
        e = ggml_reshape_2d(c, e, hp_.d_ple * PH, T);
        ggml_build_forward_expand(g, ggml_cpy(c, e, ggml_view_2d(c, h_ple_, n_embd, T, h_ple_->nb[1], 0)));
    }
    ggml_tensor * te = ggml_get_rows(c, wh_.get("token_embd.weight"), ggml_view_1d(c, h_tok_, T, 0));
    ggml_build_forward_expand(g, ggml_cpy(c, te, ggml_view_2d(c, h_emb_, n_embd, T, h_emb_->nb[1], 0)));
    run_on(g, false); ggml_free(c);
    if (is_ple) {
        ggml_backend_tensor_get(h_ple_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
        ggml_backend_tensor_set(t_ple_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
    }
    ggml_backend_tensor_get(h_emb_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
    ggml_backend_tensor_set(t_emb_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
}

// Layer 0's block and router on the current state, every state write suppressed.
void engine::spec_l0_route(int64_t T, bool is_ple, std::vector<int32_t> & ids) {
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count;
    assert_that(T >= 1 && T <= 2, "spec_l0_route: one or two tokens");
    assert_that(ids.size() == (size_t) hp_.n_expert_used * T, "spec_l0_route: room for every routed id");
    const graph_ctx gc = new_graph_ctx(4096);
    ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
    graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, n_past_);
    gb.set_gpu_fusion(w_.on_gpu(), t_hcmean_);
    gb.set_persist(false);                         // read the state, write nothing
    ggml_tensor * r = ggml_repeat_4d(c, ggml_reshape_3d(c, ggml_view_2d(c, t_emb_, n_embd, T, t_emb_->nb[1], 0), n_embd, 1, T),
                                     n_embd, hc, T, 1);
    r = ggml_reshape_3d(c, r, n_embd, hc, T);
    if (is_ple) r = gb.ple(ggml_view_2d(c, t_ple_, n_embd, T, t_ple_->nb[1], 0), r, 0);
    const hc_mixed ma = gb.hc_mix(r, 0, /*ffn=*/false, true);
    ggml_tensor * cur = gb.deltanet(ma.out, 0);
    r = gb.hc_combine(r, cur, ma.inject);
    ggml_tensor * cur2 = gb.hc_mix(r, 0, /*ffn=*/true, true).out;
    ggml_tensor * sl = gb.moe_route(cur2, 0).sel;
    ggml_tensor * slc = ggml_cont(c, sl);
    ggml_build_forward_expand(g, slc);
    run_on(g, true);
    ggml_backend_tensor_get(slc, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_free(c);
}

void engine::save_work_set(work_set & ws) const {
    ws.res[0] = res_[0]; ws.res[1] = res_[1];
    ws.cur = t_cur_; ws.emb = t_emb_; ws.sh = t_sh_; ws.pg = t_pg_; ws.pc = t_pc_; ws.ple = t_ple_;
    ws.inject = t_inject_; ws.sel = t_sel_; ws.w = t_w_; ws.tok = inp_tok_; ws.pos = inp_pos_; ws.plei = inp_ple_;
}
void engine::load_work_set(const work_set & ws) {
    res_[0] = ws.res[0]; res_[1] = ws.res[1];
    t_cur_ = ws.cur; t_emb_ = ws.emb; t_sh_ = ws.sh; t_pg_ = ws.pg; t_pc_ = ws.pc; t_ple_ = ws.ple;
    t_inject_ = ws.inject; t_sel_ = ws.sel; t_w_ = ws.w; inp_tok_ = ws.tok; inp_pos_ = ws.pos; inp_ple_ = ws.plei;
}

// A streamed prefill is about to run: hand the expert tier's dynamic buffer
// back to the device, then take the prefill's memory from it -- the
// n_batch work set, the MoE chunk allocator; the staging follows on the first
// load_layer. Undone by prefill_leave().
bool engine::prefill_enter(std::string & err) {
    if (in_prefill_) return true;
    assert_that(pwctx_ == nullptr && pwbuf_ == nullptr, "prefill_enter: no prefill work set yet");
    assert_that(galloc_pf_dyn_ == nullptr, "prefill_enter: no prefill allocator yet");
    ec_.lend_begin();
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count, B = cfg_.n_batch, U = hp_.n_expert_used, PH = hp_.ple_n_head();
    ggml_init_params wp{}; wp.mem_size = ggml_tensor_overhead() * 64; wp.no_alloc = true;
    pwctx_ = ggml_init(wp); assert_that(pwctx_ != nullptr, "ggml_init: pwctx_");
    work_set ws;
    ws.res[0] = ggml_new_tensor_3d(pwctx_, GGML_TYPE_F32, n_embd, hc, B);
    ws.res[1] = ggml_new_tensor_3d(pwctx_, GGML_TYPE_F32, n_embd, hc, B);
    ws.cur    = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    ws.emb    = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    ws.sh     = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    ws.pg     = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    ws.pc     = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    ws.ple    = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    ws.inject = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, hc, B);
    ws.sel    = ggml_new_tensor_2d(pwctx_, GGML_TYPE_I32, U, B);
    ws.w      = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, U, B);
    ws.tok    = ggml_new_tensor_1d(pwctx_, GGML_TYPE_I32, B);
    ws.pos    = ggml_new_tensor_1d(pwctx_, GGML_TYPE_I32, B * 4);
    ws.plei   = ggml_new_tensor_1d(pwctx_, GGML_TYPE_I32, PH * B);
    pwbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(pwctx_, w_.buft());
    if (!pwbuf_) { ggml_free(pwctx_); pwctx_ = nullptr; ec_.lend_end(); err = "failed to allocate the prefill work set"; return false; }
    load_work_set(ws);
    if (pf_.on_device()) { galloc_pf_dyn_ = ggml_gallocr_new(w_.buft()); galloc_pf_ = galloc_pf_dyn_; assert_that(galloc_pf_dyn_ != nullptr, "ggml_gallocr_new: prefill"); }
    in_prefill_ = true;
    return true;
}

void engine::prefill_leave() {
    if (!in_prefill_) return;
    assert_that(pwctx_ != nullptr && pwbuf_ != nullptr, "prefill_leave: the prefill work set is in place");
    assert_that(dec_ws_.cur != nullptr, "prefill_leave: the decode work set was saved");
    ec_.settle_promotions();
    pf_.release_device();
    pf_.forget_tier_copies();   // the tier may reuse those slots from here on
    if (galloc_pf_dyn_) { ggml_gallocr_free(galloc_pf_dyn_); galloc_pf_dyn_ = nullptr; galloc_pf_ = nullptr; }
    // The shared allocator grew to the attention chunk's arena; start it over
    // so decode's graphs get a small one.
    if (galloc_gpu_) { ggml_gallocr_free(galloc_gpu_); galloc_gpu_ = ggml_gallocr_new(w_.buft()); assert_that(galloc_gpu_ != nullptr, "ggml_gallocr_new: gpu"); }
    load_work_set(dec_ws_);
    if (pwbuf_) { ggml_backend_buffer_free(pwbuf_); pwbuf_ = nullptr; }
    if (pwctx_) { ggml_free(pwctx_); pwctx_ = nullptr; }
    in_prefill_ = false;
    ec_.lend_end();
    client_lent_ = false;
    sync_tier_epoch();
}

void engine::sync_tier_epoch() {
    assert_that(gM_.size() == gA_.size() && gA_bucket_.size() == gA_.size(), "sync_tier_epoch: per-layer tables sized together");
    assert_that(tier_epoch_seen_ <= ec_.tier_epoch(), "sync_tier_epoch: the tier epoch only advances");
    if (ec_.tier_epoch() == tier_epoch_seen_) return;
    // The dynamic tier moved: every replayed graph that folds an expert
    // matmul over it holds stale pointers. Rebuild them all next token.
    for (auto & lg : gA_) { if (lg.ga) ggml_gallocr_free(lg.ga); if (lg.ctx) ggml_free(lg.ctx); lg = layer_graph{}; }
    for (auto & mg : gM_) { if (mg.ga) ggml_gallocr_free(mg.ga); if (mg.ctx) ggml_free(mg.ctx); mg = moe_graph{}; }
    std::fill(gA_bucket_.begin(), gA_bucket_.end(), -1);
    tier_epoch_seen_ = ec_.tier_epoch();
}

void engine::vram_lend_begin() {
    assert_that(mi_ != nullptr, "vram_lend_begin: engine initialised");
    assert_that(tier_epoch_seen_ <= ec_.tier_epoch(), "vram_lend_begin: the tier epoch only advances");
    if (in_prefill_) return;               // already lent, and the prefill returns it
    ec_.lend_begin();
    client_lent_ = true;
}

void engine::vram_lend_end() {
    assert_that(mi_ != nullptr, "vram_lend_end: engine initialised");
    assert_that(tier_epoch_seen_ <= ec_.tier_epoch(), "vram_lend_end: the tier epoch only advances");
    if (in_prefill_ || !client_lent_) return;
    ec_.lend_end();
    client_lent_ = false;
    sync_tier_epoch();
}

// Adds the time since `t` to `acc` when it goes out of scope, if `on`.
struct scoped_add_time {
    double & acc; bool on; std::chrono::steady_clock::time_point t;
    ~scoped_add_time() {
        assert_that(acc >= 0.0, "accumulated time is non-negative");
        if (on) acc += secs_since(t);
        assert_that(acc >= 0.0, "accumulated time stays non-negative");
    }
};

// Short-lived per-step graphs (inputs, CPU experts, head) share one persistent buffer: a
// 16384-node context is ~6 MB, and allocating it per call meant an mmap/munmap and fresh
// page faults several times per layer. One such context is alive at a time (asserted).
engine::graph_ctx engine::new_tmp_ctx() {
    if (tmp_ctx_busy_) { fprintf(stderr, "[qwfn] two temporary graph contexts at once\n"); abort(); }
    const size_t sz = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    if (tmp_ctx_buf_.size() < sz) tmp_ctx_buf_.resize(sz);
    assert_that(tmp_ctx_buf_.size() >= sz, "new_tmp_ctx: the shared buffer holds a context");
    ggml_init_params p{}; p.mem_size = sz; p.mem_buffer = tmp_ctx_buf_.data(); p.no_alloc = true;
    graph_ctx gc;
    gc.ctx = ggml_init(p); assert_that(gc.ctx != nullptr, "ggml_init: a graph context");
    gc.gf  = ggml_new_graph_custom(gc.ctx, 16384, false);
    tmp_ctx_busy_ = true;
    assert_that(gc.ctx != nullptr && gc.gf != nullptr, "new_tmp_ctx: context and graph");
    return gc;
}

void engine::free_tmp_ctx(ggml_context * c) {
    assert_that(c != nullptr, "free_tmp_ctx: a context");
    assert_that(tmp_ctx_busy_, "free_tmp_ctx: the temporary context is in use");
    ggml_free(c); tmp_ctx_busy_ = false;
}

// Token ids and PLE row indices of the T tokens ending `hist`, for the device
// graphs and for the host gather.
void engine::upload_tok_ple(const int32_t * hist, int32_t n_hist, int64_t T) {
    const int64_t PH = hp_.ple_n_head();
    assert_that(hist != nullptr && T >= 1, "upload_tok_ple: at least one token");
    assert_that((size_t) T * 4 <= ggml_nbytes(inp_tok_), "upload_tok_ple: the token input holds T");
    ggml_backend_tensor_set(inp_tok_, hist + (n_hist - T), 0, (size_t) T * 4);
    std::vector<int32_t> & rows = ple_rows_host_;
    rows.assign((size_t) PH * T, 0);
    for (int64_t i = 0; i < T; i++) {
        const ple_rows r = ple_rows_for(hp_, hist, n_hist, (n_hist - T) + i);
        for (uint32_t h = 0; h < r.n; h++) rows[i * PH + h] = (int32_t) r.row[h];
    }
    ggml_backend_tensor_set(inp_ple_,   rows.data(), 0, rows.size() * 4);
    ggml_backend_tensor_set(h_ple_idx_, rows.data(), 0, rows.size() * 4);
}

// PLE rows and token embeddings gathered on the host, then the PLE rows uploaded;
// the embeddings stay in h_emb_. `tmp`: in the shared temporary context.
void engine::gather_ple_emb(const int32_t * hist, int32_t n_hist, int64_t T, bool tmp) {
    const int64_t n_embd = hp_.n_embd, PH = hp_.ple_n_head();
    assert_that(hist != nullptr && T >= 1, "gather_ple_emb: at least one token");
    assert_that((size_t) n_embd * T <= xfer_.size(), "gather_ple_emb: the transfer buffer holds T rows");
    ggml_backend_tensor_set(h_tok_, hist + (n_hist - T), 0, (size_t) T * 4);
    const graph_ctx gc = tmp ? new_tmp_ctx() : new_graph_ctx(16384);
    ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
    ggml_tensor * idx = ggml_view_1d(c, h_ple_idx_, PH * T, 0);
    ggml_tensor * e = ggml_get_rows(c, wh_.get("per_layer_token_embd.weight"), idx);
    e = ggml_reshape_2d(c, e, hp_.d_ple * PH, T);
    ggml_build_forward_expand(g, ggml_cpy(c, e, ggml_view_2d(c, h_ple_, n_embd, T, h_ple_->nb[1], 0)));
    ggml_tensor * te = ggml_get_rows(c, wh_.get("token_embd.weight"), ggml_view_1d(c, h_tok_, T, 0));
    ggml_build_forward_expand(g, ggml_cpy(c, te, ggml_view_2d(c, h_emb_, n_embd, T, h_emb_->nb[1], 0)));
    run_on(g, false);
    if (tmp) free_tmp_ctx(c); else ggml_free(c);
    ggml_backend_tensor_get(h_ple_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
    ggml_backend_tensor_set(t_ple_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
}

bool engine::has_overrides(int64_t n_past, int64_t T) const {
    assert_that(n_past >= 0 && T >= 1, "has_overrides: a range of positions");
    assert_that(ov_.size() == ov_pos_.size() * (size_t) hp_.n_embd, "has_overrides: one row per substituted position");
    for (int32_t p : ov_pos_) if (p >= n_past && p < n_past + T) return true;
    return false;
}

// ---- embedding + wide residual ----------------------------------------
// Split in two when image embeddings have to be substituted: gather the
// token embeddings, patch the image rows on the host, then repeat into the
// hyper-connection streams.
void engine::upload_emb_residual(int64_t n_past, int64_t T, bool tmp) {
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count;
    assert_that(n_past >= 0 && T >= 1, "upload_emb_residual: a range of positions");
    assert_that((size_t) n_embd * T <= xfer_.size(), "upload_emb_residual: the transfer buffer holds T rows");
    const bool has_ov = has_overrides(n_past, T);
    ggml_backend_tensor_get(h_emb_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
    if (has_ov) {
        for (size_t k = 0; k < ov_pos_.size(); k++) {
            const int32_t p = ov_pos_[k];
            if (p < n_past || p >= n_past + T) continue;
            memcpy(xfer_.data() + (size_t) (p - n_past) * n_embd,
                   ov_.data()  + k * (size_t) n_embd, (size_t) n_embd * sizeof(float));
        }
    }
    ggml_backend_tensor_set(t_emb_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
    const graph_ctx gc = tmp ? new_tmp_ctx() : new_graph_ctx(16384);
    ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
    ggml_tensor * r = ggml_repeat_4d(c, ggml_reshape_3d(c, ggml_view_2d(c, t_emb_, n_embd, T, t_emb_->nb[1], 0), n_embd, 1, T),
                                     n_embd, hc, T, 1);
    ggml_build_forward_expand(g, ggml_cpy(c, r,
            ggml_view_3d(c, res_[0], n_embd, hc, T, res_[0]->nb[1], res_[0]->nb[2], 0)));
    run_on(g, true);
    if (tmp) free_tmp_ctx(c); else ggml_free(c);
}

// The head's row for the position before this batch (the previous batch's or the
// last decoded position's wide residual, still in t_hlast_) pairs with this batch's
// first token; without it the head would have a hole.
bool engine::mtp_head_gap(const int32_t * hist, int32_t n_hist, int64_t T, std::string & err) {
    assert_that(mtp_on_ && mtp_kv_valid_, "mtp_head_gap: the head's cache is complete so far");
    assert_that(n_past_ > 0 && T >= 1, "mtp_head_gap: a batch after earlier positions");
    if (mtp_have_h_ && mtp_h_rows_ >= 1) {
        if (!mtp_step(hist + (n_hist - T), 1, err)) return false;
    } else {
        mtp_kv_valid_ = false;
    }
    return true;
}

struct engine::prefill_ctx {
    const int32_t * hist = nullptr;
    int32_t n_hist = 0;
    int64_t T = 0, Tm = 0, n_past = 0;
    int     sections[4] = { 0, 0, 0, 0 };
    int     cur_res = 0;
    bool    pending = false;
};

// Layer-major prefill of T (<= n_batch) tokens: for every layer, graph A over
// all T tokens in compute chunks (attention layers chunked further to the
// n_kv * T cap), then ONE sweep of the layer's experts serving all T through
// mul_mat_id in the same chunks. The sweep is what a prefill costs, so its
// count per prompt goes from ceil(n / ubatch), with the ubatch shrinking as
// the context grows, to ceil(n / n_batch).
bool engine::eval_prefill_big(const int32_t * hist, int32_t n_hist, int32_t T, std::string & err) {
    assert_that(in_prefill_, "eval_prefill_big: the prefill work set is in place");
    assert_that(T >= 1 && T <= (int32_t) cfg_.n_batch && n_past_ + T <= (int32_t) cfg_.n_ctx, "eval_prefill_big: a batch that fits");
    prefill_ctx p;
    p.hist = hist; p.n_hist = n_hist; p.T = T;
    p.n_past = n_past_;
    const auto t0 = std::chrono::steady_clock::now();
    pool_dirty_ = true;
    for (int k = 0; k < 4; k++) p.sections[k] = hp_.mrope_sections[k];
    p.Tm = std::min<int64_t>(T, std::max<uint32_t>(64, cfg_.prefill_chunk));

    if (mtp_on_ && mtp_kv_valid_ && p.n_past > 0 && !mtp_head_gap(hist, n_hist, T, err)) return false;

    // ---- inputs for all T tokens ------------------------------------------
    upload_tok_ple(hist, n_hist, T);
    gather_ple_emb(hist, n_hist, T, /*tmp=*/false);
    upload_emb_residual(p.n_past, T, /*tmp=*/false);

    for (uint32_t il = 0; il < hp_.n_layer; il++)
        if (!prefill_layer(p, il, err)) return false;

    // ---- draft head: its KV rows for this batch's positions ---------------
    // Position i pairs the folded residual with the embedding of token i+1, which
    // this batch holds for every position but its last; that one waits in
    // t_hlast_ for the next batch's first token (or the first sampled token).
    if (mtp_on_ && mtp_kv_valid_ && T >= 2) prefill_mtp_kv(p);

    prefill_head(p);
    ec_.settle_promotions();
    if (mtp_on_) { mtp_have_h_ = true; mtp_h_rows_ = 1; }
    n_past_ += T;
    const double dt = secs_since(t0);
    t_prefill += dt; n_prefill += T;
    return true;
}

bool engine::prefill_layer(prefill_ctx & p, uint32_t il, std::string & err) {
    const int64_t U = hp_.n_expert_used, T = p.T;
    assert_that(il < hp_.n_layer, "prefill_layer: layer in range");
    assert_that(T >= 1 && p.Tm >= 1, "prefill_layer: tokens and a chunk size");
    const bool is_ple  = std::find(hp_.ple_layers.begin(), hp_.ple_layers.end(), (int32_t) il) != hp_.ple_layers.end();
    const bool is_attn = hp_.is_attn_layer(il);

    // ---- graph A over all T, chunked ------------------------------------
    for (int64_t off = 0; off < T; ) {
        int64_t Tc = std::min<int64_t>(p.Tm, T - off);
        if (is_attn) Tc = std::min<int64_t>(Tc, max_ubatch((int32_t) (p.n_past + off)));
        if (!prefill_chunk_a(p, il, off, Tc, is_ple, err)) return false;
        off += Tc;
    }
    p.cur_res = 1 - p.cur_res;
    p.pending = true;
    ggml_backend_tensor_get(t_sel_, sel_.data(), 0, (size_t) U * T * sizeof(int32_t));
    ggml_backend_tensor_get(t_w_,   wgt_.data(), 0, (size_t) U * T * sizeof(float));

    if (!prefill_layer_moe(p, il, err)) return false;
    ggml_backend_tensor_set(t_pg_, zeros_.data(), 0, (size_t) hp_.n_embd * T * sizeof(float));

    if (cfg_.prefill_warm > 0) prefill_warm_layer(p, il);
    prefill_progress_.store(((uint64_t) (il + 1) << 48) | ((uint64_t) hp_.n_layer << 32) | (uint32_t) T,
                            std::memory_order_relaxed);
    return true;
}

bool engine::prefill_chunk_a(prefill_ctx & p, uint32_t il, int64_t off, int64_t Tc, bool is_ple, std::string & err) {
    assert_that(il < hp_.n_layer, "prefill_chunk_a: layer in range");
    assert_that(off >= 0 && Tc >= 1 && off + Tc <= p.T, "prefill_chunk_a: a chunk inside the batch");
    const bool is_attn = hp_.is_attn_layer(il);
    const int64_t n_past_c = p.n_past + off;
    attn_inputs ai;
    if (is_attn && !build_attn_inputs(n_past_c, Tc, ai, err)) return false;
    set_positions(inp_pos_, n_past_c, Tc, pos_host_);
    const auto ta0 = std::chrono::steady_clock::now();
    const graph_ctx gc = new_graph_ctx(16384);
    ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
    graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, n_past_c);
    ggml_tensor * r = view_res(c, res_[p.cur_res], off, Tc);
    if (p.pending) {
        ggml_tensor * tot = ggml_add(c, ggml_add(c, view_rows(c, t_sh_, off, Tc), view_rows(c, t_pg_, off, Tc)), view_rows(c, t_pc_, off, Tc));
        r = gb.hc_combine(r, tot, view_rows(c, t_inject_, off, Tc));
    }
    if (is_ple) r = gb.ple(view_rows(c, t_ple_, off, Tc), r, il);
    const hc_mixed ma = gb.hc_mix(r, il, /*ffn=*/false, true);
    ggml_tensor * cur = ma.out;
    cur = is_attn
        ? gb.sparse_attn(cur, ggml_view_1d(c, inp_pos_, Tc * 4, 0), ai.kq_mask, p.sections, il, ai.ratio ? &ai.qsa : nullptr)
        : gb.deltanet(cur, il);
    r = gb.hc_combine(r, cur, ma.inject);
    const hc_mixed mf = gb.hc_mix(r, il, /*ffn=*/true, true);
    ggml_tensor * cur2 = mf.out, * inject = mf.inject;
    const moe_routing rt = gb.moe_route(cur2, il);
    ggml_tensor * sl = rt.sel, * wt = rt.w;
    ggml_tensor * sh = gb.shared_expert(cur2, il);
    ggml_build_forward_expand(g, ggml_cpy(c, r,      view_res(c, res_[1 - p.cur_res], off, Tc)));
    ggml_build_forward_expand(g, ggml_cpy(c, cur2,   view_rows(c, t_cur_, off, Tc)));
    ggml_build_forward_expand(g, ggml_cpy(c, inject, view_rows(c, t_inject_, off, Tc)));
    ggml_build_forward_expand(g, ggml_cpy(c, sl,     view_rows(c, t_sel_, off, Tc)));
    ggml_build_forward_expand(g, ggml_cpy(c, wt,     view_rows(c, t_w_, off, Tc)));
    ggml_build_forward_expand(g, ggml_cpy(c, sh,     view_rows(c, t_sh_, off, Tc)));
    run_on(g, true);
    ggml_free(c);
    ai.release();
    t_pf_graphA += secs_since(ta0);
    return true;
}

// ---- MoE: one sweep of this layer's experts, all T tokens -----------
bool engine::prefill_layer_moe(prefill_ctx & p, uint32_t il, std::string & err) {
    assert_that(il < hp_.n_layer, "prefill_layer_moe: layer in range");
    assert_that(p.T >= 1 && p.Tm >= 1, "prefill_layer_moe: tokens and a chunk size");
    if (pf_.on_device()) ggml_backend_synchronize(w_.backend());
    const auto tr0 = std::chrono::steady_clock::now();
    if (!pf_.load_layer(il, err)) return false;
    t_pf_read += secs_since(tr0);
    pf_.prefetch_layer((il + 1) % hp_.n_layer, il + 1 < hp_.n_layer);
    const bool on_gpu = pf_.on_device();
    const tier_view staged = pf_.staged_tier();
    for (int64_t off = 0; off < p.T; off += p.Tm) {
        const int64_t Tc = std::min<int64_t>(p.Tm, p.T - off);
        if (!prefill_chunk_moe(off, Tc, staged, on_gpu, err)) return false;
    }
    return true;
}

bool engine::prefill_chunk_moe(int64_t off, int64_t Tc, const tier_view & staged, bool on_gpu, std::string & err) {
    const int64_t n_embd = hp_.n_embd, U = hp_.n_expert_used;
    assert_that(off >= 0 && Tc >= 1, "prefill_chunk_moe: a chunk of tokens");
    assert_that((size_t) (off + Tc) * U <= sel_.size(), "prefill_chunk_moe: the routing covers the chunk");
    const auto tm0 = std::chrono::steady_clock::now();
    if (!on_gpu) {
        ggml_backend_tensor_get(t_cur_, xfer_.data(), (size_t) off * n_embd * 4, (size_t) Tc * n_embd * 4);
        ggml_backend_tensor_set(h_cur_, xfer_.data(), (size_t) off * n_embd * 4, (size_t) Tc * n_embd * 4);
    }
    const graph_ctx gc = new_graph_ctx(16384);
    ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
    ggml_tensor * t_ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, U, Tc);   ggml_set_input(t_ids);
    ggml_tensor * t_wt  = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, U, Tc); ggml_set_input(t_wt);
    ggml_tensor * x   = on_gpu ? view_rows(c, t_cur_, off, Tc) : view_rows(c, h_cur_, off, Tc);
    ggml_tensor * out = on_gpu ? view_rows(c, t_pc_,  off, Tc) : view_rows(c, h_partial_, off, Tc);
    ggml_tensor * o = moe_id_graph(c, staged, t_ids, t_wt, x, n_embd, hp_.n_ff_exp, (int) U, Tc);
    ggml_build_forward_expand(g, ggml_cpy(c, o, out));
    if (!ggml_gallocr_alloc_graph(on_gpu ? galloc_pf_ : galloc_cpu_, g)) {
        err = "prefill galloc failed"; ggml_free(c); return false;
    }
    ggml_backend_tensor_set(t_ids, sel_.data() + (size_t) off * U, 0, (size_t) U * Tc * sizeof(int32_t));
    ggml_backend_tensor_set(t_wt,  wgt_.data() + (size_t) off * U, 0, (size_t) U * Tc * sizeof(float));
    if (ggml_backend_graph_compute(on_gpu ? w_.backend() : wh_.backend(), g) != GGML_STATUS_SUCCESS) {
        err = "prefill compute failed"; ggml_free(c); return false;
    }
    ggml_free(c);
    if (!on_gpu) {
        ggml_backend_tensor_get(h_partial_, xfer_.data(), (size_t) off * n_embd * 4, (size_t) Tc * n_embd * 4);
        ggml_backend_tensor_set(t_pc_, xfer_.data(), (size_t) off * n_embd * 4, (size_t) Tc * n_embd * 4);
    }
    t_pf_moe += secs_since(tm0);
    return true;
}

void engine::prefill_warm_layer(const prefill_ctx & p, uint32_t il) {
    const int64_t U = hp_.n_expert_used, T = p.T;
    assert_that(il < hp_.n_layer, "prefill_warm_layer: layer in range");
    assert_that((size_t) T * U <= sel_.size(), "prefill_warm_layer: the routing covers the batch");
    const auto tw0 = std::chrono::steady_clock::now();
    std::vector<expert_cache::warm_item> items;
    std::vector<int32_t> pos(hp_.n_expert, -1);
    for (int64_t t = T - 1; t >= 0; t--) {
        for (int64_t j = 0; j < U; j++) {
            const int32_t e = sel_[t * U + j];
            if (e < 0 || e >= (int32_t) hp_.n_expert) continue;
            if (pos[e] >= 0) { items[pos[e]].count++; continue; }
            if (items.size() >= cfg_.prefill_warm) continue;
            expert_cache::warm_item it;
            it.expert = (uint32_t) e; it.count = 1;
            for (int q = 0; q < EXPERT_NPARTS; q++) it.part[q] = pf_.host_part_ptr((uint32_t) e, (expert_part) q);
            pos[e] = (int32_t) items.size();
            items.push_back(it);
        }
    }
    ec_.warm(il, items.data(), (uint32_t) items.size(), cfg_.prefill_warm_vram);
    t_warm += secs_since(tw0);
}

void engine::prefill_mtp_kv(prefill_ctx & p) {
    const int64_t T = p.T;
    assert_that(mtp_on_ && mtp_kv_valid_, "prefill_mtp_kv: the head's cache is complete so far");
    assert_that(T >= 2 && p.pending, "prefill_mtp_kv: a batch with folded layers");
    const auto tk0 = std::chrono::steady_clock::now();
    const int64_t Tk = std::min<int64_t>(p.Tm, 1024);
    for (int64_t off = 0; off + 1 < T; off += Tk) {
        const int64_t n = std::min<int64_t>(Tk, T - 1 - off);
        set_positions(inp_pos_, p.n_past + off, n, pos_host_);
        const graph_ctx gc = new_graph_ctx(16384);
        ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
        graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, p.n_past + off);
        ggml_tensor * r = view_res(c, res_[p.cur_res], off, n);
        ggml_tensor * tot = ggml_add(c, ggml_add(c, view_rows(c, t_sh_, off, n), view_rows(c, t_pg_, off, n)), view_rows(c, t_pc_, off, n));
        r = gb.hc_combine(r, tot, view_rows(c, t_inject_, off, n));
        graph_builder gm(c, &hpm_, &wm_, &w_); gm.bind(&st_mtp_, g, p.n_past + off);
        gm.mtp_head_kv(r, view_rows(c, t_emb_, off + 1, n), ggml_view_1d(c, inp_pos_, 4 * n, 0), p.sections, (int) hpm_.n_layer - 1);
        run_on(g, true);
        ggml_free(c);
    }
    t_mtp += secs_since(tk0);
}

// ---- head: logits for the last position -------------------------------
void engine::prefill_head(const prefill_ctx & p) {
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count, T = p.T;
    assert_that(p.pending && T >= 1, "prefill_head: after the layers");
    assert_that(logits_.size() >= (size_t) n_vocab_, "prefill_head: room for one position's logits");
    const graph_ctx gc = new_graph_ctx(16384);
    ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
    graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, p.n_past);
    ggml_tensor * r = view_res(c, res_[p.cur_res], T - 1, 1);
    ggml_tensor * tot = ggml_add(c, ggml_add(c, view_rows(c, t_sh_, T - 1, 1), view_rows(c, t_pg_, T - 1, 1)), view_rows(c, t_pc_, T - 1, 1));
    r = gb.hc_combine(r, tot, view_rows(c, t_inject_, T - 1, 1));
    if (mtp_on_)   // the last position's wide residual, for its head row and the first draft
        ggml_build_forward_expand(g, ggml_cpy(c, r, ggml_view_3d(c, t_hlast_, n_embd, hc, 1, t_hlast_->nb[1], t_hlast_->nb[2], 0)));
    ggml_tensor * o = gb.hc_mix(r, -1, false, false).out;
    ggml_tensor * logits = ggml_mul_mat(c, w_.get("output.weight"), o);
    ggml_set_output(logits);
    ggml_build_forward_expand(g, logits);
    run_on(g, true);
    ggml_backend_tensor_get(logits, logits_.data(), 0, (size_t) n_vocab_ * sizeof(float));   // one position: logits_ holds two for a decoded pair
    ggml_free(c);
}

// eval_batch's call-wide state.
struct engine::step_ctx {
    const int32_t * hist = nullptr;
    int32_t n_hist = 0;
    int64_t T = 0, n_past = 0, n_kv = 0;
    bool    decode = false, cbatch = false, force_decode = false, use_qd = false;
    int     sections[4] = { 0, 0, 0, 0 };
    // Per-call attention inputs of the mask-based path, freed with the step.
    ggml_context *        ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    ggml_tensor *         kq_mask = nullptr;
    qsa_inputs            qsa;
    uint32_t              ratio = 0;
    int     cur_res = 0;
    bool    pending = false;
    std::vector<expert_handle> & eh;   // a step's union of experts: the engine's eh_ scratch
    std::chrono::steady_clock::time_point t0, tf_begin;

    explicit step_ctx(std::vector<expert_handle> & handles) : eh(handles) {}

    ggml_tensor * v2(ggml_context * c, ggml_tensor * t) const { return view_rows(c, t, 0, T); }
    void release_inputs() {
        assert_that(!ibuf || ictx, "step inputs: a buffer only with its context");
        assert_that(T >= 0, "step inputs: a token count");
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (ictx) ggml_free(ictx);
        ibuf = nullptr; ictx = nullptr;
    }
    ~step_ctx() { release_inputs(); }
};

ggml_tensor * engine::vres(ggml_context * c, const step_ctx & s, int which) const {
    return view_res(c, res_[which], 0, s.T);
}
ggml_tensor * engine::vpos(ggml_context * c, const step_ctx & s) const {
    return ggml_view_1d(c, inp_pos_, s.T * 4, 0);
}

// Decode MoE on the GPU: by default one mul_mat_id per part over the VRAM
// tier, computed INSIDE the next layer's graph A (which folds it into the
// residual anyway). That removes a graph submission and a stream sync per
// layer, and keeps the number of live CUDA graphs at the 36 replayed
// layers -- ggml caches at most 64, and a separate persistent MoE graph
// per layer (84 total) made it evict and never capture anything.
// Cold-file blocks get their own mul_mat_id pass (part_rows).
// In-graph VRAM MoE: graph A of layer L runs L's VRAM-resident routed
// experts itself. The router's ids index two device tables (tier slot and
// 1/0 mask by expert id, uploaded whenever the layer's residency changes)
// so no host round trip stands between the router and the expert
// matmuls, and the residual passed to the next layer's router prediction
// then misses only the CPU-served experts. Same mul_mat_id kernels, same
// summation order as the host-driven graph: a non-resident expert points
// at slot 0 with weight 0 either way.
bool engine::moe_in_graph(bool decode, uint32_t il) const {
    return decode && moe_in_graph_ && ec_.gpu_tier(il).n_slots > 0;
}

// In-place graph outputs. Give the node that produces an output the persistent
// tensor's memory (at dst_off bytes) instead of copying into it afterwards:
// gallocr leaves a node that already has data alone, views derive theirs from
// it, and the backend computes straight into it. Every read of the same
// persistent tensor in this graph -- the fold of the previous layer's partials
// and activation -- precedes the write in dependency order. One token only,
// contiguous, same type, and the node must be the view root (offset 0).
static bool place(int64_t T, ggml_tensor * t, ggml_tensor * dst, size_t dst_off) {
    assert_that(t != nullptr, "place: a graph node");
    assert_that(T >= 1, "place: at least one position");
    if (T > 2 || !dst || !dst->data || !dst->buffer) return false;
    ggml_tensor * root = t; size_t off = 0;
    while (root->view_src) { off += root->view_offs; root = root->view_src; }
    if (off != 0 || root->data || root->type != dst->type || !ggml_is_contiguous(root)) return false;
    if (ggml_nbytes(root) + dst_off > ggml_nbytes(dst)) return false;
    root->data   = (char *) dst->data + dst_off;
    root->buffer = dst->buffer;
    // An output: the allocator must never hand its memory to a child (a
    // single-child op that can run in place would otherwise write over it),
    // and it must be computed even when nothing else in the graph reads it.
    ggml_set_output(root);
    return true;
}

void engine::upload_vtable(bool decode, uint32_t il) {
    assert_that(il <= hp_.n_layer, "upload_vtable: layer in range (or one past the last)");
    assert_that(!moe_in_graph_ || vslot_ver_.size() == hp_.n_layer, "upload_vtable: one version per layer");
    if (!decode || !moe_in_graph_ || il >= hp_.n_layer) return;
    if (ec_.gpu_tier(il).n_slots == 0 || ec_.vram_version(il) == vslot_ver_[il]) return;
    const size_t ne = hp_.n_expert;
    if (p_vslot_) {
        // Pinned staging, queued on the compute stream: after the promotion
        // copies it describes, before the graph that reads it. One buffer
        // is enough: the previous layer's graph ran synchronously.
        ec_.vram_table(il, (int32_t *) p_vslot_->data, (float *) p_vmask_->data);
        ggml_backend_tensor_set_async(w_.backend(), t_vslot_[il], p_vslot_->data, 0, ne * sizeof(int32_t));
        ggml_backend_tensor_set_async(w_.backend(), t_vmask_[il], p_vmask_->data, 0, ne * sizeof(float));
    } else {
        vt_slot_.assign(ne, 0); vt_mask_.assign(ne, 0.0f);
        ec_.vram_table(il, vt_slot_.data(), vt_mask_.data());
        ggml_backend_tensor_set(t_vslot_[il], vt_slot_.data(), 0, ne * sizeof(int32_t));
        ggml_backend_tensor_set(t_vmask_[il], vt_mask_.data(), 0, ne * sizeof(float));
    }
    vslot_ver_[il] = ec_.vram_version(il);
}

// The late fold of layer `prev`: its promoted experts, by slot, from
// t_gids_/t_gw_ (zeros when there were none: slot 0 with weight 0).
// Decode only: the batched paths compute every expert of a layer themselves
// and leave t_gids_/t_gw_ holding the last decode step's values, and the
// per-position view below is sized for a decode step's T of 1 or 2, not a
// prompt's (a 27-token batch read 1 KB past the two-column tensors: garbage
// ids and weights folded into every position, gibberish from the next turn on).
ggml_tensor * engine::late_fold(const step_ctx & s, ggml_context * c, uint32_t prev, ggml_tensor * pg) {
    assert_that(c != nullptr && pg != nullptr, "late_fold: a graph and the partial it adds to");
    assert_that(prev < hp_.n_layer, "late_fold: layer in range");
    if (!s.decode || !moe_in_graph(s.decode, prev) || n_late_ <= 0) return pg;
    const int64_t T = s.T;
    ggml_tensor * ids = ggml_view_2d(c, t_gids_, n_late_, T, t_gids_->nb[1], 0);
    ggml_tensor * w   = ggml_view_3d(c, t_gw_, 1, n_late_, T, t_gw_->nb[1], t_gw_->nb[2], 0);
    return ggml_add(c, pg, moe_id_graph(c, ec_.gpu_tier(prev), ids, w, t_cur_, hp_.n_embd, hp_.n_ff_exp, n_late_, T, /*fused_sum=*/true));
}

bool engine::eval_batch(const int32_t * hist, int32_t n_hist, int32_t T, std::string & err,
                        bool cbatch, bool force_decode) {
    // Callers check the context first; a missed check is still an error, not an abort.
    if (n_past_ + T > (int32_t) cfg_.n_ctx) { err = "context exhausted"; return false; }
    assert_that(T >= 1 && hist != nullptr && n_hist >= T, "eval_batch: T tokens of history to evaluate");
    assert_that(logits_.size() >= (size_t) n_vocab_ * (1 + MTP_MAX_DRAFTS), "eval_batch: room for every position's logits");
    step_ctx s(eh_);
    s.hist = hist; s.n_hist = n_hist; s.T = T; s.force_decode = force_decode;
    s.n_past = n_past_; s.n_kv = s.n_past + T;
    s.decode = T == 1 || (force_decode && T <= 1 + MTP_MAX_DRAFTS);   // a verify step: a token and its drafts
    s.tf_begin = std::chrono::steady_clock::now();   // step profile (decode): pre / head / total
    scoped_add_time eval_timer_{t_eval_decode, s.decode, s.tf_begin};
    // A step is either a decode (a token and its drafts) or a short prompt served through the
    // expert cache in one batch; a batched prefill goes through eval_prefill_big. Checked
    // before anything is mutated.
    s.cbatch = cbatch && !s.decode && T > 1 && scr_buf_ != nullptr;
    if (!s.decode && !s.cbatch) { err = "eval_batch: a batched prefill must go through eval_prefill_big"; return false; }
    // A later turn: the head's row for the last decoded position pairs its
    // wide residual (still in t_hlast_) with this batch's first token.
    if (mtp_on_ && T > 1 && !force_decode && s.n_past > 0 && mtp_kv_valid_ && !mtp_head_gap(hist, n_hist, T, err)) return false;
    s.t0 = std::chrono::steady_clock::now();
    // Decode attention with per-token cost flat in the context; anything else
    // (prefill, dense attention) uses the mask-based path and leaves the pooled
    // block keys to be rebuilt on the next decode token.
    s.use_qd = s.decode && qbuf_ != nullptr && cfg_.use_qsa;
    if (!s.decode) pool_dirty_ = true;
    upload_vtable(s.decode, 0);
    for (int k = 0; k < 4; k++) s.sections[k] = hp_.mrope_sections[k];

    // ---- per-call inputs --------------------------------------------------
    upload_step_inputs(s);
    if (!build_step_inputs(s, err)) return false;

    // ---- PLE rows and token embeddings gathered on the host, then uploaded --
    gather_ple_emb(hist, n_hist, T, /*tmp=*/true);
    upload_emb_residual(s.n_past, T, /*tmp=*/true);

    if (s.decode) t_pre += secs_since(s.tf_begin);
    s.cur_res = 0; s.pending = false;
    s.eh.assign((size_t) (1 + MTP_MAX_DRAFTS) * hp_.n_expert_used, expert_handle{});
    pred_.clear(); pred2_a_.clear(); pred2_b_.clear();

    for (uint32_t il = 0; il < hp_.n_layer; il++)
        if (!eval_layer(s, il, err)) return false;
    if (!s.decode) ec_.settle_promotions();   // any promotions the batched step issued have landed

    if (deferred_wait_) { if (!ec_.fetch_end()) { err = "expert read failed"; return false; } deferred_wait_ = false; }

    step_head(s);
    if (mtp_on_ && !step_mtp(s, err)) return false;
    rb_depth_ = (force_decode && T >= 2 && rbbuf_ != nullptr) ? (int) std::min<int64_t>(T - 1, rb_nsnap_) : 0;
    rb_valid_ = rb_depth_ > 0;
    s.release_inputs();
    n_past_ += T;

    const double dt = secs_since(s.t0);
    if (s.decode) { t_decode += dt; n_decode += T; } else { t_prefill += dt; n_prefill += T; }
    return true;
}

void engine::upload_step_inputs(step_ctx & s) {
    const int64_t T = s.T;
    assert_that(T >= 1 && s.hist != nullptr, "upload_step_inputs: at least one token");
    assert_that(s.n_past >= 0, "upload_step_inputs: a valid position");
    upload_tok_ple(s.hist, s.n_hist, T);

    // M-RoPE positions are section-major: all t, then all h, then all w, then 0.
    set_positions(inp_pos_, s.n_past, T, pos_host_);
    if (inp_pos_one_ && T <= 1 + MTP_MAX_DRAFTS) {   // [p, p, p, 0] per position, for single-position attention calls
        int32_t p1[4 * (1 + MTP_MAX_DRAFTS)] = { 0 };
        for (int64_t i = 0; i < T; i++) p1[i * 4] = p1[i * 4 + 1] = p1[i * 4 + 2] = (int32_t) (s.n_past + i);
        ggml_backend_tensor_set(inp_pos_one_, p1, 0, (size_t) T * 4 * sizeof(int32_t));
    }
}

// Shapes vary with n_kv, so the mask and the QSA inputs are built per call --
// except on the decode QSA path, which needs neither.
bool engine::build_step_inputs(step_ctx & s, std::string & err) {
    const int64_t T = s.T, n_kv = s.n_kv, n_past = s.n_past;
    const bool use_qd = s.use_qd;
    assert_that(T >= 1 && n_kv == n_past + T, "build_step_inputs: the step's key range");
    assert_that(s.ictx == nullptr && s.ibuf == nullptr, "build_step_inputs: built once per step");
    const auto t_in0 = std::chrono::steady_clock::now();
    ggml_init_params ip{}; ip.mem_size = ggml_tensor_overhead() * 16; ip.no_alloc = true;
    s.ictx = use_qd ? nullptr : ggml_init(ip);   // rule 3 deviation: ggml's per-step input context, see docs/CODING_RULES.md
    assert_that(use_qd || s.ictx != nullptr, "ggml_init: step inputs");
    s.kq_mask = use_qd ? nullptr : ggml_new_tensor_2d(s.ictx, GGML_TYPE_F16, n_kv, T);

    uint32_t ratio = 0;
    if (cfg_.use_qsa) {
        for (uint32_t il = 0; il < hp_.n_layer; il++)
            if (hp_.is_attn_layer(il) && il < hp_.compress_ratios.size() && hp_.compress_ratios[il] > 0)
                { ratio = (uint32_t) hp_.compress_ratios[il]; break; }
    }
    s.ratio = ratio;
    const int64_t n_blocks = ratio ? (n_kv + ratio - 1) / ratio : 0;
    if (ratio && !use_qd) {
        s.qsa.cell_blk  = ggml_new_tensor_1d(s.ictx, GGML_TYPE_I32, n_kv);
        s.qsa.blk_cells = ggml_new_tensor_1d(s.ictx, GGML_TYPE_I32, ratio * n_blocks);
        s.qsa.blk_pos   = ggml_new_tensor_1d(s.ictx, GGML_TYPE_I32, 4 * n_blocks);
        s.qsa.bias      = ggml_new_tensor_2d(s.ictx, GGML_TYPE_F32, n_blocks, T);
        s.qsa.ratio     = ratio;
        s.qsa.n_blocks  = n_blocks;
    }
    s.ibuf = use_qd ? nullptr : ggml_backend_alloc_ctx_tensors_from_buft(s.ictx, w_.buft());
    if (!use_qd && !s.ibuf) { ggml_free(s.ictx); s.ictx = nullptr; err = "failed to allocate per-call inputs"; return false; }

    if (use_qd) qsa_decode_prepare((int32_t) n_past);

    if (use_qd && s.decode) for (int64_t k = 1; k < T; k++) qsa_decode_prepare_k((int) k, (int32_t) (n_past + k));
    if (!use_qd) {
        std::vector<uint16_t> m((size_t) n_kv * T, f16_of(-INFINITY));
        for (int64_t i = 0; i < T; i++)
            for (int64_t j = 0; j <= n_past + i; j++) m[i * n_kv + j] = f16_of(0.0f);
        ggml_backend_tensor_set(s.kq_mask, m.data(), 0, m.size() * 2);
    }
    if (ratio && !use_qd) fill_step_qsa(s, n_blocks);
    if (s.decode) t_inputs += secs_since(t_in0);
    return true;
}

void engine::fill_step_qsa(step_ctx & s, int64_t n_blocks) {
    const int64_t T = s.T, n_kv = s.n_kv, n_past = s.n_past;
    const uint32_t ratio = s.ratio;
    assert_that(ratio > 0 && s.ibuf != nullptr, "fill_step_qsa: allocated block tables");
    assert_that(n_blocks == (n_kv + ratio - 1) / ratio, "fill_step_qsa: blocks cover the keys");
    const int64_t n_bid = n_kv / ratio;                   // only whole blocks are pooled
    const bool have_dead = n_bid < n_blocks;
    const int64_t dead = have_dead ? n_bid : n_blocks - 1;
    std::vector<int32_t> cb(n_kv), bc((size_t) ratio * n_blocks, 0), bp((size_t) 4 * n_blocks, 0);
    std::vector<float>   bi((size_t) n_blocks * T);
    for (int64_t j = 0; j < n_kv; j++)
        cb[j] = (int32_t) (j < n_bid * (int64_t) ratio ? j / ratio : dead);
    for (int64_t b = 0; b < n_bid; b++) {
        for (uint32_t k = 0; k < ratio; k++) bc[b * ratio + k] = (int32_t) (b * ratio + k);
        for (int sec = 0; sec < 4; sec++) bp[sec * n_blocks + b] = (int32_t) (b * ratio);
    }
    // The partial tail block, if any: its real cells (the last one repeated) and position, as decode
    // maps it. Left at zero it pointed at cell 0, and a query inside it -- the last 1-3 tokens of a chunk
    // that ends mid-block -- could not attend to itself or the cells just before it.
    if (n_bid < n_blocks) {
        for (uint32_t k = 0; k < ratio; k++)
            bc[n_bid * ratio + k] = (int32_t) std::min<int64_t>(n_bid * (int64_t) ratio + k, n_kv - 1);
        for (int sec = 0; sec < 4; sec++) bp[sec * n_blocks + n_bid] = (int32_t) (n_bid * ratio);
    }
    for (int64_t i = 0; i < T; i++) {
        const int64_t q = n_past + i;
        const int64_t tail = ((q + 1) / ratio) * ratio;   // the ragged tail stays visible
        for (int64_t b = 0; b < n_blocks; b++)
            bi[i * n_blocks + b] = (b >= n_bid) ? -INFINITY
                                 : (b * (int64_t) ratio >= tail ? 1e9f : 0.0f);
        if (have_dead) bi[i * n_blocks + dead] = 1e9f;
        // blocks wholly after q are not forced (see build_attn_inputs)
        for (int64_t b = tail / ratio; b < n_blocks; b++)
            if (bi[i * n_blocks + b] > 0.0f && b * (int64_t) ratio > q) bi[i * n_blocks + b] = -1e9f;
    }
    ggml_backend_tensor_set(s.qsa.cell_blk,  cb.data(), 0, cb.size() * 4);
    ggml_backend_tensor_set(s.qsa.blk_cells, bc.data(), 0, bc.size() * 4);
    ggml_backend_tensor_set(s.qsa.blk_pos,   bp.data(), 0, bp.size() * 4);
    ggml_backend_tensor_set(s.qsa.bias,      bi.data(), 0, bi.size() * 4);
}

// One layer of a step: graph A (replayed or built), the routing read back, the
// prediction scored, then the routed experts (decode or the cache-served batch).
bool engine::eval_layer(step_ctx & s, uint32_t il, std::string & err) {
    const int64_t T = s.T;
    assert_that(il < hp_.n_layer, "eval_layer: layer in range");
    assert_that(T >= 1 && (s.decode || s.cbatch), "eval_layer: a decode step or a cache-served batch");
    const bool is_ple = std::find(hp_.ple_layers.begin(), hp_.ple_layers.end(), (int32_t) il)
                      != hp_.ple_layers.end();
    // ---- graph A (GPU) -------------------------------------------------
    // Replayable only for the recurrent layers, and only during decode: the
    // sparse-attention layers reshape with n_kv every token.
    const bool replayable = cfg_.reuse_graphs && s.decode && (!hp_.is_attn_layer(il) || s.use_qd);
    // An attention graph is shaped by its block bucket; a new bucket means a new graph.
    // Readback pack (t_pack_): decode, one token, the GPU MoE in the graph so
    // nothing on the device needs t_sel_/t_w_ afterwards. Decided here so the
    // build and the readback of a cached graph agree.
    // Every verify width (T <= 1 + MTP_MAX_DRAFTS) too: the pack is sized for the
    // widest step and laid out per T; one D2H copy instead of five.
    const bool pack_ok = t_pack_ && s.decode && T <= 1 + MTP_MAX_DRAFTS && moe_in_graph(s.decode, il);
    // A graph that speculatively runs an attention successor's block is shaped by the bucket too.
    const bool spec_attn_next = cfg_.spec_block && s.decode && s.use_qd && il + 1 < hp_.n_layer
                                && hp_.is_attn_layer(il + 1) && spec_block_mask_[il + 1];
    if (replayable && gA_[il].gf && (((hp_.is_attn_layer(il) || spec_attn_next) && gA_bucket_[il] != qd_.n_bucket) || gA_T_[il] != (uint8_t) T)) {
        if (gA_[il].ga)  ggml_gallocr_free(gA_[il].ga);
        if (gA_[il].ctx) ggml_free(gA_[il].ctx);
        gA_[il] = layer_graph{};
    }
    // Whether the graph that runs for this layer writes the readback pack:
    // the cached graph's own property when it is replayed, this build's
    // otherwise. gA_pack_ describes the CACHED graph only. (It used to be
    // overwritten by every build, including a later turn's prompt batch,
    // whose graphs never pack; the next decode then replayed turn 1's
    // packing graphs and read the routing from tensors they never write:
    // stale expert ids in every layer, fast garbage, surviving reset.)
    bool ran_packed = false;
    if (replayable && gA_[il].gf) {
        ran_packed = gA_pack_[il] != 0;
        if (!replay_layer_a(il, err)) return false;
    } else {
        if (!build_layer_a(s, il, is_ple, replayable, pack_ok, err)) return false;
        ran_packed = pack_ok;
    }
    s.cur_res = 1 - s.cur_res;
    s.pending = true;

    const bool packed = s.decode && ran_packed;
    read_routing(s, packed);
    score_predictions(s, il);

    if (s.decode) return decode_moe(s, il, packed, err);
    if (s.cbatch) return cbatch_moe(s, il, err);
    qwfn::assert_that(false, "eval_batch: neither a decode step nor a cache-served batch");
    return true;
}

bool engine::replay_layer_a(uint32_t il, std::string & err) {
    assert_that(il < gA_.size(), "replay_layer_a: layer in range");
    assert_that(gA_[il].gf != nullptr && gA_[il].ga != nullptr, "replay_layer_a: a cached graph and its allocator");
    const auto ta0 = std::chrono::steady_clock::now();
    // The allocator's pass re-assigns the same addresses every time for a
    // cached graph (22 us a layer); once is enough. The CUDA backend then
    // compares every node's properties on every replay (44-91 us a layer)
    // unless the graph carries the uid it recorded: it does, set when cached.
    if (!gA_[il].allocated) {
        if (!ggml_gallocr_alloc_graph(gA_[il].ga, gA_[il].gf)) { err = "replay alloc failed"; return false; }
        gA_[il].allocated = true;
    }
    const auto ta1 = std::chrono::steady_clock::now();
    // Split the replay: the allocator's pass, the launch (the property check
    // over every node and the graph submit), the wait for the device.
    if (ggml_backend_graph_compute_async(w_.backend(), gA_[il].gf) != GGML_STATUS_SUCCESS) {
        err = "replay compute failed"; return false;
    }
    const auto ta2 = std::chrono::steady_clock::now();
    ggml_backend_synchronize(w_.backend());
    const auto ta3 = std::chrono::steady_clock::now();
    t_replay_alloc  += std::chrono::duration<double>(ta1 - ta0).count();
    t_replay_launch += std::chrono::duration<double>(ta2 - ta1).count();
    t_replay_wait   += std::chrono::duration<double>(ta3 - ta2).count();
    n_replay++;
    const double dtA = std::chrono::duration<double>(ta3 - ta0).count();
    t_layerA += dtA;
    if (hp_.is_attn_layer(il)) { t_layerA_attn += dtA; n_layerA_attn++; }
    else                       { t_layerA_rec  += dtA; n_layerA_rec++;  }
    return true;
}

// The tensors of one layer's graph A that its later parts use.
struct engine::layer_a {
    ggml_context * c = nullptr;
    ggml_cgraph  * g = nullptr;
    ggml_tensor  * r = nullptr, * inject = nullptr, * cur2 = nullptr, * sl = nullptr, * wt = nullptr,
                 * sh = nullptr, * pg_here = nullptr;
    std::chrono::steady_clock::time_point ta0;
};

bool engine::build_layer_a(step_ctx & s, uint32_t il, bool is_ple, bool replayable, bool pack_ok, std::string & err) {
    assert_that(il < hp_.n_layer, "build_layer_a: layer in range");
    assert_that(s.T >= 1 && (!pack_ok || t_pack_ != nullptr), "build_layer_a: the pack exists when used");
    layer_a la;
    la.ta0 = std::chrono::steady_clock::now();
    const graph_ctx gc = new_graph_ctx(16384);
    la.c = gc.ctx; la.g = gc.gf;
    graph_builder gb(la.c, &hp_, &w_); gb.bind(&st_, la.g, s.n_past);
    gb.set_gpu_fusion(w_.on_gpu(), t_hcmean_);
    if (s.T >= 2 && rbbuf_) { gb.set_rollback(rb_rs_[il], rb_conv_[il], rb_nsnap_); gb.set_rollback_ple(rb_ple_conv_); }
    layer_a_core(s, gb, la, il, is_ple);
    layer_a_speculate(s, gb, la, il, pack_ok);
    layer_a_outputs(s, la, pack_ok);
    return layer_a_run(s, la, il, replayable, pack_ok, err);
}

// Decode attention for the layer's T positions: one call, or for a
// pair two chained calls so the second reads through the first's writes.
ggml_tensor * engine::attn_decode(const step_ctx & s, graph_builder & gbx, ggml_context * c, ggml_tensor * x, uint32_t l) {
    const int64_t T = s.T;
    assert_that(l < pool_cache_.size(), "attn_decode: a pooled key cache per layer");
    assert_that(T >= 1 && T <= 1 + MTP_MAX_DRAFTS, "attn_decode: a token and at most its drafts");
    qd_.pool_cache = pool_cache_[l];
    if (T == 1) return gbx.sparse_attn_decode(x, vpos(c, s), s.sections, (int) l, qd_);
    // T positions as T chained calls: each reads through the writes of the ones before.
    // The per-position projections (no dependence on earlier positions) run once
    // for all T.
    graph_builder::qsa_chain ch;
    graph_builder::qsa_proj pre = gbx.qsa_decode_proj(x, vpos(c, s), s.sections, (int) l);
    ggml_tensor * out = nullptr;
    for (int64_t k = 0; k < T; k++) {
        qsa_decode_inputs & q = k == 0 ? qd_ : qdk_[k - 1];
        q.pool_cache = pool_cache_[l];
        ggml_tensor * xk = ggml_view_2d(c, x, hp_.n_embd, 1, x->nb[1], (size_t) k * x->nb[1]);
        ggml_tensor * ok = gbx.sparse_attn_decode(xk, ggml_view_1d(c, inp_pos_one_, 4, (size_t) k * 4 * sizeof(int32_t)), s.sections, (int) l, q, &ch,
                                                  &pre, k);
        out = out ? ggml_concat(c, out, ok, 1) : ok;
    }
    return gbx.qsa_out_proj(out, (int) l);
}

// The layer's block up to its routing: the previous layer's MoE folded in, PLE,
// the token mixer, the router and the shared expert, and the in-graph VRAM MoE.
void engine::layer_a_core(const step_ctx & s, graph_builder & gb, layer_a & la, uint32_t il, bool is_ple) {
    ggml_context * c = la.c;
    const int64_t U = hp_.n_expert_used, T = s.T;
    assert_that(c != nullptr && la.g != nullptr, "layer_a_core: a graph context");
    assert_that(il < hp_.n_layer, "layer_a_core: layer in range");
    ggml_tensor * r = vres(c, s, s.cur_res);
    if (s.pending) {
        ggml_tensor * pg = il > 0 ? late_fold(s, c, il - 1, s.v2(c, t_pg_)) : s.v2(c, t_pg_);
        ggml_tensor * tot = t_rscale_
            ? ggml_add(c, s.v2(c, t_sh_), ggml_mul(c, ggml_add(c, pg, s.v2(c, t_pc_)), t_rscale_))
            : ggml_add(c, ggml_add(c, s.v2(c, t_sh_), pg), s.v2(c, t_pc_));
        r = gb.hc_combine(r, tot, s.v2(c, t_inject_));
    }
    if (is_ple) r = gb.ple(s.v2(c, t_ple_), r, il);

    const hc_mixed ma = gb.hc_mix(r, il, /*ffn=*/false, true);
    ggml_tensor * cur = ma.out;
    if (!hp_.is_attn_layer(il)) {
        cur = gb.deltanet(cur, il);
    } else if (s.use_qd) {
        cur = attn_decode(s, gb, c, cur, il);
    } else {
        cur = gb.sparse_attn(cur, vpos(c, s), s.kq_mask, s.sections, il, s.ratio ? &s.qsa : nullptr);
    }
    r = gb.hc_combine(r, cur, ma.inject);

    const hc_mixed mf = gb.hc_mix(r, il, /*ffn=*/true, true);
    ggml_tensor * cur2 = mf.out, * inject = mf.inject;
    gb.gate_drop = s.decode ? cfg_.gate_drop : 0.0f;
    const moe_routing rt = gb.moe_route(cur2, il);
    ggml_tensor * sl = rt.sel, * wt = rt.w;
    gb.gate_drop = 0.0f;
    ggml_tensor * sh = gb.shared_expert(cur2, il);

    ggml_tensor * pg_here = nullptr;
    if (moe_in_graph(s.decode, il)) {
        ggml_tensor * slc = ggml_is_contiguous(sl) ? sl : ggml_cont(c, sl);
        ggml_tensor * wtc = ggml_is_contiguous(wt) ? wt : ggml_cont(c, wt);
        ggml_tensor * sl1  = ggml_reshape_1d(c, slc, U * T);
        ggml_tensor * slot = ggml_reshape_2d(c, ggml_get_rows(c, t_vslot_[il], sl1), U, T);      // [U, T]
        ggml_tensor * mask = ggml_reshape_3d(c, ggml_get_rows(c, t_vmask_[il], sl1), 1, U, T);   // [1, U, T]
        ggml_tensor * w    = ggml_mul(c, ggml_reshape_3d(c, wtc, 1, U, T), mask);
        pg_here = moe_id_graph(c, ec_.gpu_tier(il), slot, w, cur2, hp_.n_embd, hp_.n_ff_exp, (int) U, T, /*fused_sum=*/true);
        if (place(T, pg_here, t_pg_, 0)) ggml_build_forward_expand(la.g, pg_here);
        else ggml_build_forward_expand(la.g, ggml_cpy(c, pg_here, s.v2(c, t_pg_)));
    }
    la.r = r; la.inject = inject; la.cur2 = cur2; la.sl = sl; la.wt = wt; la.sh = sh; la.pg_here = pg_here;
}

// Speculate the next layer's routing from `r`, which is this layer's
// residual before its own MoE lands. Costs one extra hc_mix plus a
// 2560x512 router matmul on the GPU; buys reads that overlap this
// layer's MoE instead of stalling the next one. The two-ahead
// prediction applies layer L+2's mixer and router to the same
// residual -- missing two MoE contributions instead of one, so less
// accurate, but its reads get two layers of compute to land in.
void engine::layer_a_speculate(const step_ctx & s, graph_builder & gb, layer_a & la, uint32_t il, bool pack_ok) {
    ggml_context * c = la.c;
    const int64_t T = s.T;
    assert_that(c != nullptr && la.r != nullptr, "layer_a_speculate: after the layer's block");
    assert_that(il < hp_.n_layer, "layer_a_speculate: layer in range");
    if (!(cfg_.speculate && s.decode && il + 1 < hp_.n_layer)) return;
    // The residual `r` is missing this layer's whole FFN output. The
    // shared expert -- always on, already computed above -- is a
    // large, known part of it; with the VRAM MoE in this graph, the
    // routed sum is mostly known too (90% of routed experts are
    // VRAM-served at 128K). So with the in-graph MoE, predict from the
    // residual missing only the CPU-served experts. Without it, predict
    // from the bare residual: folding the shared expert in alone raised
    // accuracy only 81.2% -> 81.6% and cost an hc_combine per layer in
    // the replayed graph.
    ggml_tensor * rp = la.r;
    if (la.pg_here) rp = gb.hc_combine(la.r, ggml_add(c, la.sh, la.pg_here), la.inject);
    const uint32_t iln = il + 1;
    rp = spec_block_residual(s, gb, c, rp, iln);
    // Every candidate up to QWFN_SPEC_MAX comes back with its logit;
    // how many are read is decided on the host (speculate_depth, then
    // the margin gate).
    const int depth = (int) QWFN_SPEC_MAX;
    const route_prediction pr = gb.moe_route_predict(rp, (int) iln, depth, true);
    spec_outputs(s, la, pr.ids, pr.scores, pack_ok);
    if (cfg_.speculate_ahead >= 2 && il + 2 < hp_.n_layer) {
        ggml_tensor * selnext2 = gb.moe_route_predict(rp, (int) il + 2, depth).ids;
        ggml_build_forward_expand(la.g, ggml_cpy(c, selnext2,
                ggml_view_2d(c, t_selnext2_, depth, T, t_selnext2_->nb[1], 0)));
    }
}

// Speculative block (cfg spec_block): the residual predictor's
// remaining error is layer L+1's own block, which it cannot see.
// Run that block here, on this residual -- L+1's PLE injection if
// it has one, its attention mixer, its DeltaNet or decode sparse
// attention, its combine -- with every state write suppressed, and
// predict from the residual it produces. The exact graph of L+1
// recomputes the block on the exact residual and does the writes;
// an attention block's cache rows written here are rewritten at the
// same position there before anything else reads them.
ggml_tensor * engine::spec_block_residual(const step_ctx & s, graph_builder & gb, ggml_context * c, ggml_tensor * rp, uint32_t iln) {
    assert_that(c != nullptr && rp != nullptr, "spec_block_residual: a residual to predict from");
    assert_that(iln < hp_.n_layer && iln < spec_block_mask_.size(), "spec_block_residual: predicted layer in range");
    const bool spec_blk = cfg_.spec_block && spec_block_mask_[iln] && (!hp_.is_attn_layer(iln) || s.use_qd);
    if (!spec_blk) return rp;
    gb.set_persist(false);
    const bool ple_next = std::find(hp_.ple_layers.begin(), hp_.ple_layers.end(), (int32_t) iln) != hp_.ple_layers.end();
    if (ple_next) rp = gb.ple(s.v2(c, t_ple_), rp, (int) iln);
    const hc_mixed m1 = gb.hc_mix(rp, (int) iln, /*ffn=*/false, true);
    ggml_tensor * c1 = m1.out, * inj2 = m1.inject;
    ggml_tensor * blk = nullptr;
    if (!hp_.is_attn_layer(iln)) {
        blk = gb.deltanet(c1, (int) iln);
    } else {
        blk = attn_decode(s, gb, c, c1, iln);
    }
    gb.set_persist(true);
    return gb.hc_combine(rp, blk, inj2);
}

void engine::spec_outputs(const step_ctx & s, layer_a & la, ggml_tensor * selnext, ggml_tensor * scores, bool pack_ok) {
    ggml_context * c = la.c;
    ggml_cgraph * g = la.g;
    const int64_t n_embd = hp_.n_embd, U = hp_.n_expert_used, T = s.T;
    const int depth = (int) QWFN_SPEC_MAX;
    assert_that(selnext != nullptr && scores != nullptr, "spec_outputs: the predicted ids and their logits");
    assert_that(c != nullptr && g != nullptr, "spec_outputs: a graph context");
    if (pack_ok) {   // into the readback pack (I32 -> F32 for the ids), see t_pack_
        ggml_build_forward_expand(g, ggml_cpy(c, selnext,
                ggml_view_1d(c, t_pack_, depth * T, (size_t) (T * n_embd + 2 * U * T) * sizeof(float))));
        if (place(T, scores, t_cur_, (size_t) (T * n_embd + 2 * U * T + depth * T) * sizeof(float))) ggml_build_forward_expand(g, scores);
        else ggml_build_forward_expand(g, ggml_cpy(c, scores,
                ggml_view_1d(c, t_pack_, depth * T, (size_t) (T * n_embd + 2 * U * T + depth * T) * sizeof(float))));
    } else {
        ggml_build_forward_expand(g, ggml_cpy(c, selnext,
                ggml_view_2d(c, t_selnext_, depth, T, t_selnext_->nb[1], 0)));
        ggml_build_forward_expand(g, ggml_cpy(c, scores,
                ggml_view_2d(c, t_specscore_, depth, T, t_specscore_->nb[1], 0)));
    }
}

// Outputs in place where the shapes allow (one token): the node that
// produces an output gets the persistent tensor's memory, so the copy
// kernel goes. See `place` above.
void engine::layer_a_outputs(const step_ctx & s, layer_a & la, bool pack_ok) {
    ggml_context * c = la.c;
    ggml_cgraph * g = la.g;
    const int64_t n_embd = hp_.n_embd, U = hp_.n_expert_used, T = s.T;
    assert_that(c != nullptr && g != nullptr, "layer_a_outputs: a graph context");
    assert_that(la.r && la.cur2 && la.inject && la.sl && la.wt && la.sh, "layer_a_outputs: the layer's outputs");
    ggml_tensor * r = la.r, * cur2 = la.cur2, * inject = la.inject, * sl = la.sl, * wt = la.wt, * sh = la.sh;
    ggml_build_forward_expand(g, place(T, r, res_[1 - s.cur_res], 0) ? r : ggml_cpy(c, r, vres(c, s, 1 - s.cur_res)));
    ggml_build_forward_expand(g, place(T, cur2, t_cur_, 0) ? cur2 : ggml_cpy(c, cur2, s.v2(c, t_cur_)));   // row 0 of t_cur_ is the pack's head
    ggml_build_forward_expand(g, place(T, inject, t_inject_, 0) ? inject : ggml_cpy(c, inject, s.v2(c, t_inject_)));
    if (pack_ok) {
        ggml_build_forward_expand(g, place(T, wt, t_cur_, (size_t) T * n_embd * sizeof(float)) ? wt
                : ggml_cpy(c, wt, ggml_view_1d(c, t_pack_, U * T, (size_t) T * n_embd * sizeof(float))));
        ggml_build_forward_expand(g, ggml_cpy(c, sl, ggml_view_1d(c, t_pack_, U * T, (size_t) (T * n_embd + U * T) * sizeof(float))));   // I32 -> F32: a real conversion
    } else {
        ggml_build_forward_expand(g, ggml_cpy(c, sl,     s.v2(c, t_sel_)));
        ggml_build_forward_expand(g, ggml_cpy(c, wt,     s.v2(c, t_w_)));
    }
    ggml_build_forward_expand(g, place(T, sh, t_sh_, 0) ? sh : ggml_cpy(c, sh, s.v2(c, t_sh_)));
}

// Keep the context alive and give the graph its own allocator, so
// the addresses it was built against do not move next token. If
// the device cannot spare the buffer, fall back to the shared
// allocator rather than failing: replay is an optimisation.
bool engine::cache_layer_a(const step_ctx & s, layer_a & la, uint32_t il, bool pack_ok, bool & cached, std::string & err) {
    assert_that(il < gA_.size() && gA_[il].gf == nullptr, "cache_layer_a: no cached graph for the layer yet");
    assert_that(la.c != nullptr && la.g != nullptr, "cache_layer_a: a built graph");
    ggml_cgraph * g = la.g;
    ggml_gallocr_t ga = ggml_gallocr_new(w_.buft()); assert_that(ga != nullptr, "ggml_gallocr_new: ga");
    if (ga && ggml_gallocr_alloc_graph(ga, g)) {
        gA_[il].ctx = la.c; gA_[il].gf = g; gA_[il].ga = ga;
        // A nonzero uid unique to this build: the CUDA backend skips its
        // per-replay property walk while it sees the same uid, and a rebuilt
        // graph (new T or bucket) gets a new one.
        static uint64_t graph_uid = 0;
        g->uid = ++graph_uid;
        gA_bucket_[il] = qd_.n_bucket; gA_T_[il] = (uint8_t) s.T; gA_pack_[il] = pack_ok;
        cached = true;
        if (ggml_backend_graph_compute(w_.backend(), g) != GGML_STATUS_SUCCESS) {
            err = "compute failed"; return false;
        }
    } else {
        if (ga) ggml_gallocr_free(ga);
        static bool warned = false;
        if (!warned) { warned = true;
            fprintf(stderr, "[qwfn] graph replay disabled: no device memory for the per-layer allocators\n"); }
    }
    return true;
}

bool engine::layer_a_run(const step_ctx & s, layer_a & la, uint32_t il, bool replayable, bool pack_ok, std::string & err) {
    ggml_context * c = la.c;
    ggml_cgraph * g = la.g;
    assert_that(c != nullptr && g != nullptr, "layer_a_run: a built graph");
    assert_that(il < hp_.n_layer, "layer_a_run: layer in range");
    bool cached = false;
    if (replayable && !cache_layer_a(s, la, il, pack_ok, cached, err)) return false;
    if (!cached) {
        const auto tb = std::chrono::steady_clock::now();
        if (!ggml_gallocr_alloc_graph(galloc_gpu_, g)) {
            // Out of device memory for this graph: fail the request, not the process.
            err = "out of VRAM for a " + std::to_string(s.T) + "-token batch at " + std::to_string(n_past_) +
                  " tokens of context (layer " + std::to_string(il) + "); lower the batch or raise the reserve";
            fprintf(stderr, "[qwfn] %s\n", err.c_str());
            ggml_free(c);
            return false;
        }
        const auto tc = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(w_.backend(), g) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[qwfn] compute failed\n"); abort();
        }
        const auto td = std::chrono::steady_clock::now();
        if (s.decode && hp_.is_attn_layer(il)) {
            t_attn_build   += std::chrono::duration<double>(tb - la.ta0).count()
                            + std::chrono::duration<double>(tc - tb).count();
            t_attn_compute += std::chrono::duration<double>(td - tc).count();
        }
        ggml_free(c);
    }
    const double dtA = secs_since(la.ta0);
    t_layerA += dtA;
    if (s.decode) {
        if (hp_.is_attn_layer(il)) { t_layerA_attn += dtA; n_layerA_attn++; }
        else                       { t_layerA_rec  += dtA; n_layerA_rec++;  }
    }
    return true;
}

void engine::read_routing(const step_ctx & s, bool packed) {
    const int64_t n_embd = hp_.n_embd, U = hp_.n_expert_used, T = s.T;
    assert_that(T >= 1 && (size_t) U * T <= sel_.size(), "read_routing: room for the step's routing");
    assert_that(!packed || (size_t) T * (n_embd + 2 * U + 2 * QWFN_SPEC_MAX) <= pack_host_.size(), "read_routing: the pack fits its host copy");
    if (packed) {
        const size_t pn = (size_t) T * (n_embd + 2 * U + 2 * QWFN_SPEC_MAX);
        ggml_backend_tensor_get(t_pack_, pack_host_.data(), 0, pn * sizeof(float));
        const float * pk = pack_host_.data();
        ggml_backend_tensor_set(h_cur_, pk, 0, (size_t) T * n_embd * sizeof(float));   // host tensor: a memcpy
        const float * pw = pk + T * n_embd, * ps = pw + U * T, * pp = ps + U * T, * pc = pp + QWFN_SPEC_MAX * T;
        for (int64_t k = 0; k < U * T; k++) { wgt_[k] = pw[k]; sel_[k] = (int32_t) lrintf(ps[k]); }
        for (int64_t k = 0; k < (int64_t) QWFN_SPEC_MAX * T; k++) { pred_next_[k] = (int32_t) lrintf(pp[k]); scores_next_[k] = pc[k]; }
    } else {
        ggml_backend_tensor_get(t_sel_, sel_.data(), 0, (size_t) U * T * sizeof(int32_t));
        ggml_backend_tensor_get(t_w_,   wgt_.data(), 0, (size_t) U * T * sizeof(float));
    }
}

// Score the prediction made one layer ago against what actually happened.
// Only where one exists: at layer 0, pred_ still holds the prediction made
// at layer n-2 for layer n-1, already scored; counting it against layer 0's
// routing added ten near-certain misses per token to the reported rate.
void engine::score_predictions(const step_ctx & s, uint32_t il) {
    const int64_t U = hp_.n_expert_used, T = s.T;
    assert_that(il < hp_.n_layer && il < pred_hits_layer.size(), "score_predictions: layer in range");
    assert_that(pred_.empty() || pred_.size() >= (size_t) QWFN_SPEC_MAX * T, "score_predictions: a prediction per position");
    if (cfg_.speculate && s.decode && il > 0 && !pred_.empty()) {
        for (int64_t j = 0; j < T; j++) {
            const int32_t * pj = pred_.data() + j * QWFN_SPEC_MAX, * sj = sel_.data() + j * U;
            for (int64_t e = 0; e < U; e++) {
                pred_total++; pred_total_layer[il]++;
                for (int64_t k = 0; k < U; k++)
                    if (pj[k] == sj[e]) { pred_hits++; pred_hits_layer[il]++; break; }
            }
            // Precision by rank and by confidence margin: the gate's calibration data.
            for (size_t k = 0; k < QWFN_SPEC_MAX; k++) {
                bool ok = false;
                for (int64_t e = 0; e < U; e++) if (pj[k] == sj[e]) { ok = true; break; }
                rank_total[k]++; if (ok) rank_hits[k]++;
                const int b = margin_bucket(pred_margin_[j * QWFN_SPEC_MAX + k]);
                margin_total[b]++; if (ok) margin_hits[b]++;
            }
        }
    }
    // And the one made two layers ago, if the two-ahead path is on.
    if (cfg_.speculate && s.decode && !pred2_a_.empty()) {
        for (int64_t e = 0; e < U; e++) {
            pred2_total++;
            for (int64_t k = 0; k < U; k++)
                if (pred2_a_[k] == sel_[e]) { pred2_hits++; break; }
        }
    }
}


// The gate weight of distinct expert e (index into ids_) at position j; 0 where
// that position does not route to it.
float engine::w_tok(int64_t e, int64_t j) const {
    const int64_t U = hp_.n_expert_used;
    assert_that(e >= 0 && (size_t) e < ids_.size(), "w_tok: a distinct expert");
    assert_that(j >= 0 && (size_t) (j + 1) * U <= sel_.size(), "w_tok: a position of the step");
    for (int64_t k = 0; k < U; k++) if (sel_[j * U + k] == ids_[e]) return wgt_[j * U + k];
    return 0.0f;
}

// ---- decode: cache-served, one matmul per expert ----------------
bool engine::decode_moe(step_ctx & s, uint32_t il, bool packed, std::string & err) {
    const int64_t n_embd = hp_.n_embd, T = s.T;
    assert_that(s.decode && il < hp_.n_layer, "decode_moe: a decode step, layer in range");
    assert_that(s.eh.size() >= (size_t) hp_.n_expert_used * T, "decode_moe: a handle per routed expert");
    moe_layer & ml = *ml_;
    ml.reset(il, packed);
    moe_collect(s, ml);
    if (!moe_fetch(s, ml, err)) return false;

    // Resident now; the misses -- and this layer's speculative reads
    // that have not landed -- are streaming in behind these. The CPU
    // experts already here are computed while those land, and every CPU
    // expert's weighted row goes to one host buffer that is summed once,
    // in selection order: the same additions in the same order as one
    // pass over all of them, so the output is byte-identical to it (this
    // model is violently sensitive to accumulation order -- simply
    // reversing the ten additions changes the second generated token).
    // Skip-miss keeps the one-pass form after settling the speculative
    // reads first.
    //
    // Device placement is never negotiable: an expert resident in VRAM
    // must run on the GPU whatever the overlap, or the CPU backend
    // dereferences a device pointer.
    ml.two_pass = !t_rscale_;
    // In-flight promotions reach VRAM at the settle: the host-driven GPU MoE runs before fetch_end.
    if (!ml.two_pass || (ec_.has_deferred() && !moe_in_graph(s.decode, il))) {
        if (!ec_.settle_pending((const uint32_t *) ids_.data(), (uint32_t) ml.n_u, (bool *) ml.rdy.data())) {
            err = "expert read failed"; return false;
        }
    }
    moe_classify(s, ml);

    // Whether this layer has demand reads in flight. The speculative
    // reads for the next layer share the NVMe with them: issued first,
    // they take ~2/3 of the device's concurrency and a demand burst that
    // should land in ~0.4 ms takes ~1 ms. So when there are demand
    // misses, the speculative submission waits until they have landed.
    // 71% of layers have no miss and keep the full window.
    // With 3-draft verify steps, pinned CPU threads and the Q4K draft head
    // deferring wins: expert I/O wait on the code replay 5.75/5.76 s vs
    // 6.67-7.31 s early, output unchanged (it only moves reads).
    const bool had_miss = ec_.has_inflight();

    // VRAM-resident experts first: their kernels run behind everything
    // this thread does from here to the settle.
    moe_gpu_launch(s, ml, ml.on_gpu);

    if (!had_miss) issue_prefetch(s, ml);

    if (!packed) {
        ggml_backend_tensor_get(t_cur_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
        ggml_backend_tensor_set(h_cur_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
    }

    ml.nfl = (size_t) n_embd * T;
    ml.acc.assign(ml.nfl, 0.0f);
    if (!moe_cpu(s, ml, err)) return false;
    moe_finish(s, ml);
    return true;
}

// The experts of all T positions, each once (a pair shares about a
// third of them); per position a weight of 0 where it does not route there.
void engine::moe_collect(const step_ctx & s, moe_layer & ml) {
    const int64_t U = hp_.n_expert_used, T = s.T;
    assert_that((size_t) U * T <= sel_.size() && (size_t) U * T <= ids_.size(), "moe_collect: room for the step's experts");
    assert_that(ml.n_u == 0, "moe_collect: a fresh layer");
    int64_t n_u = 0;
    for (int64_t k = 0; k < U * T; k++) {
        const uint32_t e = (uint32_t) sel_[k]; bool dup = false;
        if (cfg_.gate_drop > 0.0f && wgt_[k] == 0.0f) { n_exp_dropped++; continue; }   // dropped by the router graph
        for (int64_t q = 0; q < n_u; q++) if ((uint32_t) ids_[q] == e) { dup = true; break; }
        if (!dup) ids_[n_u++] = e;
    }
    ml.n_u = n_u;
}

bool engine::moe_fetch(step_ctx & s, moe_layer & ml, std::string & err) {
    const uint32_t il = ml.il;
    assert_that(il < hp_.n_layer, "moe_fetch: layer in range");
    assert_that(ml.n_u >= 0 && (size_t) ml.n_u <= s.eh.size(), "moe_fetch: a handle per distinct expert");
    ml.rdy.assign(ml.n_u, 0);
    const auto ti = std::chrono::steady_clock::now();
    if (deferred_wait_) {   // the previous layer's skipped misses: their reads had a layer to land
        if (!ec_.fetch_end()) { err = "expert read failed"; return false; }
        deferred_wait_ = false;
    }
    const uint64_t reads0 = ec_.stats().n_reads;
    if (!ec_.fetch_begin(il, (const uint32_t *) ids_.data(), (uint32_t) ml.n_u,
                         s.eh.data(), (bool *) ml.rdy.data())) {
        err = "expert fetch failed"; return false;
    }
    {
        const double dti = secs_since(ti);
        t_io += dti;
        if (prof_io_begin.size() != hp_.n_layer) { prof_io_begin.assign(hp_.n_layer, 0.0); prof_io_end.assign(hp_.n_layer, 0.0); prof_reads.assign(hp_.n_layer, 0); }
        prof_io_begin[il] += dti; prof_reads[il] += ec_.stats().n_reads - reads0;
    }
    return true;
}

void engine::moe_classify(const step_ctx & s, moe_layer & ml) {
    assert_that(ml.rdy.size() == (size_t) ml.n_u, "moe_classify: readiness per distinct expert");
    assert_that(ml.on_gpu.empty() && ml.all_cpu.empty(), "moe_classify: a fresh layer");
    for (int64_t e = 0; e < ml.n_u; e++) {
        // `late` only matters to the in-graph path (its graph already ran);
        // the host-driven paths compute promoted experts like any other.
        if (s.eh[e].on_gpu) { (s.eh[e].late && moe_in_graph(s.decode, ml.il) ? ml.late_gpu : ml.on_gpu).push_back((int) e); continue; }
        ml.all_cpu.push_back((int) e);                       // selection order
        (ml.rdy[e] ? ml.ready_cpu : ml.late_cpu).push_back((int) e);
    }
}

// The rows of a group of CPU experts into h_wd_, at their positions in
// the selection order; then the one canonical sum over all of them.
bool engine::moe_cpu(step_ctx & s, moe_layer & ml, std::string & err) {
    const uint32_t il = ml.il;
    assert_that(il < hp_.n_layer && ml.acc.size() == ml.nfl, "moe_cpu: layer in range, the sum allocated");
    assert_that(ml.pos_in_all.empty(), "moe_cpu: a fresh layer");
    ml.pos_in_all.assign((size_t) ml.n_u, -1);
    for (size_t k = 0; k < ml.all_cpu.size(); k++) ml.pos_in_all[ml.all_cpu[k]] = (int) k;
    if (ml.two_pass) {
        part_rows(s, ml, ml.ready_cpu);    // the reads land during this
        const auto tw = std::chrono::steady_clock::now();
        if (!ec_.fetch_end()) { err = "expert read failed"; return false; }
        { const double dtw = secs_since(tw); t_io += dtw; prof_io_end[il] += dtw; }
        issue_prefetch(s, ml);             // no-op if already issued
        part_rows(s, ml, ml.late_cpu);
        reduce_rows(s, ml);
    } else if (t_rscale_ && !ml.late_cpu.empty()) {
        // skip_miss: compute with what is resident, renormalise the gates
        // of the experts that took part, and let the misses' reads land
        // while the next layer runs (waited for at its fetch).
        issue_prefetch(s, ml);
        moe_add_cpu(s, ml, ml.ready_cpu, true);
        float kept = 1.0f; for (int e : ml.late_cpu) kept -= wgt_[e];
        const float scale = kept > 1e-3f ? 1.0f / kept : 1.0f;
        ggml_backend_tensor_set(t_rscale_, &scale, 0, 4);
        n_exp_skipped += ml.late_cpu.size();
        deferred_wait_ = ec_.has_inflight();
    } else {
        const auto tw = std::chrono::steady_clock::now();
        if (!ec_.fetch_end()) { err = "expert read failed"; return false; }
        { const double dtw = secs_since(tw); t_io += dtw; prof_io_end[il] += dtw; }
        issue_prefetch(s, ml);             // no-op if already issued
        moe_add_cpu(s, ml, ml.all_cpu, true);   // one group, selection order
        if (t_rscale_) { const float one = 1.0f; ggml_backend_tensor_set(t_rscale_, &one, 0, 4); }
    }
    return true;
}

void engine::moe_finish(const step_ctx & s, moe_layer & ml) {
    const size_t nfl = ml.nfl;
    assert_that(ml.acc.size() == nfl && nfl == (size_t) hp_.n_embd * s.T, "moe_finish: the CPU sum covers the step");
    assert_that(ml.il < hp_.n_layer, "moe_finish: layer in range");
    moe_gpu_settle();                      // t_pg_ is complete past this point
    ec_.settle_promotions();               // the async H2D copies too; frees their RAM slots
    // The CPU partial goes up asynchronously from pinned memory: the copy
    // is queued on the compute stream ahead of the next layer's graph, and
    // the staging is next written only after that graph has run
    // synchronously, so the DMA has long completed. Pageable memory would
    // make cudaMemcpyAsync block the caller anyway (see the promotions).
    if (p_pc_ && nfl * sizeof(float) <= ggml_nbytes(p_pc_)) {
        memcpy(p_pc_->data, ml.acc.data(), nfl * sizeof(float));
        ggml_backend_tensor_set_async(w_.backend(), t_pc_, p_pc_->data, 0, nfl * sizeof(float));
    } else {
        ggml_backend_tensor_set(t_pc_, ml.acc.data(), 0, nfl * sizeof(float));
    }
    upload_vtable(s.decode, ml.il + 1);    // next layer's residency, after this layer's promotions
}

// Build one per-expert MoE graph over `which`, aliasing cached blocks
// so nothing is copied: the GPU fallback when the layer's mul_mat_id
// graph cannot be built. Returns the context; the caller runs it.
engine::graph_ctx engine::moe_expert_graph(const step_ctx & s, const std::vector<int> & which,
                                           ggml_tensor * in, ggml_tensor * out) {
    const int64_t n_embd = hp_.n_embd;
    assert_that(in != nullptr && out != nullptr, "moe_expert_graph: input and output");
    assert_that(!which.empty(), "moe_expert_graph: at least one expert");
    const graph_ctx gc = new_graph_ctx(16384);
    ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
    graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, s.n_past);
    std::vector<ggml_tensor *> & tg = sc_tg_, & tu = sc_tu_, & td = sc_td_;
    std::vector<float> & ww = sc_ww_;
    tg.clear(); tu.clear(); td.clear(); ww.clear();
    for (int e : which) {
        ggml_tensor * a[EXPERT_NPARTS];
        a[EXPERT_GATE] = ggml_new_tensor_2d(c, s.eh[e].type[EXPERT_GATE], n_embd, hp_.n_ff_exp);
        a[EXPERT_UP]   = ggml_new_tensor_2d(c, s.eh[e].type[EXPERT_UP],   n_embd, hp_.n_ff_exp);
        a[EXPERT_DOWN] = ggml_new_tensor_2d(c, s.eh[e].type[EXPERT_DOWN], hp_.n_ff_exp, n_embd);
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            a[q]->buffer = s.eh[e].buffer;
            a[q]->data   = (void *) s.eh[e].part[q];
        }
        tg.push_back(a[EXPERT_GATE]); tu.push_back(a[EXPERT_UP]); td.push_back(a[EXPERT_DOWN]);
        ww.push_back(wgt_[e]);
    }
    ggml_tensor * o = gb.moe_apply(s.v2(c, in), tg.data(), tu.data(), td.data(),
                                   ww.data(), (int) which.size());
    ggml_build_forward_expand(g, ggml_cpy(c, o, s.v2(c, out)));
    return gc;
}

// The CPU experts of `which`: one mul_mat_id per part over the RAM
// arena, experts by slot. Same kernels and summation order as one
// mul_mat per expert, but one activation quantisation instead of two
// per expert and a handful of barriers instead of ~15.
void engine::moe_part(const step_ctx & s, const moe_layer & ml, const std::vector<int> & which,
                      ggml_tensor * in, ggml_tensor * out) {
    const int64_t n_embd = hp_.n_embd, T = s.T;
    assert_that(in != nullptr && out != nullptr, "moe_part: input and output");
    assert_that(ml.il < hp_.n_layer, "moe_part: layer in range");
    const auto tm0 = std::chrono::steady_clock::now();
    if (which.empty()) {
        ggml_backend_tensor_set(out, zeros_.data(), 0, (size_t) n_embd * T * sizeof(float));
        return;
    }
    const tier_view tv = ec_.ram_tier(ml.il);
    const int n = (int) which.size();
    const graph_ctx gc = new_graph_ctx(16384);
    ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
    ggml_tensor * ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, n, T);   ggml_set_input(ids);
    ggml_tensor * w   = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, n, T); ggml_set_input(w);
    ggml_tensor * acc = moe_id_graph(c, tv, ids, w, in, n_embd, hp_.n_ff_exp, n, T);
    ggml_build_forward_expand(g, ggml_cpy(c, acc, s.v2(c, out)));
    if (!ggml_gallocr_alloc_graph(galloc_cpu_, g)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
    std::vector<int32_t> & sid = sc_sid_; std::vector<float> & sw = sc_sw_;
    sid.assign((size_t) n * T, 0); sw.assign((size_t) n * T, 0.0f);
    for (int64_t j = 0; j < T; j++)
        for (int k = 0; k < n; k++) { sid[j * n + k] = s.eh[which[k]].slot; sw[j * n + k] = w_tok(which[k], j); }
    ggml_backend_tensor_set(ids, sid.data(), 0, sid.size() * sizeof(int32_t));
    ggml_backend_tensor_set(w,   sw.data(),  0, sw.size() * sizeof(float));
    if (ggml_backend_graph_compute(wh_.backend(), g) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[qwfn] compute failed\n"); abort();
    }
    ggml_free(c);
    const double dt = secs_since(tm0);
    t_moe_cpu += dt; n_exp_cpu += which.size();
}

void engine::moe_add_cpu(const step_ctx & s, moe_layer & ml, const std::vector<int> & which, bool first) {
    const size_t nfl = ml.nfl;
    assert_that(ml.acc.size() == nfl && nfl <= xfer_.size(), "moe_add_cpu: the sum and the transfer buffer hold the step");
    assert_that(nfl == (size_t) hp_.n_embd * s.T, "moe_add_cpu: one row per position");
    if (which.empty()) return;
    moe_part(s, ml, which, h_cur_, h_partial_);
    if (first) {
        ggml_backend_tensor_get(h_partial_, ml.acc.data(), 0, nfl * sizeof(float));
    } else {
        ggml_backend_tensor_get(h_partial_, xfer_.data(), 0, nfl * sizeof(float));
        for (size_t k = 0; k < nfl; k++) ml.acc[k] += xfer_[k];
    }
}

// Launch the GPU MoE without waiting: the kernels execute while this
// thread waits on the NVMe and runs the CPU MoE. Settled below,
// before t_pg_ is needed.
void engine::moe_gpu_launch(const step_ctx & s, const moe_layer & ml, const std::vector<int> & which) {
    const uint32_t il = ml.il;
    const int64_t n_embd = hp_.n_embd, U = hp_.n_expert_used, T = s.T;
    assert_that(il < hp_.n_layer && il < gM_.size(), "moe_gpu_launch: layer in range");
    assert_that(which.size() <= (size_t) ml.n_u, "moe_gpu_launch: experts of this layer's step");
    const auto tm0 = std::chrono::steady_clock::now();
    if (moe_in_graph(s.decode, il)) { moe_late_upload(s, ml, which, tm0); return; }
    if (which.empty()) {
        ggml_backend_tensor_set(t_pg_, zeros_.data(), 0, (size_t) n_embd * T * sizeof(float));
        return;
    }
    if (gM_[il].gf || build_moe_gpu_graph(il)) {
        // Slot ids and weights for all n_expert_used positions; the
        // ones not in VRAM point at slot 0 with weight 0.
        std::vector<int32_t> & gids = sc_gids_; std::vector<float> & gw = sc_gw_;
        gids.assign(U, 0); gw.assign(U, 0.0f);
        for (int e : which) { gids[e] = s.eh[e].slot; gw[e] = wgt_[e]; }
        ggml_backend_tensor_set(t_gids_, gids.data(), 0, (size_t) U * sizeof(int32_t));
        ggml_backend_tensor_set(t_gw_,   gw.data(),   0, (size_t) U * sizeof(float));
        if (!ggml_gallocr_alloc_graph(gM_[il].ga, gM_[il].gf)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
        if (ggml_backend_graph_compute_async(w_.backend(), gM_[il].gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[qwfn] compute failed\n"); abort();
        }
    } else {
        const graph_ctx gc = moe_expert_graph(s, which, t_cur_, t_pg_);
        ggml_cgraph * g = gc.gf;
        ggml_context * c = gc.ctx;
        if (!ggml_gallocr_alloc_graph(galloc_moe_, g)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
        if (ggml_backend_graph_compute_async(w_.backend(), g) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[qwfn] compute failed\n"); abort();
        }
        moe_ctx_ = c;
    }
    moe_inflight_ = true;
    t_moe_gpu += secs_since(tm0);
    n_exp_gpu += which.size();
}

// Already done: graph A of this layer computed `which` into
// t_pg_. The experts promoted by this fetch were not there
// yet: their slots go up for the next graph's late fold,
// through pinned staging on the compute stream -- after the
// promotion copies, before the graph that reads them.
void engine::moe_late_upload(const step_ctx & s, const moe_layer & ml, const std::vector<int> & which,
                             std::chrono::steady_clock::time_point tm0) {
    const int64_t T = s.T;
    const std::vector<int> & late_gpu = ml.late_gpu;
    assert_that(moe_in_graph(s.decode, ml.il), "moe_late_upload: the in-graph MoE");
    assert_that((size_t) late_w_ * T <= (size_t) ggml_nelements(t_gids_), "moe_late_upload: the fold's table holds the step");
    if ((int) late_gpu.size() > n_late_) { fprintf(stderr, "[qwfn] %zu late experts, fold holds %d\n", late_gpu.size(), n_late_); abort(); }
    std::vector<int32_t> & gids = sc_gids_; std::vector<float> & gw = sc_gw_;
    gids.assign((size_t) late_w_ * T, 0); gw.assign((size_t) late_w_ * T, 0.0f);
    for (int64_t j = 0; j < T; j++)
        for (size_t k = 0; k < late_gpu.size(); k++) { gids[j * late_w_ + k] = s.eh[late_gpu[k]].slot; gw[j * late_w_ + k] = w_tok(late_gpu[k], j); }
    if (p_gids_) {
        memcpy(p_gids_->data, gids.data(), gids.size() * sizeof(int32_t));
        memcpy(p_gw_->data,   gw.data(),   gw.size() * sizeof(float));
        ggml_backend_tensor_set_async(w_.backend(), t_gids_, p_gids_->data, 0, gids.size() * sizeof(int32_t));
        ggml_backend_tensor_set_async(w_.backend(), t_gw_,   p_gw_->data,   0, gw.size() * sizeof(float));
    } else {
        ggml_backend_tensor_set(t_gids_, gids.data(), 0, gids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(t_gw_,   gw.data(),   0, gw.size() * sizeof(float));
    }
    n_exp_gpu += which.size() + late_gpu.size();
    t_moe_gpu += secs_since(tm0);
}

void engine::moe_gpu_settle() {
    assert_that(!moe_ctx_ || moe_inflight_, "moe_gpu_settle: a fallback graph only while in flight");
    assert_that(w_.backend() != nullptr, "moe_gpu_settle: the device backend");
    if (!moe_inflight_) return;
    const auto tm0 = std::chrono::steady_clock::now();
    ggml_backend_synchronize(w_.backend());
    if (moe_ctx_) ggml_free(moe_ctx_);
    moe_ctx_ = nullptr;
    moe_inflight_ = false;
    const double dt = secs_since(tm0);
    t_moe_gpu += dt; t_moe_gpu_sync += dt;
}

// Issue the predicted reads, so they overlap this layer's MoE
// compute rather than stalling the next fetch. One submission for
// the L+1 set and, when two-ahead is on, the L+2 set.
void engine::issue_prefetch(const step_ctx & s, moe_layer & ml) {
    const uint32_t il = ml.il;
    const int64_t U = hp_.n_expert_used, T = s.T;
    assert_that(il < hp_.n_layer, "issue_prefetch: layer in range");
    assert_that(T >= 1 && T <= 1 + MTP_MAX_DRAFTS, "issue_prefetch: a decode step");
    if (ml.prefetched || !cfg_.speculate || il + 1 >= hp_.n_layer) return;
    ml.prefetched = true;
    const uint32_t depth = std::min<uint32_t>(QWFN_SPEC_MAX, std::max<uint32_t>((uint32_t) U, cfg_.speculate_depth));
    const uint32_t K = QWFN_SPEC_MAX;
    pred_.resize(K * T); spec_scores_.resize(K * T); pred_margin_.resize(K * T);
    if (ml.packed) {
        std::copy(pred_next_.begin(), pred_next_.begin() + K * T, pred_.begin());
        std::copy(scores_next_.begin(), scores_next_.begin() + K * T, spec_scores_.begin());
    } else {
        ggml_backend_tensor_get(t_selnext_,   pred_.data(),        0, (size_t) K * T * sizeof(int32_t));
        ggml_backend_tensor_get(t_specscore_, spec_scores_.data(), 0, (size_t) K * T * sizeof(float));
    }
    // Margin to the routing cut-off (see engine_config::spec_margin), per position.
    const uint32_t Uu = (uint32_t) U;
    for (int64_t j = 0; j < T; j++) {
        const float * sc = spec_scores_.data() + j * K;
        for (uint32_t r = 0; r < K; r++)
            pred_margin_[j * K + r] = Uu < K ? (r < Uu ? sc[r] - sc[Uu] : sc[Uu - 1] - sc[r]) : 1e9f;
    }
    // pf_recent, not pf_outstanding: what is read decides what the RAM tier
    // holds, and so what can be promoted; it must not depend on I/O timing.
    const bool gate = cfg_.spec_margin > 0.0f && ec_.pf_recent() >= cfg_.spec_gate_inflight;
    std::vector<uint32_t> & pf1 = pf1_, & pf2 = pf2_;
    pf1.clear();
    for (int64_t j = 0; j < T; j++)
        for (uint32_t e = 0; e < depth; e++) {
            if (gate && pred_margin_[j * K + e] < cfg_.spec_margin) { pf_gated++; continue; }
            const uint32_t id = (uint32_t) pred_[j * K + e];
            if (std::find(pf1.begin(), pf1.end(), id) == pf1.end()) pf1.push_back(id);
        }

    pred2_a_.swap(pred2_b_);
    pred2_b_.clear();
    expert_cache::pf_set sets[2] = { { il + 1, pf1.data(), (uint32_t) pf1.size() }, {} };
    uint32_t n_sets = 1;
    if (cfg_.speculate_ahead >= 2 && il + 2 < hp_.n_layer) {
        pred2_b_.resize(depth);
        ggml_backend_tensor_get(t_selnext2_, pred2_b_.data(), 0, (size_t) depth * sizeof(int32_t));
        const uint32_t depth2 = std::min(depth, cfg_.speculate_depth2);
        pf2.assign(depth2, 0);
        for (uint32_t e = 0; e < depth2; e++) pf2[e] = (uint32_t) pred2_b_[e];
        if (depth2) sets[n_sets++] = { il + 2, pf2.data(), depth2 };
    }
    ec_.prefetch_begin(sets, n_sets);
}

void engine::part_rows_view(const step_ctx & s, const moe_layer & ml, const std::vector<int> & which, const tier_view & tv) {
    const int64_t n_embd = hp_.n_embd, T = s.T;
    assert_that(ml.pos_in_all.size() == (size_t) ml.n_u, "part_rows_view: selection positions per expert");
    assert_that(T >= 1 && (size_t) T <= (size_t) h_wd_->ne[2], "part_rows_view: a row block per position");
    if (which.empty()) return;
    const auto tm0 = std::chrono::steady_clock::now();
    const int n = (int) which.size();
    if (T >= 2) { part_rows_pairs(s, ml, which, tv, tm0); return; }
    const graph_ctx gc = new_tmp_ctx();
    ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
    ggml_tensor * ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, n, T);   ggml_set_input(ids);
    ggml_tensor * w   = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, n, T); ggml_set_input(w);
    ggml_tensor * wd  = moe_id_wd(c, tv, ids, w, h_cur_, n_embd, hp_.n_ff_exp, n, T);   // [n_embd, n, T]
    for (int k = 0; k < n; k++)
        ggml_build_forward_expand(g, ggml_cpy(c,
                ggml_view_2d(c, wd,    n_embd, T, wd->nb[2],    (size_t) k * wd->nb[1]),
                ggml_view_2d(c, h_wd_, n_embd, T, h_wd_->nb[2], (size_t) ml.pos_in_all[which[k]] * h_wd_->nb[1])));
    if (!ggml_gallocr_alloc_graph(galloc_cpu_, g)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
    std::vector<int32_t> & sid = sc_sid_; std::vector<float> & sw = sc_sw_;
    sid.assign((size_t) n * T, 0); sw.assign((size_t) n * T, 0.0f);
    for (int64_t j = 0; j < T; j++)
        for (int k = 0; k < n; k++) { sid[j * n + k] = s.eh[which[k]].slot; sw[j * n + k] = w_tok(which[k], j); }
    ggml_backend_tensor_set(ids, sid.data(), 0, sid.size() * sizeof(int32_t));
    ggml_backend_tensor_set(w,   sw.data(),  0, sw.size() * sizeof(float));
    if (ggml_backend_graph_compute(wh_.backend(), g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "[qwfn] compute failed\n"); abort(); }
    free_tmp_ctx(c);
    t_moe_cpu += secs_since(tm0);
    n_exp_cpu += which.size();
}

// Only the (expert, token) pairs the router actually chose. The
// T-column form computes every CPU expert for every token and
// multiplies the unchosen ones by a zero gate; the i-quant dot
// products are arithmetic-bound, so that doubled the CPU MoE of a
// verify step. Each pair's arithmetic is unchanged; the unchosen
// rows of h_wd_ are written as zeros, as the zero gate made them.
void engine::part_rows_pairs(const step_ctx & s, const moe_layer & ml, const std::vector<int> & which, const tier_view & tv,
                             std::chrono::steady_clock::time_point tm0) {
    const int64_t n_embd = hp_.n_embd, T = s.T;
    const int n = (int) which.size();
    assert_that(T >= 2 && n >= 1, "part_rows_pairs: a multi-position step with experts");
    assert_that(ml.pos_in_all.size() == (size_t) ml.n_u, "part_rows_pairs: selection positions per expert");
    std::vector<int32_t> & pk = sc_pk_, & pj = sc_pj_;
    pk.clear(); pj.clear();
    for (int64_t j = 0; j < T; j++)
        for (int k = 0; k < n; k++) {
            if (w_tok(which[k], j) != 0.0f) { pk.push_back(k); pj.push_back((int32_t) j); }
            else {
                float * row = (float *) ((char *) h_wd_->data + (size_t) j * h_wd_->nb[2] + (size_t) ml.pos_in_all[which[k]] * h_wd_->nb[1]);
                memset(row, 0, (size_t) n_embd * sizeof(float));
            }
        }
    const int P = (int) pk.size();
    if (P > 0) {
        const graph_ctx gc = new_tmp_ctx();
        ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
        ggml_tensor * tok_rows = ggml_new_tensor_1d(c, GGML_TYPE_I32, P);   ggml_set_input(tok_rows);
        ggml_tensor * ids  = ggml_new_tensor_2d(c, GGML_TYPE_I32, 1, P); ggml_set_input(ids);
        ggml_tensor * w    = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, 1, P); ggml_set_input(w);
        ggml_tensor * xin  = ggml_get_rows(c, s.v2(c, h_cur_), tok_rows);   // [n_embd, P]
        ggml_tensor * wd   = moe_id_wd(c, tv, ids, w, xin, n_embd, hp_.n_ff_exp, 1, P);   // [n_embd, 1, P]
        for (int q = 0; q < P; q++)
            ggml_build_forward_expand(g, ggml_cpy(c,
                    ggml_view_1d(c, wd, n_embd, (size_t) q * wd->nb[2]),
                    ggml_view_1d(c, h_wd_, n_embd, (size_t) pj[q] * h_wd_->nb[2] + (size_t) ml.pos_in_all[which[pk[q]]] * h_wd_->nb[1])));
        if (!ggml_gallocr_alloc_graph(galloc_cpu_, g)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
        std::vector<int32_t> & sid = sc_sid_; std::vector<float> & sw = sc_sw_;
        sid.assign(P, 0); sw.assign(P, 0.0f);
        for (int q = 0; q < P; q++) { sid[q] = s.eh[which[pk[q]]].slot; sw[q] = w_tok(which[pk[q]], pj[q]); }
        ggml_backend_tensor_set(tok_rows, pj.data(), 0, (size_t) P * sizeof(int32_t));
        ggml_backend_tensor_set(ids, sid.data(), 0, (size_t) P * sizeof(int32_t));
        ggml_backend_tensor_set(w,   sw.data(),  0, (size_t) P * sizeof(float));
        if (ggml_backend_graph_compute(wh_.backend(), g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "[qwfn] compute failed\n"); abort(); }
        free_tmp_ctx(c);
    }
    t_moe_cpu += secs_since(tm0);
    n_exp_cpu += which.size();
}

// Blocks from the cold file have their own types: one pass per file.
void engine::part_rows(const step_ctx & s, const moe_layer & ml, const std::vector<int> & which) {
    assert_that(ml.il < hp_.n_layer, "part_rows: layer in range");
    assert_that(ml.pos_in_all.size() == (size_t) ml.n_u, "part_rows: selection positions per expert");
    if (!cfg_.use_cold_tier) { part_rows_view(s, ml, which, ec_.ram_tier(ml.il)); return; }
    std::vector<int> & hot = sc_hot_, & cold = sc_cold_;
    hot.clear(); cold.clear();
    for (int k : which) (s.eh[k].from_cold ? cold : hot).push_back(k);
    part_rows_view(s, ml, hot, ec_.ram_tier(ml.il));
    part_rows_view(s, ml, cold, ec_.ram_tier_cold(ml.il));
}

void engine::reduce_rows(const step_ctx & s, moe_layer & ml) {
    const int64_t n_embd = hp_.n_embd, T = s.T;
    const int n = (int) ml.all_cpu.size();
    assert_that(ml.acc.size() == (size_t) n_embd * T, "reduce_rows: the sum covers the step");
    assert_that((size_t) n <= (size_t) h_wd_->ne[1], "reduce_rows: a row per CPU expert");
    if (n == 0) return;   // acc stays zero
    const auto tm0 = std::chrono::steady_clock::now();
    // The same left-to-right F32 sum a chain of ggml_add nodes computes, per
    // element, without a CPU graph (n-1 nodes, a barrier each) per layer.
    for (int64_t j = 0; j < T; j++) {
        float * out = ml.acc.data() + (size_t) j * n_embd;
        const char * base = (const char *) h_wd_->data + (size_t) j * h_wd_->nb[2];
        memcpy(out, base, (size_t) n_embd * sizeof(float));
        for (int k = 1; k < n; k++) {
            const float * r = (const float *) (base + (size_t) k * h_wd_->nb[1]);
            for (int64_t i = 0; i < n_embd; i++) out[i] += r[i];
        }
    }
    t_moe_cpu += secs_since(tm0);
}



// ---- short prompt: the union of its experts through the cache ----
// Distinct experts of the batch, ascending (ascending file offsets
// for the reads), fetched in chunks the RAM tier can hold at once.
// Per chunk: VRAM-resident experts compute from the tier, the rest
// are uploaded into the scratch and compute from there; both are one
// mul_mat_id per part with the batch's routing as ids, pairs outside
// the chunk pointed at a real slot with weight 0. The partials add
// up in t_pg_. Misses are admitted to the RAM tier as they are read,
// which is the warm-up generation needs anyway.
bool engine::cbatch_moe(step_ctx & s, uint32_t il, std::string & err) {
    assert_that(s.cbatch && il < hp_.n_layer, "cbatch_moe: a cache-served batch, layer in range");
    assert_that(scr_buf_ != nullptr && scr_slots_ > 0, "cbatch_moe: the device scratch");
    cbatch_layer & cl = *cl_;
    cl.reset(il);
    cbatch_setup(s, cl);
    if (cl.pipe && !cl.uniq.empty() && !cbatch_begin_chunk(cl, 0)) {
        err = "expert fetch failed"; return false;
    }
    for (size_t c0 = 0; c0 < cl.uniq.size(); c0 += cl.chunk)
        if (!cbatch_chunk(s, cl, c0, err)) return false;
    if (cl.pend_ctx) { ggml_backend_synchronize(w_.backend()); ggml_free(cl.pend_ctx); cl.pend_ctx = nullptr; }
    ec_.settle_promotions();
    ggml_backend_tensor_set(t_pc_, zeros_.data(), 0, (size_t) hp_.n_embd * s.T * sizeof(float));
    return true;
}

void engine::cbatch_setup(const step_ctx & s, cbatch_layer & cl) {
    const uint32_t il = cl.il;
    const int64_t U = hp_.n_expert_used, T = s.T;
    assert_that(il < hp_.n_layer && cl.uniq.empty(), "cbatch_setup: a fresh layer");
    assert_that((size_t) U * T <= sel_.size(), "cbatch_setup: the routing covers the batch");
    std::vector<uint32_t> & cnt = cl.cnt;
    cnt.assign(hp_.n_expert, 0);
    for (int64_t k = 0; k < U * T; k++) if (sel_[k] >= 0 && sel_[k] < (int32_t) hp_.n_expert) cnt[sel_[k]]++;
    for (uint32_t e = 0; e < hp_.n_expert; e++) if (cnt[e]) cl.uniq.push_back(e);

    // Misses land in the prefill's idle host staging, not in the RAM tier.
    cl.bounce = pf_.host_scratch(cl.bounce_bytes);
    cl.bounce_slots = cl.bounce && ec_.block_bytes(il) ? (uint32_t) std::min<size_t>(1u << 16, cl.bounce_bytes / ec_.block_bytes(il)) : 0;
    // Pipelined (default): chunk c+1's misses are read while chunk c's uploads
    // and expert matmuls run; the bounce is split in two halves, one per chunk
    // in flight. The chunks, their members and the order their partials add
    // into t_pg_ are those of the serial loop, which runs when the bounce
    // has fewer than two slots.
    cl.pipe = cl.bounce_slots >= 2;
    cl.chunk = std::max<uint32_t>(1, std::min<uint32_t>(
            std::min<uint32_t>(scr_slots_, cfg_.cache_batch_chunk),
            cl.bounce_slots ? (cl.pipe ? cl.bounce_slots / 2 : cl.bounce_slots) : std::max<uint32_t>(1, ec_.ram_tier(il).n_slots / 2)));
    cl.half_bytes = cl.pipe ? (cl.bounce_bytes / 2) & ~(size_t) 4095 : cl.bounce_bytes;
    ggml_backend_tensor_set(t_pg_, zeros_.data(), 0, (size_t) hp_.n_embd * T * sizeof(float));

    for (int q = 0; q < EXPERT_NPARTS; q++) cl.slice[q] = mi_->expert_range(il, 0, (expert_part) q).nbytes;
    cl.gt = ec_.gpu_tier(il);
    cl.rt = ec_.ram_tier(il);

    cl.chs[0].assign(cl.chunk, expert_handle{});
    cl.chs[1].assign(cl.chunk, expert_handle{});
    cl.uniq_pos.assign(hp_.n_expert, -1);
}

bool engine::cbatch_begin_chunk(cbatch_layer & cl, size_t cb) {
    assert_that(cl.pipe && cl.chunk >= 1, "cbatch_begin_chunk: the pipelined loop");
    assert_that(cb < cl.uniq.size(), "cbatch_begin_chunk: a chunk inside the batch's experts");
    const uint32_t nb = (uint32_t) std::min<size_t>(cl.chunk, cl.uniq.size() - cb);
    uint8_t * half = cl.bounce + ((cb / cl.chunk) % 2) * cl.half_bytes;
    return ec_.fetch_batch_begin(cl.il, cl.uniq.data() + cb, nb, cl.chs[(cb / cl.chunk) % 2].data(), half, cl.half_bytes);
}

bool engine::cbatch_chunk(step_ctx & s, cbatch_layer & cl, size_t c0, std::string & err) {
    assert_that(c0 < cl.uniq.size() && cl.chunk >= 1, "cbatch_chunk: a chunk inside the batch's experts");
    assert_that(s.T >= 2, "cbatch_chunk: a batch");
    const uint32_t n = (uint32_t) std::min<size_t>(cl.chunk, cl.uniq.size() - c0);
    std::vector<expert_handle> & ch = cl.chs[cl.pipe ? (c0 / cl.chunk) % 2 : 0];
    const auto ti = std::chrono::steady_clock::now();
    bool fetched;
    if (cl.pipe) {
        fetched = ec_.fetch_batch_end();
        // The previous chunk's uploads read the other bounce half and its
        // graph used the shared allocator: both done before either is reused.
        ggml_backend_synchronize(w_.backend());
        if (cl.pend_ctx) { ggml_free(cl.pend_ctx); cl.pend_ctx = nullptr; }
        if (fetched && c0 + cl.chunk < cl.uniq.size()) fetched = cbatch_begin_chunk(cl, c0 + cl.chunk);
    } else {
        fetched = cl.bounce_slots ? ec_.fetch_batch(cl.il, cl.uniq.data() + c0, n, ch.data(), cl.bounce, cl.bounce_bytes)
                                  : ec_.fetch(cl.il, cl.uniq.data() + c0, n, ch.data());
    }
    if (!fetched) {
        if (cl.pend_ctx) { ggml_backend_synchronize(w_.backend()); ggml_free(cl.pend_ctx); }
        err = "expert fetch failed"; return false;
    }
    { const double dti_cb = secs_since(ti); t_io += dti_cb; t_io_cbatch += dti_cb; }
    const auto tm0 = std::chrono::steady_clock::now();

    cl.mem_t.clear(); cl.mem_s.clear();
    const uint32_t m = cbatch_stage(cl, c0, n, ch, cl.mem_t, cl.mem_s);

    cbatch_build_term(s, cl, cl.tt, cl.mem_t);
    cbatch_build_term(s, cl, cl.ts, cl.mem_s);
    if (cl.tt.width == 0 && cl.ts.width == 0) return true;
    cbatch_run(s, cl, m, n, tm0);
    return true;
}

// The chunk's experts: VRAM-resident ones by tier slot, the rest uploaded into
// the scratch. Returns the scratch slots used.
uint32_t engine::cbatch_stage(cbatch_layer & cl, size_t c0, uint32_t n, const std::vector<expert_handle> & ch,
                              std::vector<int32_t> & mem_t, std::vector<int32_t> & mem_s) {
    assert_that(n >= 1 && ch.size() >= n, "cbatch_stage: a handle per chunk entry");
    assert_that(n <= scr_slots_, "cbatch_stage: the scratch holds the chunk");
    std::fill(cl.uniq_pos.begin(), cl.uniq_pos.end(), -1);
    cl.tt.slots.assign(n, -1); cl.ts.slots.assign(n, -1);
    uint32_t m = 0;   // scratch slots used, packed from 0
    for (uint32_t i = 0; i < n; i++) {
        cl.uniq_pos[cl.uniq[c0 + i]] = (int32_t) i;
        if (ch[i].on_gpu) { cl.tt.slots[i] = ch[i].slot; mem_t.push_back((int32_t) i); continue; }
        // RAM-resident: into the NEXT scratch slot, on the compute
        // stream. Packed, not at index i: MMQ over-reads the last
        // row of every expert into the following slot, and a slot
        // skipped here would hold an earlier layer's bytes, which
        // decoded as this layer's type can be NaN scales -- seen as
        // NaN logits on the second turn of a chat, once some of a
        // chunk's experts were already in VRAM.
        cl.ts.slots[i] = (int32_t) m; mem_s.push_back((int32_t) i);
        for (int q = 0; q < EXPERT_NPARTS; q++)
            ggml_backend_tensor_set_async(w_.backend(), scr_xfer_, ch[i].part[q],
                    scr_part_off_[q] + (size_t) m * cl.slice[q], cl.slice[q]);
        m++;
    }
    // The bytes past the last packed slot get read by the same
    // over-read; keep them zero (synchronous memset, disjoint).
    for (int q = 0; q < EXPERT_NPARTS; q++)
        ggml_backend_tensor_memset(scr_xfer_, 0, scr_part_off_[q] + (size_t) m * cl.slice[q], 8192);
    return m;
}

void engine::cbatch_build_term(const step_ctx & s, cbatch_layer & cl, cb_term & tm, std::vector<int32_t> & members /* chunk indices */) {
    const int64_t U = hp_.n_expert_used, T = s.T;
    assert_that(cl.uniq_pos.size() == hp_.n_expert, "cbatch_build_term: a chunk index per expert");
    assert_that(members.size() <= tm.slots.size(), "cbatch_build_term: members inside the chunk");
    tm.width = 0; tm.ids.clear(); tm.w.clear();
    if (members.empty()) return;
    // Chunk indices per token: at most U each, one per routed pair.
    cl.per_tok.assign((size_t) T * U, 0); cl.per_tok_n.assign((size_t) T, 0);
    for (int64_t t = 0; t < T; t++)
        for (int64_t j = 0; j < U; j++) {
            const int32_t i = cl.uniq_pos[sel_[t * U + j]];
            if (i >= 0 && tm.slots[i] >= 0) cl.per_tok[t * U + cl.per_tok_n[t]++] = i;
        }
    for (int64_t t = 0; t < T; t++) tm.width = std::max<int>(tm.width, cl.per_tok_n[t]);
    if (tm.width == 0) return;
    tm.ids.assign((size_t) tm.width * T, 0); tm.w.assign((size_t) tm.width * T, 0.0f);
    for (int64_t t = 0; t < T; t++) {
        const std::span<const int32_t> lst(cl.per_tok.data() + t * U, (size_t) cl.per_tok_n[t]);
        int j = 0;
        for (int32_t i : lst) {
            tm.ids[t * tm.width + j] = tm.slots[i];
            // the gate weight of this (token, expert) pair
            float wv = 0.0f;
            for (int64_t jj = 0; jj < U; jj++) if (cl.uniq_pos[sel_[t * U + jj]] == i) { wv = wgt_[t * U + jj]; break; }
            tm.w[t * tm.width + j] = wv;
            j++;
        }
        // pad with members this token does not use (distinct), weight 0
        for (size_t m = 0; m < members.size() && j < tm.width; m++) {
            const int32_t i = members[m];
            if (std::find(lst.begin(), lst.end(), i) != lst.end()) continue;
            tm.ids[t * tm.width + j] = tm.slots[i];
            j++;
        }
    }
}

void engine::cbatch_run(const step_ctx & s, cbatch_layer & cl, uint32_t m, uint32_t n,
                        std::chrono::steady_clock::time_point tm0) {
    const int64_t n_embd = hp_.n_embd, T = s.T;
    const cb_term & tt = cl.tt, & ts = cl.ts;
    assert_that(tt.width > 0 || ts.width > 0, "cbatch_run: a term to compute");
    assert_that(cl.pend_ctx == nullptr, "cbatch_run: the previous chunk's graph was settled");
    const graph_ctx gc = new_graph_ctx(16384);
    ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
    ggml_tensor * acc = s.v2(c, t_pg_);
    ggml_tensor * tid = nullptr, * tw = nullptr, * sid = nullptr, * sw = nullptr;
    if (tt.width > 0) {
        tid = ggml_new_tensor_2d(c, GGML_TYPE_I32, tt.width, T);   ggml_set_input(tid);
        tw  = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, tt.width, T); ggml_set_input(tw);
        acc = ggml_add(c, acc, moe_id_graph(c, cl.gt, tid, tw, t_cur_, n_embd, hp_.n_ff_exp, tt.width, T));
    }
    if (ts.width > 0) {
        tier_view sv;
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            sv.part[q]   = scr_base_ + scr_part_off_[q];
            sv.stride[q] = cl.slice[q];
            sv.type[q]   = cl.rt.type[q];
        }
        sv.n_slots = m;
        sv.buffer  = scr_buf_;
        sid = ggml_new_tensor_2d(c, GGML_TYPE_I32, ts.width, T);   ggml_set_input(sid);
        sw  = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, ts.width, T); ggml_set_input(sw);
        acc = ggml_add(c, acc, moe_id_graph(c, sv, sid, sw, t_cur_, n_embd, hp_.n_ff_exp, ts.width, T));
    }
    ggml_build_forward_expand(g, ggml_cpy(c, acc, s.v2(c, t_pg_)));
    if (!ggml_gallocr_alloc_graph(galloc_gpu_, g)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
    if (tid) { ggml_backend_tensor_set(tid, tt.ids.data(), 0, tt.ids.size() * 4); ggml_backend_tensor_set(tw, tt.w.data(), 0, tt.w.size() * 4); }
    if (sid) { ggml_backend_tensor_set(sid, ts.ids.data(), 0, ts.ids.size() * 4); ggml_backend_tensor_set(sw, ts.w.data(), 0, ts.w.size() * 4); }
    if (cl.pipe) {
        if (ggml_backend_graph_compute_async(w_.backend(), g) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[qwfn] compute failed\n"); abort();
        }
        cl.pend_ctx = c;
    } else {
        if (ggml_backend_graph_compute(w_.backend(), g) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[qwfn] compute failed\n"); abort();
        }
        ggml_free(c);
    }
    t_moe_gpu += secs_since(tm0);
    n_exp_gpu += n;
}

// ---- head --------------------------------------------------------------
void engine::step_head(const step_ctx & s) {
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count, T = s.T;
    assert_that(s.pending, "step_head: after the layers");
    assert_that(logits_.size() >= (size_t) n_vocab_ * (s.decode ? T : 1), "step_head: room for the step's logits");
    const auto th0 = std::chrono::steady_clock::now();
    scoped_add_time head_timer_{t_head, s.decode, th0};
    const graph_ctx gc = new_tmp_ctx();
    ggml_context * c = gc.ctx; ggml_cgraph * g = gc.gf;
    graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, s.n_past);
    ggml_tensor * r = vres(c, s, s.cur_res);
    if (s.pending) {
        ggml_tensor * pg = late_fold(s, c, hp_.n_layer - 1, s.v2(c, t_pg_));
        ggml_tensor * tot = t_rscale_
            ? ggml_add(c, s.v2(c, t_sh_), ggml_mul(c, ggml_add(c, pg, s.v2(c, t_pc_)), t_rscale_))
            : ggml_add(c, ggml_add(c, s.v2(c, t_sh_), pg), s.v2(c, t_pc_));
        r = gb.hc_combine(r, tot, s.v2(c, t_inject_));
    }
    if (mtp_on_)   // the wide residual of every position, for the draft head
        ggml_build_forward_expand(g, ggml_cpy(c, r, ggml_view_3d(c, t_hlast_, n_embd, hc, T, t_hlast_->nb[1], t_hlast_->nb[2], 0)));
    // Logits for the final position -- or for every position of a verify step.
    const int64_t n_out = s.decode ? T : 1;
    if (n_out == 1) r = ggml_view_3d(c, r, n_embd, hc, 1, r->nb[1], r->nb[2], (size_t) (T - 1) * r->nb[2]);
    ggml_tensor * o = gb.hc_mix(r, -1, false, false).out;
    ggml_tensor * logits = ggml_mul_mat(c, w_.get("output.weight"), o);
    ggml_set_output(logits);
    ggml_build_forward_expand(g, logits);
    run_on(g, true);
    ggml_backend_tensor_get(logits, logits_.data(), 0, (size_t) n_vocab_ * n_out * sizeof(float));
    free_tmp_ctx(c);
}

bool engine::step_mtp(const step_ctx & s, std::string & err) {
    const int64_t T = s.T, n_past = s.n_past;
    assert_that(mtp_on_, "step_mtp: the head is loaded");
    assert_that(T >= 1 && s.hist != nullptr, "step_mtp: the step's tokens");
    mtp_have_h_ = true; mtp_h_rows_ = T;
    if (T > 1 && !s.force_decode && mtp_kv_valid_) {
        // A batch is the head's prompt: its rows 0..T-2 pair with embeddings
        // 1..T-1 at positions n_past..n_past+T-2, and the drafts for positions
        // whose target is inside the prompt are scored. A later turn's batch
        // needs the row before it as well (mtp_gap_ below, run before the
        // trunk replaced the wide residual of the last decoded position).
        // In chunks: the head materialises [n_vocab, n] logits to score its drafts,
        // 1 MB per position, and a 500-token batch (prefill_decode_max) would ask for
        // half a gigabyte of device memory in one graph.
        constexpr int64_t HC = 64;
        for (int64_t off = 0; off < T - 1; off += HC) {
            const int64_t n = std::min<int64_t>(HC, T - 1 - off);
            const int64_t n_act = std::max<int64_t>(0, std::min<int64_t>(n, T - 2 - off));
            if (!mtp_draft(n_past + off, n, off, t_emb_, 1 + off, s.hist + (s.n_hist - T) + 2 + off, n_act, err)) return false;
        }
    }
    return true;
}

bool engine::mtp_draft(int64_t pos, int64_t n, int64_t h_row, ggml_tensor * e_src, int64_t e_row,
                       const int32_t * actual, int64_t n_actual, std::string & err, ggml_tensor * h_src) {
    assert_that(mtp_on_ && t_mtp_pos_ != nullptr, "mtp_draft: the head and its inputs");
    assert_that(pos >= 0 && n >= 1 && h_row >= 0 && e_row >= 0 && e_src != nullptr, "mtp_draft: positions and rows to draft from");
    const auto t0 = std::chrono::steady_clock::now();
    if (!h_src) h_src = t_hlast_;
    mtp_draft2_ = -1;
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count;
    set_positions(t_mtp_pos_, pos, n, pos_host_);
    ggml_init_params ip{}; ip.mem_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(1024, false); ip.no_alloc = true;
    // rule 3 deviation: the draft head's per-draft ggml context, see docs/CODING_RULES.md
    ggml_context * c = ggml_init(ip); assert_that(c != nullptr, "ggml_init: c");
    ggml_cgraph *  g = ggml_new_graph_custom(c, 1024, false);
    graph_builder gb(c, &hpm_, &wm_, &w_); gb.bind(&st_mtp_, g, pos);
    // The head's attention is dense over its own cache. One query sees every key
    // written so far, so it needs no mask at all; two positions (after an accepted
    // pair) need one -inf, at the second position's own key for the first row, on
    // a persistent buffer viewed as [n_kv, 2]. Building a [n_kv, n] mask plus the
    // unused QSA tables into a fresh device buffer per draft was most of a draft.
    ggml_tensor * mask = nullptr;
    ggml_context * mctx = nullptr; ggml_backend_buffer_t mbuf = nullptr;   // a prompt batch's mask, sized for it
    if (n >= 2) {
        mask = mtp_mask(c, pos, n, mctx, mbuf);
        if (!mask) { ggml_free(mctx); ggml_free(c); err = "no device memory for the head's prompt mask"; return false; }
    }
    ggml_tensor * h = ggml_view_3d(c, h_src, n_embd, hc, n, h_src->nb[1], h_src->nb[2], (size_t) h_row * h_src->nb[2]);
    ggml_tensor * e = ggml_view_2d(c, e_src, n_embd, n, e_src->nb[1], (size_t) e_row * e_src->nb[1]);
    ggml_tensor * pv = ggml_view_1d(c, t_mtp_pos_, 4 * n, 0);
    const int il = (int) hpm_.n_layer - 1;
    // First half on the GPU: up to the routing and the shared expert.
    mtp_first_half(gb, c, g, h, e, pv, mask, il, n);
    ggml_free(c);
    const auto tq1 = std::chrono::steady_clock::now();
    t_mtp_pre += std::chrono::duration<double>(tq1 - t0).count();
    mtp_cpu_moe(n, il);
    const auto tq2 = std::chrono::steady_clock::now();
    t_mtp_moe += std::chrono::duration<double>(tq2 - tq1).count();
    // Second half on the GPU: fold the partial and the shared expert, mix, project.
    ggml_init_params ip3{}; ip3.mem_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false); ip3.no_alloc = true;
    // rule 3 deviation: the draft head's per-draft ggml context, see docs/CODING_RULES.md
    c = ggml_init(ip3); assert_that(c != nullptr, "ggml_init: c");
    g = ggml_new_graph_custom(c, 256, false);
    ggml_tensor * logits = nullptr;
    ggml_tensor * am = mtp_second_half(c, g, pos, n, il, actual != nullptr, logits);   // the argmax of each position's logits, on the device
    t_mtp_post += secs_since(tq2);
    mtp_readback(pos, n, actual, n_actual, logits, am);
    if (mbuf) ggml_backend_buffer_free(mbuf);
    if (mctx) ggml_free(mctx);
    ggml_free(c);
    t_mtp += secs_since(t0);
    return true;
}

// The causal mask of an n-position draft (n >= 2); nullptr when a prompt batch's
// own mask buffer cannot be allocated (mctx is then set for the caller to free).
ggml_tensor * engine::mtp_mask(ggml_context * c, int64_t pos, int64_t n, ggml_context *& mctx, ggml_backend_buffer_t & mbuf) {
    assert_that(c != nullptr && n >= 2 && pos >= 0, "mtp_mask: a draft of two or more positions");
    assert_that(mctx == nullptr && mbuf == nullptr, "mtp_mask: no mask buffer yet");
    ggml_tensor * mask = nullptr;
    const int64_t n_kv = pos + n;
    std::vector<uint16_t> & m = mtp_mask_host_;
    m.assign((size_t) n_kv * n, 0);
    const uint16_t ninf = f16_of(-INFINITY);
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = pos + i + 1; j < n_kv; j++) m[(size_t) i * n_kv + j] = ninf;
    if ((size_t) n_kv * n <= (size_t) t_mtp_mask_->ne[0]) {
        ggml_backend_tensor_set(t_mtp_mask_, m.data(), 0, m.size() * 2);
        mask = ggml_view_2d(c, t_mtp_mask_, n_kv, n, (size_t) n_kv * 2, 0);
    } else {
        // A later turn's prompt batch: up to prefill_decode_max positions over
        // the whole context, once per turn.
        ggml_init_params mp{}; mp.mem_size = ggml_tensor_overhead() * 2; mp.no_alloc = true;
        // rule 3 deviation: the draft head's per-draft ggml context, see docs/CODING_RULES.md
        mctx = ggml_init(mp); assert_that(mctx != nullptr, "ggml_init: mctx");
        mask = ggml_new_tensor_2d(mctx, GGML_TYPE_F16, n_kv, n);
        mbuf = ggml_backend_alloc_ctx_tensors_from_buft(mctx, w_.buft());
        if (!mbuf) return nullptr;
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * 2);
    }
    return mask;
}

void engine::mtp_first_half(graph_builder & gb, ggml_context * c, ggml_cgraph * g, ggml_tensor * h, ggml_tensor * e,
                            ggml_tensor * pv, ggml_tensor * mask, int il, int64_t n) {
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count;
    assert_that(c && g && h && e && pv, "mtp_first_half: the head's inputs");
    assert_that(n >= 1 && (n == 1 || mask != nullptr), "mtp_first_half: a mask for two or more positions");
    int sections[4] = { hp_.mrope_sections[0], hp_.mrope_sections[1], hp_.mrope_sections[2], hp_.mrope_sections[3] };
    const mtp_pre_out o = gb.mtp_head_pre(h, e, pv, mask, sections, il);
    ggml_tensor * res = o.res, * cur = o.cur, * inj = o.inject, * sl = o.sel, * wt = o.w, * sh = o.sh;
    ggml_build_forward_expand(g, ggml_cpy(c, res, ggml_view_3d(c, t_m_res_, n_embd, hc, n, t_m_res_->nb[1], t_m_res_->nb[2], 0)));
    ggml_build_forward_expand(g, ggml_cpy(c, cur, view_rows(c, t_m_cur_, 0, n)));
    ggml_build_forward_expand(g, ggml_cpy(c, inj, view_rows(c, t_m_inject_, 0, n)));
    ggml_build_forward_expand(g, ggml_cpy(c, sl,  view_rows(c, t_m_sel_, 0, n)));
    ggml_build_forward_expand(g, ggml_cpy(c, wt,  view_rows(c, t_m_w_, 0, n)));
    ggml_build_forward_expand(g, ggml_cpy(c, sh,  view_rows(c, t_m_sh_, 0, n)));
    run_on(g, true);
}

// The routed experts on the CPU from host memory, the trunk's summation order.
void engine::mtp_cpu_moe(int64_t n, int il) {
    const int64_t n_embd = hp_.n_embd, U = hp_.n_expert_used;
    assert_that(n >= 1 && (size_t) n * n_embd <= xfer_.size(), "mtp_cpu_moe: the transfer buffer holds the draft");
    assert_that(il >= 0 && h_m_cur_ != nullptr, "mtp_cpu_moe: the head's host work set");
    std::vector<int32_t> & ids = mtp_ids_host_; std::vector<float> & ww = mtp_w_host_;
    ids.assign((size_t) U * n, 0); ww.assign((size_t) U * n, 0.0f);
    ggml_backend_tensor_get(t_m_sel_, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_get(t_m_w_,   ww.data(),  0, ww.size() * sizeof(float));
    ggml_backend_tensor_get(t_m_cur_, xfer_.data(), 0, (size_t) n * n_embd * sizeof(float));
    ggml_backend_tensor_set(h_m_cur_, xfer_.data(), 0, (size_t) n * n_embd * sizeof(float));
    ggml_backend_tensor_set(h_m_ids_, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(h_m_w_,   ww.data(),  0, ww.size() * sizeof(float));
    {
        ggml_init_params ip2{}; ip2.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false); ip2.no_alloc = true;
        // rule 3 deviation: the draft head's per-draft ggml context, see docs/CODING_RULES.md
        ggml_context * c2 = ggml_init(ip2); assert_that(c2 != nullptr, "ggml_init: c2");
        ggml_cgraph *  g2 = ggml_new_graph_custom(c2, 64, false);
        graph_builder gbh(c2, &hpm_, &wmh_);
        ggml_tensor * o = gbh.moe_resident(ggml_view_2d(c2, h_m_cur_, n_embd, n, h_m_cur_->nb[1], 0),
                                           ggml_view_2d(c2, h_m_ids_, U, n, h_m_ids_->nb[1], 0),
                                           ggml_view_3d(c2, h_m_w_, 1, U, n, h_m_w_->nb[1], h_m_w_->nb[2], 0), il, &wmh_);
        ggml_build_forward_expand(g2, ggml_cpy(c2, o, ggml_view_2d(c2, h_m_partial_, n_embd, n, h_m_partial_->nb[1], 0)));
        run_on(g2, false);
        ggml_free(c2);
    }
    ggml_backend_tensor_get(h_m_partial_, xfer_.data(), 0, (size_t) n * n_embd * sizeof(float));
    ggml_backend_tensor_set(t_m_pc_, xfer_.data(), 0, (size_t) n * n_embd * sizeof(float));
}

// Returns the argmax of each position's logits; `logits` is the graph's logits tensor.
ggml_tensor * engine::mtp_second_half(ggml_context * c, ggml_cgraph * g, int64_t pos, int64_t n, int il, bool scoring,
                                      ggml_tensor *& logits) {
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count;
    assert_that(c != nullptr && g != nullptr, "mtp_second_half: a graph context");
    assert_that(n >= 1 && pos >= 0, "mtp_second_half: positions to draft");
    graph_builder gb3(c, &hpm_, &wm_, &w_); gb3.bind(&st_mtp_, g, pos);
    ggml_tensor * res3 = ggml_view_3d(c, t_m_res_, n_embd, hc, n, t_m_res_->nb[1], t_m_res_->nb[2], 0);
    ggml_tensor * moe3 = ggml_add(c, view_rows(c, t_m_sh_, 0, n), view_rows(c, t_m_pc_, 0, n));
    const mtp_post_out po = gb3.mtp_head_post(res3, moe3, view_rows(c, t_m_inject_, 0, n), il);
    logits = po.logits;
    ggml_tensor * hres = po.hres;
    ggml_tensor * am = ggml_argmax(c, logits);
    ggml_set_output(am);
    ggml_build_forward_expand(g, am);
    // Kept for a second draft from the head's own residual (mtp_draft_next).
    if (!scoring) ggml_build_forward_expand(g, ggml_cpy(c, hres, ggml_view_3d(c, t_m_hres_, n_embd, hc, n, t_m_hres_->nb[1], t_m_hres_->nb[2], 0)));
    run_on(g, true);
    return am;
}

// n int32s back instead of n x 248K logits and a host scan -- unless the
// caller samples the draft itself, which needs the last position's logits.
void engine::mtp_readback(int64_t pos, int64_t n, const int32_t * actual, int64_t n_actual,
                          ggml_tensor * logits, ggml_tensor * am) {
    assert_that(logits != nullptr && am != nullptr, "mtp_readback: the head's outputs");
    assert_that(n >= 1 && n_actual <= n, "mtp_readback: scored positions within the draft");
    std::vector<int32_t> & top = mtp_top_;
    top.assign((size_t) n, 0);
    ggml_backend_tensor_get(am, top.data(), 0, top.size() * sizeof(int32_t));
    mtp_have_logits_ = false;
    if (mtp_want_logits_ && !actual) {
        const int64_t nv = logits->ne[0];
        mtp_logits_.resize((size_t) nv);
        ggml_backend_tensor_get(logits, mtp_logits_.data(), (size_t) (n - 1) * nv * sizeof(float), (size_t) nv * sizeof(float));
        mtp_have_logits_ = true;
    }
    for (int64_t j = 0; j < n; j++) {
        if (actual) {
            if (j < n_actual) { mtp_prompt_n++; if (actual[j] == top[j]) mtp_prompt_acc++; }
        } else {
            mtp_draft_ = top[j];
            mtp_draft_top_[0] = top[j]; mtp_draft_top_[1] = mtp_draft_top_[2] = -1;
        }
    }
    if (!actual) { mtp_hres_rows_ = n; mtp_last_pos_ = pos + n - 1; }
}

bool engine::mtp_draft_next(std::string & err, int32_t from_tok) {
    assert_that(mtp_n_drafts_ >= 0 && mtp_n_drafts_ <= MTP_MAX_DRAFTS, "mtp_draft_next: draft count in range");
    assert_that(!mtp_on_ || t_mtp_emb_ != nullptr, "mtp_draft_next: the head's embedding input");
    mtp_draft2_ = -1;
    if (!mtp_ready() || mtp_draft_ < 0 || mtp_hres_rows_ < 1 || !t_m_hres_) return true;
    const int64_t n_embd = hp_.n_embd;
    const int32_t d1 = mtp_draft_;
    const int32_t dlast = from_tok >= 0 ? from_tok : mtp_n_drafts_ >= 1 ? mtp_drafts_[mtp_n_drafts_ - 1] : d1;   // chain from the last draft, or the caller's
    // That draft's embedding, gathered on the host like mtp_step's.
    ggml_backend_tensor_set(h_mtp_tok_, &dlast, 0, sizeof(int32_t));
    {
        ggml_init_params ip{}; ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false); ip.no_alloc = true;
        // rule 3 deviation: the draft head's per-draft ggml context, see docs/CODING_RULES.md
        ggml_context * c = ggml_init(ip); assert_that(c != nullptr, "ggml_init: c");
        ggml_cgraph *  g = ggml_new_graph_custom(c, 8, false);
        ggml_tensor * te = ggml_get_rows(c, wh_.get("token_embd.weight"), ggml_view_1d(c, h_mtp_tok_, 1, 0));
        ggml_build_forward_expand(g, ggml_cpy(c, te, ggml_view_2d(c, h_mtp_emb_, n_embd, 1, h_mtp_emb_->nb[1], 0)));
        run_on(g, false);
        ggml_free(c);
    }
    ggml_backend_tensor_get(h_mtp_emb_, xfer_.data(), 0, (size_t) n_embd * sizeof(float));
    ggml_backend_tensor_set(t_mtp_emb_, xfer_.data(), 0, (size_t) n_embd * sizeof(float));
    // The head at the position after its last draft, its own residual for that
    // position as the hidden input. Its cache row there is speculative and is
    // rewritten by the next mtp_step before anything real attends to it.
    const int64_t pos2 = mtp_last_pos_ + 1, hrow = mtp_hres_rows_ - 1;
    const bool want = mtp_want_logits_;
    const bool kept = want && mtp_have_logits_ && !mtp_logits_.empty();
    if (kept) mtp_keep_ = mtp_logits_;   // the first draft's distribution, restored below (copy into reused storage)
    if (!mtp_draft(pos2, 1, hrow, t_mtp_emb_, 0, nullptr, 0, err, t_m_hres_)) return false;
    mtp_draft2_ = mtp_draft_;
    mtp_draft_  = d1;
    mtp_hres_rows_ = 1; mtp_last_pos_ = pos2;
    if (want) { mtp_logits2_ = mtp_logits_; mtp_have_logits2_ = mtp_have_logits_; if (kept) { mtp_logits_ = mtp_keep_; mtp_have_logits_ = true; } }
    if (from_tok >= 0 && mtp_n_drafts_ >= 1 && mtp_n_drafts_ < MTP_MAX_DRAFTS) {
        // The caller replaced the last draft by its sample; record the chained one after it.
        mtp_drafts_[mtp_n_drafts_ - 1] = from_tok;
        const int j = mtp_n_drafts_;
        mtp_drafts_[j] = mtp_draft2_; mtp_n_drafts_ = j + 1;
        mtp_have_logits_k_[j] = mtp_have_logits2_;
        if (mtp_have_logits2_) mtp_logits_k_[j] = mtp_logits2_;
    }
    return true;
}

void engine::qsa_decode_prepare_k(int k, int32_t n_past2) {
    assert_that(k >= 1 && k <= MTP_MAX_DRAFTS, "qsa_decode_prepare_k: a later position of the step");
    assert_that(qsa_ratio_ > 0 && n_past2 >= 1, "qsa_decode_prepare_k: the decode QSA state, a position after the first");
    qsa_decode_inputs & qd2_ = qdk_[k - 1];
    const qsa_decode_inputs & qprev = k == 1 ? qd_ : qdk_[k - 2];
    const int64_t r     = qsa_ratio_;
    const int64_t NBmax = qd_.bias->ne[0];
    const int32_t b_last = n_past2 / (int32_t) r;
    const int64_t n_bid  = (n_past2 + 1) / r;
    const int32_t wi = n_past2;
    ggml_backend_tensor_set(qd2_.write_idx, &wi, 0, 4);
    std::vector<int32_t> mi(r);
    for (int64_t m = 0; m < r; m++) mi[m] = (int32_t) (b_last * r + m);
    ggml_backend_tensor_set(qd2_.member_idx, mi.data(), 0, mi.size() * 4);
    int32_t bp[4] = { (int32_t) (b_last * r), (int32_t) (b_last * r), (int32_t) (b_last * r), (int32_t) (b_last * r) };
    ggml_backend_tensor_set(qd2_.blk_pos, bp, 0, sizeof bp);
    ggml_backend_tensor_set(qd2_.blk_idx, &b_last, 0, 4);
    const float nf = (float) n_past2;
    ggml_backend_tensor_set(qd2_.npast_f, &nf, 0, 4);
    // The previous position's bias, then this position's window on top of it.
    ggml_backend_tensor_copy(qprev.bias, qd2_.bias);
    float win[3]; int64_t b0 = std::max<int64_t>(0, b_last - 1), n = 0;
    for (int64_t b = b0; b <= b_last + 1 && b < NBmax; b++, n++)
        win[n] = b < n_bid ? 0.0f : (b == b_last ? 1e9f : -INFINITY);
    ggml_backend_tensor_set(qd2_.bias, win, (size_t) b0 * 4, (size_t) n * 4);
    // One bucket for both positions (the graph's shape); the later one bounds it.
    int64_t NB = ((b_last + 1 + 255) / 256) * 256;
    NB = std::max<int64_t>(NB, 768);
    NB = std::min<int64_t>(NB, NBmax);
    qd_.n_bucket = std::max(qd_.n_bucket, NB);
    qd_.k_blocks = std::min<int64_t>(qd_.k_blocks, qd_.n_bucket);
    for (int j = 0; j < MTP_MAX_DRAFTS; j++) { qdk_[j].n_bucket = qd_.n_bucket; qdk_[j].k_blocks = qd_.k_blocks; }
}

const float * engine::eval_decode(const int32_t * hist, int32_t n_hist, int32_t n_new, std::string & err) {
    assert_that(mi_ != nullptr, "eval_decode: engine initialised");
    assert_that(logits_.size() >= (size_t) n_vocab_ * (1 + MTP_MAX_DRAFTS), "eval_decode: room for every position's logits");
    if (n_new < 1 || n_new > 1 + MTP_MAX_DRAFTS) { err = "eval_decode: a token and at most " + std::to_string(MTP_MAX_DRAFTS) + " drafts"; return nullptr; }
    if (n_past_ + n_new > (int32_t) cfg_.n_ctx) { err = "context exhausted"; return nullptr; }
    if (n_new >= 2 && cfg_.skip_miss) { err = "eval_decode: a verified step and --skip-miss do not combine"; return nullptr; }
    ec_.set_max_promotions(cfg_.promote_per_layer * (uint32_t) n_new);   // a step of n positions looks up more experts per layer
    const bool ok = eval_batch(hist, n_hist, n_new, err, /*cache_batched=*/false, /*force_decode=*/true);
    ec_.set_max_promotions(cfg_.promote_per_layer);
    if (!ok) return nullptr;
    return logits_.data() + (size_t) (n_new - 1) * n_vocab_;
}

bool engine::rollback_n(int n_back, std::string & err) {
    if (n_back < 1 || n_back > rb_depth_) { err = "rollback: no snapshot of the state " + std::to_string(n_back) + " tokens back"; return false; }
    assert_that(rbbuf_ != nullptr && n_back <= rb_nsnap_, "rollback_n: a snapshot that many tokens back");
    assert_that(rb_rs_.size() == hp_.n_layer && rb_conv_.size() == hp_.n_layer, "rollback_n: per-layer snapshots");
    const auto t0 = std::chrono::steady_clock::now();
    // Slot n_back-1 of every snapshot, as views (same backend: a device copy).
    ggml_init_params vp{}; vp.mem_size = ggml_tensor_overhead() * (2 * hp_.n_layer + 4); vp.no_alloc = true;
    ggml_context * vc = ggml_init(vp); assert_that(vc != nullptr, "ggml_init: vc");
    const size_t s = (size_t) (n_back - 1);
    for (uint32_t il = 0; il < hp_.n_layer; il++) {
        if (!rb_rs_[il]) continue;
        ggml_tensor * rs = ggml_view_3d(vc, rb_rs_[il], rb_rs_[il]->ne[0], rb_rs_[il]->ne[1], rb_rs_[il]->ne[2], rb_rs_[il]->nb[1], rb_rs_[il]->nb[2], s * rb_rs_[il]->nb[3]);
        ggml_tensor * cv = ggml_view_2d(vc, rb_conv_[il], rb_conv_[il]->ne[0], rb_conv_[il]->ne[1], rb_conv_[il]->nb[1], s * rb_conv_[il]->nb[2]);
        ggml_backend_view_init(rs); ggml_backend_view_init(cv);   // a view made outside an allocator has no buffer of its own
        ggml_backend_tensor_copy(rs, st_.rs_state(il));
        ggml_backend_tensor_copy(cv, st_.rs_conv(il));
    }
    if (rb_ple_conv_) {
        ggml_tensor * pc = ggml_view_2d(vc, rb_ple_conv_, rb_ple_conv_->ne[0], rb_ple_conv_->ne[1], rb_ple_conv_->nb[1], s * rb_ple_conv_->nb[2]);
        ggml_backend_view_init(pc);
        ggml_backend_tensor_copy(pc, st_.ple_conv());
    }
    ggml_free(vc);
    // The KV, indexer and pooled-key rows the undone positions wrote are rewritten
    // by the next tokens at those positions; the bias window is recomputed per token.
    n_past_ -= n_back;
    rb_valid_ = false; rb_depth_ = 0;
    mtp_h_rows_ = std::max<int64_t>(1, mtp_h_rows_ - n_back);
    n_rollback++;
    t_rollback += secs_since(t0);
    return true;
}

bool engine::mtp_step(const int32_t * next_toks, int n, std::string & err) {
    assert_that(mtp_n_drafts_ >= 0 && mtp_n_drafts_ <= MTP_MAX_DRAFTS, "mtp_step: draft count in range");
    assert_that(!mtp_on_ || (h_mtp_tok_ != nullptr && t_mtp_emb_ != nullptr), "mtp_step: the head's embedding inputs");
    mtp_draft_ = -1; mtp_n_drafts_ = 0;
    if (!mtp_ready() || n < 1 || n > 1 + MTP_MAX_DRAFTS || mtp_h_rows_ < n || n_past_ < n) return true;
    const int64_t n_embd = hp_.n_embd;
    // The embeddings of the tokens that follow each position, gathered on the host.
    ggml_backend_tensor_set(h_mtp_tok_, next_toks, 0, (size_t) n * sizeof(int32_t));
    {
        ggml_init_params ip{}; ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false); ip.no_alloc = true;
        // rule 3 deviation: the draft head's per-draft ggml context, see docs/CODING_RULES.md
        ggml_context * c = ggml_init(ip); assert_that(c != nullptr, "ggml_init: c");
        ggml_cgraph *  g = ggml_new_graph_custom(c, 8, false);
        ggml_tensor * te = ggml_get_rows(c, wh_.get("token_embd.weight"), ggml_view_1d(c, h_mtp_tok_, n, 0));
        ggml_build_forward_expand(g, ggml_cpy(c, te, ggml_view_2d(c, h_mtp_emb_, n_embd, n, h_mtp_emb_->nb[1], 0)));
        run_on(g, false);
        ggml_free(c);
    }
    ggml_backend_tensor_get(h_mtp_emb_, xfer_.data(), 0, (size_t) n * n_embd * sizeof(float));
    ggml_backend_tensor_set(t_mtp_emb_, xfer_.data(), 0, (size_t) n * n_embd * sizeof(float));
    if (!mtp_draft(n_past_ - n, n, mtp_h_rows_ - n, t_mtp_emb_, 0, nullptr, 0, err)) return false;
    if (mtp_draft_ >= 0) {
        mtp_drafts_[0] = mtp_draft_; mtp_n_drafts_ = 1;
        mtp_have_logits_k_[0] = mtp_have_logits_;
        if (mtp_have_logits_) mtp_logits_k_[0] = mtp_logits_;
    }
    return true;
}

bool engine::mtp_draft_more(int k, std::string & err) {
    assert_that(mtp_n_drafts_ >= 0 && mtp_n_drafts_ <= MTP_MAX_DRAFTS, "mtp_draft_more: draft count in range");
    assert_that(mtp_n_drafts_ == 0 || mtp_on_, "mtp_draft_more: drafts only with the head");
    while (mtp_n_drafts_ >= 1 && mtp_n_drafts_ < std::min(k, MTP_MAX_DRAFTS)) {
        if (!mtp_draft_next(err)) return false;
        if (mtp_draft2_ < 0) break;
        const int j = mtp_n_drafts_;
        mtp_drafts_[j] = mtp_draft2_; mtp_n_drafts_ = j + 1;
        mtp_have_logits_k_[j] = mtp_have_logits2_;
        if (mtp_have_logits2_) mtp_logits_k_[j] = mtp_logits2_;
    }
    return true;
}

// Confidence-margin buckets for the prediction statistics: router logit
// differences, finer near zero where the routing cut-off lives.
static const float kMarginEdges[engine::SPEC_MARGIN_BUCKETS - 1] =
    { 0.05f, 0.1f, 0.2f, 0.3f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f };
int engine::margin_bucket(float m) {
    assert_that(SPEC_MARGIN_BUCKETS >= 2, "margin_bucket: at least two buckets");
    int b = 0;
    while (b < SPEC_MARGIN_BUCKETS - 1 && m >= kMarginEdges[b]) b++;
    assert_that(b >= 0 && b < SPEC_MARGIN_BUCKETS, "margin_bucket: bucket in range");
    return b;
}
float engine::margin_edge(int b) {
    assert_that(b < SPEC_MARGIN_BUCKETS, "margin_edge: bucket in range");
    assert_that(kMarginEdges[0] > 0.0f, "margin_edge: the edges start above zero");
    return b <= 0 ? 0.0f : kMarginEdges[b - 1];
}

} // namespace qwfn
