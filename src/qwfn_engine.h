#pragma once
// The engine: position-tracked evaluation over a growing sequence.
//
// Two paths, because the MoE behaves differently by batch size:
//
//   decode  (1 token)  every layer selects exactly n_expert_used experts, so the
//                      expert cache serves them and each runs one matmul in
//                      place from VRAM or RAM.
//   prefill (N tokens) each token picks its own experts, so the per-layer union
//                      approaches all 512. That is a streaming pass, not a
//                      caching one -- each expert is read once and reused across
//                      all its tokens -- so it reads from the mmap'd file and
//                      leaves the cache undisturbed for decode.
//
// State (KV, indexer keys, DeltaNet recurrence, conv history) persists across
// calls; n_past advances.

#include "qwfn_check.h"
#include "qwfn_expert_cache.h"
#include "qwfn_graph.h"
#include "qwfn_model.h"
#include "qwfn_prefill.h"
#include "qwfn_state.h"
#include "qwfn_weights.h"

#include <chrono>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace qwfn {

// Widest speculative prediction the engine will read back per layer.
static constexpr uint32_t QWFN_SPEC_MAX = 16;

struct engine_config {
    uint32_t  n_ctx      = 4096;
    uint32_t  n_batch    = 256;          // largest prefill ubatch
    size_t    ram_bytes  = 8ull  << 30;
    size_t    vram_bytes = 0;
    bool      use_gpu    = true;
    bool      use_qsa    = true;
    bool      use_cold_tier = false;
    int       n_threads  = 8;
    int       evict_policy = 0;   // 0 lru, 1 lfu, 2 hybrid
    uint32_t  promote_per_layer = 2;
    // Thread pool by default. On this filesystem io_uring_submit() executes the
    // reads inline instead of queueing them, so a burst never gets the
    // concurrency it asked for; blocking preads on worker threads do. Measured
    // +10% end to end on identical work. `--io-uring` selects the old path.
    // Stage prefill experts in VRAM and run the batched MoE on the GPU.
    bool      prefill_on_gpu = true;
    // Cap on n_kv * ubatch, bounding the per-call input arena. 0 disables.
    // 48M: at 128K context that is a 384-query attention chunk. The chunk's
    // graph holds several [n_kv, T] tensors; at 96M they needed a 1.3 GB
    // arena on top of the MoE chunk's and the tier had not left that free.
    uint64_t  ubatch_kv_product = 48ull << 20;
    // The cache-batched prompt path (prompts up to prefill_decode_max) runs inside
    // the decode reserve, not the lent prefill memory, and its attention work grows
    // with tokens x context; it is also slower than the sweep once the context is
    // long (495 tokens at 101K: 20 s batched in chunks against ~12 s streamed).
    // A prompt takes the batched path only while tokens x context stays under this
    // product: 12M is the old 96-token threshold at 131K, known to fit; at 10K a
    // 512-token prompt still batches (measured 2x faster than the sweep there).
    // Above it the prompt streams. Without the rule a 440-token tool result with an
    // image at a long context asked for 440 MB and aborted the server (2026-09-10).
    uint64_t  cbatch_kv_product = 12ull << 20;
    bool      reuse_graphs = true;   // replay the position-independent layer graphs
    bool      io_threads = true;
    unsigned  io_workers = 16;
    bool      speculate = true;     // prefetch the next layer's predicted experts
    // How many of the predicted top-k to actually fetch (clamped to
    // n_expert_used). Measured on a replayed fixed sequence: depth 0 -> 11.6
    // tok/s / 85.2% hit, depth 8 -> 12.1 / 92.8%, depth 10 -> 12.4 / 94.7%.
    // Prefetching every prediction wins despite ~23% of them being wrong,
    // because a wrong prefetch only costs bandwidth while a right one removes a
    // blocking read from the critical path.
    uint32_t  speculate_depth = 10;   // may exceed n_expert_used, up to QWFN_SPEC_MAX
    // How many layers ahead to predict. 1 = the L+1 lookahead above. 2 also
    // predicts L+2 from the same residual, so its reads get two layers of
    // compute to land in rather than one.
    //
    // Measured on a replayed sequence (identical routing): 12.46 tok/s at 1,
    // 12.48 at 2 with depth2=4, 12.4 at 2 with depth2=10. It is a wash. The
    // L+2 prediction is only 69% accurate against 79% for L+1, so the extra
    // reads mostly miss -- 3,600 wasted blocks against 2,300 -- and the second
    // router matmul adds ~0.6 s of graphA per 250 tokens, which cancels what
    // the wider window buys. Off by default; the machinery stays because it
    // becomes worthwhile if prediction accuracy improves.
    uint32_t  speculate_ahead = 1;
    // Depth for the L+2 set alone, when speculate_ahead >= 2. Fetching all ten
    // of its picks measured slower than four: the misses burn NVMe bandwidth
    // the demand fetches need and evict blocks that were still useful.
    uint32_t  speculate_depth2 = 4;
    // Confidence gate on the speculative reads. A predicted candidate's margin
    // is its distance in router logits from the routing cut-off: for a rank
    // inside the predicted top-n_expert_used, its logit minus the first
    // excluded one; for a rank past it, the last included logit minus its own.
    // A candidate whose margin is below spec_margin is not read. Changes only
    // which reads are issued, never what is computed. 0 = off.
    float     spec_margin = 0.0f;
    // Apply the gate only while the last two layers issued at least this many
    // speculative reads, i.e. when the disk is the constraint; with an idle disk
    // every prediction is worth reading (measured). 0 = always. (Counted as
    // issued, not as in flight: the reads decide the RAM tier's contents, which
    // must not depend on I/O timing.)
    uint32_t  spec_gate_inflight = 0;
    // Predict layer L+1's routing by running L+1's own token mixer (PLE
    // injection, attention mixer, DeltaNet or decode sparse attention, combine)
    // on the approximate residual inside layer L's graph, with every state
    // write suppressed, and applying the router to what that produces. The
    // block is the term the residual-only predictor cannot see. Costs one
    // extra block per layer of graph A, synchronous.
    bool      spec_block = false;
    // Which predicted layers get the block: "" = all, else "1,2,5-9,47".
    std::string spec_block_layers;
    // MTP draft head, experiment stage: the nextn GGUF (the checkpoint ships one
    // block, MTP/mtp-...gguf). Loaded whole on the GPU, run after every decoded
    // token on the trunk's last residual and the sampled token, and its draft
    // scored against the token that actually followed. Nothing is verified or
    // rolled back yet: this measures the acceptance rate the rest depends on.
    std::string mtp_path;
    // Keep the one-token-back snapshots a two-token decode step needs for
    // rollback(). Set with the head; the rollback test sets it alone.
    bool rollback_snapshots = false;
    // Decode without waiting for expert misses: the token is computed from the
    // experts that are resident (gates renormalised), the misses' reads still
    // go out and land for next time. An approximation -- measure its NLL.
    bool      skip_miss = false;
    // Drafts the head proposes per step (1..QWFN_MTP_MAX_DRAFTS): the trunk verifies
    // them as one step of 1+k positions and rolls back to the last accepted one.
    uint32_t  mtp_drafts = 1;
    // Decode only: drop a routed expert whose normalised gate is below this and
    // renormalise the rest in the router graph (0 = off). An approximation,
    // priced with the replay NLL; the dropped experts are neither fetched nor computed.
    float     gate_drop = 0.0f;
    // Attention-state placement. The indexer key cache and the KV cache can live
    // in pinned host memory instead of VRAM: the device kernels read the rows a
    // token needs over PCIe (measured on the doc replay at 131K: +0.35 ms/token
    // for the indexer, +1.7 ms/token for the KV cache's 2,051-cell gathers) and
    // the VRAM they held goes to the expert tier, ~4% of decode per GB at 131K
    // and the difference between a tier and none at 256K. --state-host idx|kv,idx.
    bool      idx_host = false;
    bool      kv_host  = false;
    // Overlap the next layer's bulk expert read with this layer's upload and
    // compute during prefill. Costs a second host staging buffer (~1.8 GB),
    // taken before the RAM tier sizes itself so the memory guard sees it.
    bool      prefill_overlap = true;
    // Below this many new tokens, prefill runs as repeated single-token decodes
    // rather than the streaming batched path.
    //
    // The streaming path reads every expert of every layer -- ~53 GB -- no
    // matter how many tokens are in the ubatch, because at a useful ubatch
    // essentially all 512 are selected anyway. That is the right trade at 1024
    // tokens (~0.013 s/token) and a bad one at 25 (~0.45 s/token). The decode
    // path instead fetches only what routing actually asks for, and warms the
    // expert cache for the generation that follows instead of bypassing it.
    // This is the common case for a chat server: with prefix reuse, every turn
    // after the first feeds ~25 tokens.
    // Prompts up to this many new tokens take the cache-batched decode path (one batch,
    // its experts through the tiers, misses into a bounce without admission) instead of
    // the streamed sweep of every expert. Measured 2026-09-10 on Q4 at 131K: the sweep is
    // a flat ~12 s, the batch path 7-10 s up to ~400 tokens and level with it at 500+; the
    // larger decode buffers cost ~150 MB of VRAM.
    uint32_t  prefill_decode_max = 512;
    // MemAvailable clamp on the expert RAM tier: frac of MemAvailable minus
    // headroom, a backstop for the command line. 0.75: on a 32 GB desktop with
    // zram (25 GB available) a 16 GB arena left 1.5 GB free at the worst point
    // of a 131K run and 19 GB ran it out; 0.75 allows ~15 GB there. 0.60 built
    // a 10.5 GB tier whatever --ram asked and hid the tier's elasticity, about
    // +3% of decode per GB at 131K (2026-09-10). The console sizes the tier
    // from the process's measured needs and passes its own frac.
    double    ram_frac     = 0.75;
    size_t    ram_headroom = 3ull << 30;
    // Device memory the VRAM expert tier must leave behind for everything
    // allocated after it: the prefill MoE graph arena (~0.31 MB per token of
    // ubatch), the per-call input arena (kq_mask and the QSA bias, bounded by
    // ubatch_kv_product), and the 36 lazily-created replay allocators. 0 sizes
    // it from n_batch, which is what makes a large ubatch safe -- a fixed
    // reserve failed at ubatch 2048 with "prefill galloc failed".
    size_t    vram_reserve = 0;
    ggml_type type_k     = GGML_TYPE_F16;
    ggml_type type_v     = GGML_TYPE_F16;
    // After each streamed prefill layer, copy this many of the experts the
    // prompt routed to (most recent tokens first) into the RAM tier, and the
    // first prefill_warm_vram of them on to VRAM, so generation does not start
    // on a cold cache. 0 disables.
    uint32_t  prefill_warm      = 48;
    uint32_t  prefill_warm_vram = 32;
    // Prompts of 2..prefill_decode_max tokens: run the dense core once for the
    // whole prompt and serve the union of its experts through the cache in
    // chunks, uploading each chunk's RAM-resident experts into a VRAM scratch
    // so the batched MoE runs on the GPU. Token by token was the alternative,
    // at ~50 ms per token whatever the prompt. Needs a GPU.
    bool      cache_batched     = true;
    uint32_t  cache_batch_chunk = 48;   // experts per cache fetch, at most 64
    // Layer-major prefill: each layer's experts are streamed ONCE per eval
    // batch of up to n_batch tokens, and the compute runs in chunks of this
    // many tokens (the graph arena bound); the attention layers chunk further
    // to ubatch_kv_product. Before, the sweep was per ubatch and the ubatch
    // shrank with the context: 131K would have taken ~500 sweeps.
    uint32_t  prefill_chunk     = 2048;
    // Override the GGUF's indexer top-k (2048 cells). Past ~150K of context
    // this quant loses its grounding at the stock budget (llama.cpp's
    // recommendation there is 4096); 0 keeps the checkpoint's value.
    uint32_t  indexer_top_k     = 0;
};

class engine {
public:
    ~engine();
    engine();
    engine(const engine &) = delete;
    engine & operator=(const engine &) = delete;

    bool init(const model_index * hot, const model_index * cold,
              const engine_config & cfg, const std::string & backend_dir, std::string & err);

    // `hist` is the whole sequence including the n_new tokens being appended --
    // the PLE n-gram window needs the predecessors, so the engine cannot work
    // from the new tokens alone. Returns logits for the final position.
    const float * eval(const int32_t * hist, int32_t n_hist, int32_t n_new, std::string & err);

    // Decode one or two tokens through the decode path (the MTP verify step feeds
    // the sampled token and the draft as a pair): returns the last position's
    // logits; logits_pos(i) gives each position's. After a pair, rollback()
    // restores the state as it was after the first token and steps n_past back
    // by one -- the caches the second position wrote are rewritten by the next
    // token at that position.
    const float * eval_decode(const int32_t * hist, int32_t n_hist, int32_t n_new, std::string & err);
    const float * logits_pos(int i) const { return logits_.data() + (size_t) i * n_vocab_; }
    bool rollback(std::string & err) { return rollback_n(1, err); }
    // Undo the last n_back positions of the last multi-token step (1..T-1).
    bool rollback_n(int n_back, std::string & err);
    static constexpr int MTP_MAX_DRAFTS = 3;
    // Layer 0's routing for the T tokens that end `hist` (the positions the next
    // eval will take), computed exactly on the state as it stands, and their
    // expert reads issued now. Every other layer is predicted a layer ahead; layer 0
    // had nothing before it, so each token boundary paid a demand fetch (3-4.5 ms
    // measured). Cheap and exact: layer 0's block on the current state IS what the
    // next eval computes, only the state writes are suppressed.
    bool spec_layer0(const int32_t * hist, int32_t n_hist, int32_t T, std::string & err);
    double t_spec_l0 = 0; uint64_t n_spec_l0 = 0;
    double t_mtp_pre = 0, t_mtp_moe = 0, t_mtp_post = 0;   // where a draft's time goes: dense half, CPU experts (with the transfers), second half + LM head
    // MTP head: run it over the last n positions of the sequence (their wide
    // residuals from the last eval) with the tokens that follow each, and keep
    // the last position's draft. mtp_ready() says whether the head can draft.
    bool    mtp_step(const int32_t * next_toks, int n, std::string & err);
    int32_t mtp_draft_id() const { return mtp_draft_; }
    // A second draft, the token after the first: the head run once more from its own
    // residual and the first draft's embedding (the standard chaining of an MTP
    // module). mtp_draft2_id() holds it; the first draft stays in mtp_draft_id().
    bool    mtp_draft_next(std::string & err, int32_t from_tok = -1);   // from_tok: chain from this token instead of the last draft (a sampled draft)
    int32_t mtp_draft2_id() const { return mtp_draft2_; }
    // The drafts of the current step, in order (mtp_draft_id() is the first), and
    // each one's distribution when logits are wanted.
    int           mtp_draft_count() const { return mtp_n_drafts_; }
    int32_t       mtp_draft_k(int k) const {
        assert_that(k >= 0, "mtp_draft_k: a draft index");
        assert_that(mtp_n_drafts_ >= 0 && mtp_n_drafts_ <= MTP_MAX_DRAFTS, "mtp_draft_k: draft count in range");
        return k < mtp_n_drafts_ ? mtp_drafts_[k] : -1;
    }
    const float * mtp_logits_k(int k) const {
        assert_that(k >= 0, "mtp_logits_k: a draft index");
        assert_that(mtp_n_drafts_ >= 0 && mtp_n_drafts_ <= MTP_MAX_DRAFTS, "mtp_logits_k: draft count in range");
        return k < mtp_n_drafts_ && mtp_have_logits_k_[k] ? mtp_logits_k_[k].data() : nullptr;
    }
    // Extend the current step's drafts to k (each from the previous draft's residual).
    bool          mtp_draft_more(int k, std::string & err);
    bool    mtp_on() const { return mtp_on_; }
    // The head's logits at the draft position, read back only when asked for
    // (the server samples the draft from them at temperature); nullptr otherwise.
    void          set_mtp_logits(bool on) { mtp_want_logits_ = on; }
    const float * mtp_logits() const {
        assert_that(!mtp_have_logits_ || mtp_on_, "mtp_logits: only the head produces them");
        assert_that(!mtp_have_logits_ || !mtp_logits_.empty(), "mtp_logits: held when flagged");
        return mtp_have_logits_ ? mtp_logits_.data() : nullptr;
    }
    const float * mtp_logits2() const {
        assert_that(!mtp_have_logits2_ || mtp_on_, "mtp_logits2: only the head produces them");
        assert_that(!mtp_have_logits2_ || !mtp_logits2_.empty(), "mtp_logits2: held when flagged");
        return mtp_have_logits2_ ? mtp_logits2_.data() : nullptr;
    }
    bool    mtp_ready() const { return mtp_on_ && mtp_have_h_ && mtp_kv_valid_; }
    bool    mtp_loaded() const { return mtp_on_; }     // the head is resident (not with --skip-miss)

    // Session snapshots: the sequence state of the first n_past positions (KV,
    // indexer and conv/recurrent state, trunk and draft head) copied to host
    // memory, so a request that does not extend the current sequence (an agent's
    // side request, a second conversation) need not cost the next one a full
    // prefill of a prefix it already paid for.
    struct state_snapshot {
        int32_t n_past = 0;
        std::vector<std::vector<uint8_t>> blobs;
        // The draft head's carry: the wide residual of the last position (one row
        // of t_hlast_) and whether its KV is complete; without them drafting stays
        // off for the rest of a restored conversation.
        std::vector<uint8_t> hlast;
        bool mtp_have_h = false, mtp_kv_valid = true;
        size_t bytes() const {
            assert_that(n_past >= 0, "snapshot: a position count");
            assert_that(mtp_have_h || hlast.empty(), "snapshot: a residual row only with the head's carry");
            size_t b = 0; for (const auto & x : blobs) b += x.size(); return b;
        }
    };
    bool snapshot_save(state_snapshot & s);
    bool snapshot_load(const state_snapshot & s);
    void    reset();                       // clear state, rewind to position 0
    // Lend the expert tier's dynamic VRAM buffer to a client -- the server,
    // before it stages the vision projector for an image -- and take it back.
    // A streamed prefill lends it anyway and returns it; eval() re-syncs the
    // replayed graphs whenever the tier moved, so a lend is safe from anywhere.
    void    vram_lend_begin();
    void    vram_lend_end();

    // Substitute externally computed embeddings (vision) for the tokens at
    // absolute positions [pos, pos + n). The engine still takes token ids --
    // the placeholder <|image_pad|> ids keep the sequence and the PLE n-gram
    // window well formed -- and only their EMBEDDINGS are replaced, which is
    // exactly what the reference does.
    void set_embeddings(int32_t pos, const float * emb, int32_t n);
    void clear_embeddings() { ov_pos_.clear(); ov_.clear(); }
    int32_t n_past() const { return n_past_; }
    int64_t n_vocab() const { return n_vocab_; }

    // The vision tower shares the language model's backend and buffer type.
    ggml_backend_t             backend() const { return w_.backend(); }
    ggml_backend_buffer_type_t buft()    const { return w_.buft(); }
    bool is_attn_layer(uint32_t il) const { return hp_.is_attn_layer(il); }

    const expert_cache_stats & cache_stats() const { return ec_.stats(); }
    // CPU threads for the experts served from RAM (ggml's CPU backend), changeable
    // between evals: the console's auto-tune sweeps it on a running server.
    void set_n_threads(int n);
    int  n_threads() const { return n_threads_; }
    // The streamed prefill batch in flight: layers done of n_layer, T tokens in the batch;
    // all zero outside an eval. Updated after each layer, readable from any thread, so a
    // server can report the prefill's progress inside a batch.
    struct prefill_progress_t { uint32_t layers_done = 0, n_layer = 0; int32_t T = 0; };
    prefill_progress_t prefill_progress() const {
        const uint64_t v = prefill_progress_.load(std::memory_order_relaxed);
        return { (uint32_t) (v >> 48), (uint32_t) ((v >> 32) & 0xffff), (int32_t) (uint32_t) v };
    }
    double t_replay_alloc = 0, t_replay_launch = 0, t_replay_wait = 0; uint64_t n_replay = 0;   // the cached decode graphs' replays, split
    uint64_t prefill_bytes_read()     const { return pf_.bytes_read; }       // expert bytes the streamed sweeps read
    uint64_t prefill_bytes_from_ram() const { return pf_.bytes_from_ram; }   // ...and took from the RAM tier instead
    double   prefill_reader_busy() const { return pf_.t_reader; }
    expert_cache::census        ram_census()  const { return ec_.ram_census(); }
    const io_engine &          cache_io()    const { return ec_.io(); }
    std::string memory_summary() const;

    double t_prefill = 0, t_decode = 0, t_io = 0, t_warm = 0;
    double t_pf_graphA = 0, t_pf_moe = 0, t_pf_read = 0;   // where a layer-major prefill's time goes
    double t_moe_gpu = 0, t_moe_cpu = 0, t_layerA = 0;   // where decode time goes
    double t_pre = 0, t_head = 0, t_eval_decode = 0;     // decode steps: inputs/embedding, head + logits, whole eval
    double t_io_cbatch = 0;
    std::vector<uint8_t> tmp_ctx_buf_;     // backing store of the per-step temporary graph contexts
    bool tmp_ctx_busy_ = false;                              // the part of t_io spent in short-prompt (cbatch) prefills
    // The GPU MoE runs asynchronously, overlapped with the expert I/O wait and
    // the CPU MoE. t_moe_gpu then counts launch cost plus whatever the final
    // sync still had to wait -- t_moe_gpu_sync is that wait alone, and ~0 means
    // the overlap hid the GPU MoE completely.
    double t_moe_gpu_sync = 0;
    double t_layerA_rec = 0, t_layerA_attn = 0;   // replayed vs rebuilt (decode only)
    uint64_t n_layerA_rec = 0, n_layerA_attn = 0;
    // Decode attention layers, split: host-side graph build + allocation,
    // and the synchronous GPU compute. Per-call input construction (mask,
    // QSA inputs, the device allocation for them) is t_inputs, once per token.
    double t_attn_build = 0, t_attn_compute = 0, t_inputs = 0;
    uint64_t n_exp_gpu = 0, n_exp_cpu = 0, n_exp_skipped = 0, n_exp_dropped = 0;
    // Per-layer I/O wait profile of the decode loop (seconds summed over the run): the
    // wait when a layer's fetch is issued (deferred misses, demand reads that had to
    // block) and the wait at its end (reads still landing after the ready pass), and
    // the reads that fetch issued itself (not the speculative ones for the next layer).
    std::vector<double>   prof_io_begin, prof_io_end;
    std::vector<uint64_t> prof_reads;
    uint64_t pred_hits = 0, pred_total = 0;   // predicted-vs-actual expert overlap
    uint64_t pred2_hits = 0, pred2_total = 0; // same, for the two-ahead prediction
    // Prediction quality by rank and by confidence margin -- the gate's
    // calibration data, collected on every decode run -- and per predicted layer.
    static constexpr int SPEC_MARGIN_BUCKETS = 12;
    uint64_t rank_hits[QWFN_SPEC_MAX] = {}, rank_total[QWFN_SPEC_MAX] = {};
    uint64_t margin_hits[SPEC_MARGIN_BUCKETS] = {}, margin_total[SPEC_MARGIN_BUCKETS] = {};
    std::vector<uint64_t> pred_hits_layer, pred_total_layer;
    uint64_t pf_gated = 0;   // predicted candidates the margin gate declined to read
    // MTP experiment: drafts scored against the next token (decode), and against
    // the prompt's own tokens when the prompt went through in one batch.
    uint64_t mtp_n = 0, mtp_acc = 0, mtp_top3 = 0, mtp_prompt_n = 0, mtp_prompt_acc = 0;
    double   t_mtp = 0, t_rollback = 0;
    uint64_t n_rollback = 0;
    static int   margin_bucket(float m);
    static float margin_edge(int b);   // lower edge of bucket b
    int64_t n_prefill = 0, n_decode = 0;

private:
    bool eval_batch(const int32_t * hist, int32_t n_hist, int32_t n_new, std::string & err,
                    bool cache_batched = false, bool force_decode = false);
    bool eval_prefill_big(const int32_t * hist, int32_t n_hist, int32_t T, std::string & err);
    // Per-chunk attention inputs for the mask-based path: the causal mask and
    // the QSA block tables for Tc queries starting at n_past_c.
    struct attn_inputs {
        ggml_context *        ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor *         kq_mask = nullptr;
        qsa_inputs            qsa;
        uint32_t              ratio = 0;
        void release() {
            assert_that(!buf || ctx, "attn_inputs: a buffer only with its context");
            assert_that(ratio == 0 || qsa.ratio == ratio, "attn_inputs: block tables built for the ratio");
            if (buf) ggml_backend_buffer_free(buf);
            if (ctx) ggml_free(ctx);
            buf = nullptr; ctx = nullptr;
        }
    };
    bool build_attn_inputs(int64_t n_past_c, int64_t Tc, attn_inputs & ai, std::string & err);
    int32_t max_ubatch(int32_t n_past) const;
    void build_qsa_inputs(int64_t n_kv, int64_t T, uint32_t ratio);
    void run_on(ggml_cgraph * gf, bool gpu);

    // ---- decomposition of init and the eval paths (qwfn_engine.cpp) ----------
    struct graph_ctx { ggml_context * ctx = nullptr; ggml_cgraph * gf = nullptr; };
    struct step_ctx;       // eval_batch's call-wide state
    struct layer_a;        // one layer's graph A tensors
    struct moe_layer;      // decode MoE state of one layer
    struct cb_term;        // an id/weight matrix of the cache-served batch
    struct cbatch_layer;   // cache-served batch state of one layer
    struct prefill_ctx;    // eval_prefill_big's call-wide state
    static graph_ctx new_graph_ctx(size_t n);
    graph_ctx new_tmp_ctx();
    void      free_tmp_ctx(ggml_context * c);
    int64_t   decode_rows() const;
    // init phases
    bool init_mtp(const std::string & backend_dir, std::string & err);
    bool init_host_maps(std::string & err);
    bool init_state(std::string & err);
    bool init_work_set(std::string & err);
    void init_work_tensors();
    void init_gammas();
    void fold_inject_scale();
    void init_pack_and_spec();
    void parse_spec_block_layers();
    void init_route_tensors();
    bool init_mtp_work(std::string & err);
    bool init_host_work(std::string & err);
    void init_pinned_staging();
    void init_qsa_decode();
    void init_qsa_tensors(int64_t NBmax);
    void init_qsa_tables(int64_t NBmax);
    void init_cbatch_scratch();
    expert_cache::config expert_cache_config() const;
    void set_lend_budget(expert_cache::config & ec_cfg, const model_index * hot) const;
    bool init_expert_cache(const model_index * hot, const model_index * cold, std::string & err);
    bool init_graphs(const model_index * hot, std::string & err);
    // attention inputs, embeddings
    void qsa_pool_rebuild(int32_t n_past, int32_t b_last, int64_t n_bid);
    void fill_chunk_qsa(attn_inputs & ai, int64_t n_past_c, int64_t Tc);
    void spec_l0_inputs(const int32_t * hist, int32_t n_hist, int64_t T, bool is_ple);
    void spec_l0_route(int64_t T, bool is_ple, std::vector<int32_t> & ids);
    void upload_tok_ple(const int32_t * hist, int32_t n_hist, int64_t T);
    void gather_ple_emb(const int32_t * hist, int32_t n_hist, int64_t T, bool tmp);
    bool has_overrides(int64_t n_past, int64_t T) const;
    void upload_emb_residual(int64_t n_past, int64_t T, bool tmp);
    bool mtp_head_gap(const int32_t * hist, int32_t n_hist, int64_t T, std::string & err);
    // streamed prefill
    bool prefill_layer(prefill_ctx & p, uint32_t il, std::string & err);
    bool prefill_chunk_a(prefill_ctx & p, uint32_t il, int64_t off, int64_t Tc, bool is_ple, std::string & err);
    bool prefill_layer_moe(prefill_ctx & p, uint32_t il, std::string & err);
    bool prefill_chunk_moe(int64_t off, int64_t Tc, const tier_view & staged, bool on_gpu, std::string & err);
    void prefill_warm_layer(const prefill_ctx & p, uint32_t il);
    void prefill_mtp_kv(prefill_ctx & p);
    void prefill_head(const prefill_ctx & p);
    // decode / cache-served batch step
    ggml_tensor * vres(ggml_context * c, const step_ctx & s, int which) const;
    ggml_tensor * vpos(ggml_context * c, const step_ctx & s) const;
    bool moe_in_graph(bool decode, uint32_t il) const;
    void upload_vtable(bool decode, uint32_t il);
    ggml_tensor * late_fold(const step_ctx & s, ggml_context * c, uint32_t prev, ggml_tensor * pg);
    void upload_step_inputs(step_ctx & s);
    bool build_step_inputs(step_ctx & s, std::string & err);
    void fill_step_qsa(step_ctx & s, int64_t n_blocks);
    bool eval_layer(step_ctx & s, uint32_t il, std::string & err);
    bool replay_layer_a(uint32_t il, std::string & err);
    bool build_layer_a(step_ctx & s, uint32_t il, bool is_ple, bool replayable, bool pack_ok, std::string & err);
    ggml_tensor * attn_decode(const step_ctx & s, graph_builder & gbx, ggml_context * c, ggml_tensor * x, uint32_t l);
    void layer_a_core(const step_ctx & s, graph_builder & gb, layer_a & la, uint32_t il, bool is_ple);
    void layer_a_speculate(const step_ctx & s, graph_builder & gb, layer_a & la, uint32_t il, bool pack_ok);
    ggml_tensor * spec_block_residual(const step_ctx & s, graph_builder & gb, ggml_context * c, ggml_tensor * rp, uint32_t iln);
    void spec_outputs(const step_ctx & s, layer_a & la, ggml_tensor * selnext, ggml_tensor * scores, bool pack_ok);
    void layer_a_outputs(const step_ctx & s, layer_a & la, bool pack_ok);
    bool cache_layer_a(const step_ctx & s, layer_a & la, uint32_t il, bool pack_ok, bool & cached, std::string & err);
    bool layer_a_run(const step_ctx & s, layer_a & la, uint32_t il, bool replayable, bool pack_ok, std::string & err);
    void read_routing(const step_ctx & s, bool packed);
    void score_predictions(const step_ctx & s, uint32_t il);
    float w_tok(int64_t e, int64_t j) const;
    bool decode_moe(step_ctx & s, uint32_t il, bool packed, std::string & err);
    void moe_collect(const step_ctx & s, moe_layer & ml);
    bool moe_fetch(step_ctx & s, moe_layer & ml, std::string & err);
    void moe_classify(const step_ctx & s, moe_layer & ml);
    bool moe_cpu(step_ctx & s, moe_layer & ml, std::string & err);
    void moe_finish(const step_ctx & s, moe_layer & ml);
    graph_ctx moe_expert_graph(const step_ctx & s, const std::vector<int> & which, ggml_tensor * in, ggml_tensor * out);
    void moe_part(const step_ctx & s, const moe_layer & ml, const std::vector<int> & which, ggml_tensor * in, ggml_tensor * out);
    void moe_add_cpu(const step_ctx & s, moe_layer & ml, const std::vector<int> & which, bool first);
    void moe_gpu_launch(const step_ctx & s, const moe_layer & ml, const std::vector<int> & which);
    void moe_late_upload(const step_ctx & s, const moe_layer & ml, const std::vector<int> & which,
                         std::chrono::steady_clock::time_point tm0);
    void moe_gpu_settle();
    void issue_prefetch(const step_ctx & s, moe_layer & ml);
    void part_rows_view(const step_ctx & s, const moe_layer & ml, const std::vector<int> & which, const tier_view & tv);
    void part_rows_pairs(const step_ctx & s, const moe_layer & ml, const std::vector<int> & which, const tier_view & tv,
                         std::chrono::steady_clock::time_point tm0);
    void part_rows(const step_ctx & s, const moe_layer & ml, const std::vector<int> & which);
    void reduce_rows(const step_ctx & s, moe_layer & ml);
    bool cbatch_moe(step_ctx & s, uint32_t il, std::string & err);
    void cbatch_setup(const step_ctx & s, cbatch_layer & cl);
    bool cbatch_begin_chunk(cbatch_layer & cl, size_t cb);
    bool cbatch_chunk(step_ctx & s, cbatch_layer & cl, size_t c0, std::string & err);
    uint32_t cbatch_stage(cbatch_layer & cl, size_t c0, uint32_t n, const std::vector<expert_handle> & ch,
                          std::vector<int32_t> & mem_t, std::vector<int32_t> & mem_s);
    void cbatch_build_term(const step_ctx & s, cbatch_layer & cl, cb_term & tm, std::vector<int32_t> & members);
    void cbatch_run(const step_ctx & s, cbatch_layer & cl, uint32_t m, uint32_t n, std::chrono::steady_clock::time_point tm0);
    void step_head(const step_ctx & s);
    bool step_mtp(const step_ctx & s, std::string & err);
    // draft head
    ggml_tensor * mtp_mask(ggml_context * c, int64_t pos, int64_t n, ggml_context *& mctx, ggml_backend_buffer_t & mbuf);
    void mtp_first_half(graph_builder & gb, ggml_context * c, ggml_cgraph * g, ggml_tensor * h, ggml_tensor * e,
                        ggml_tensor * pv, ggml_tensor * mask, int il, int64_t n);
    void mtp_cpu_moe(int64_t n, int il);
    ggml_tensor * mtp_second_half(ggml_context * c, ggml_cgraph * g, int64_t pos, int64_t n, int il, bool scoring,
                                  ggml_tensor *& logits);
    void mtp_readback(int64_t pos, int64_t n, const int32_t * actual, int64_t n_actual, ggml_tensor * logits, ggml_tensor * am);

    const model_index * mi_ = nullptr;
    engine_config       cfg_;
    int                 n_threads_ = 8;
    std::atomic<uint64_t> prefill_progress_{0};   // layers_done:16 | n_layer:16 | T:32, see prefill_progress()
    hparams             hp_;

    weights        w_;    // dense core, GPU
    weights        wh_;   // PLE table + (for prefill) expert tensors, host mmap
    expert_cache   ec_;
    state          st_;
    prefill_streamer pf_;   // bulk expert reader, prefill only

    // A decode-time graph for a gated-DeltaNet layer references nothing that
    // changes with position: no n_past, no n_kv, no mask. Those 36 of 48 layers
    // can therefore be built once and replayed. Each keeps its own allocator so
    // its tensor addresses stay put between tokens, which is what lets ggml's
    // CUDA graph capture survive instead of being invalidated every layer.
    struct layer_graph {
        bool allocated = false;   // the allocator's pass ran once for this cached graph; replays skip it
        ggml_context * ctx = nullptr;
        ggml_cgraph  * gf  = nullptr;
        ggml_gallocr_t ga  = nullptr;
    };
    std::vector<layer_graph> gA_;
    std::vector<int64_t>     gA_bucket_;   // attention layers: the block bucket the graph was built for

    // Decode-time sparse attention state; see graph_builder::sparse_attn_decode.
    // Per attention layer a cache of pooled block keys; shared static tables,
    // the block bias and the five per-token inputs. Allocated before the VRAM
    // tier so it is accounted for.
    qsa_decode_inputs          qd_, qdk_[MTP_MAX_DRAFTS];   // qdk_[k-1]: position k of a multi-token decode step (k = 1..)
    ggml_tensor *              inp_pos_one_ = nullptr;   // I32 [4*Bd]: per-position [p,p,p,0] for single-position attention calls
    void qsa_decode_prepare_k(int k, int32_t n_past_k);   // qdk_[k-1]'s inputs: the previous position's bias plus its own window
    void qsa_decode_prepare2(int32_t n_past2) { qsa_decode_prepare_k(1, n_past2); }       // qd2_'s inputs, bias from qd_'s plus its own window; one bucket for both
    // Rollback snapshots (MTP): per DeltaNet layer the state and conv history as
    // they stood after the first token of a two-token step, and the PLE conv.
    ggml_context *             rbctx_ = nullptr;
    ggml_backend_buffer_t      rbbuf_ = nullptr;
    std::vector<ggml_tensor *> rb_rs_, rb_conv_;
    ggml_tensor *              rb_ple_conv_ = nullptr;
    bool                       rb_valid_ = false;   // the snapshots describe the current state minus one token
    int                        rb_depth_ = 0, rb_nsnap_ = 1;   // positions that can be rolled back after the last step; snapshots allocated
    std::vector<uint8_t>       gA_T_;               // per layer: the T the cached graph was built for
    ggml_tensor *              t_mtp_emb_ = nullptr;             // device [n_embd, Bd]: the head's next-token embeddings
    ggml_tensor *              h_mtp_tok_ = nullptr, * h_mtp_emb_ = nullptr;   // host: their gather
    std::vector<ggml_tensor *> pool_cache_;   // per layer; null for recurrent layers
    ggml_context *             qctx_ = nullptr;
    ggml_backend_buffer_t      qbuf_ = nullptr;
    uint32_t                   qsa_ratio_ = 0;
    bool                       pool_dirty_ = true;   // caches stale: after a prefill or a reset
    void qsa_decode_prepare(int32_t n_past);         // per-token inputs, bias window, bucket, rebuild

    ggml_gallocr_t galloc_gpu_ = nullptr;
    ggml_gallocr_t galloc_cpu_ = nullptr;
    // The decode GPU MoE graph gets its own allocator and outlives its launch:
    // it is submitted with ggml_backend_graph_compute_async and settled after
    // the I/O wait and the CPU MoE, so it must not share an arena with any
    // graph allocated in between.
    ggml_gallocr_t galloc_moe_ = nullptr;
    ggml_context * moe_ctx_   = nullptr;   // in-flight async per-expert MoE graph (fallback path), freed at settle
    bool           moe_inflight_ = false;  // a GPU MoE graph has been launched and not settled

    // Persistent per-layer GPU MoE graph: three ggml_mul_mat_id over the
    // layer's VRAM tier, selecting experts by slot through t_gids_ with the
    // gate weights in t_gw_. Always n_expert_used entries -- an expert that is
    // not in VRAM takes slot 0 with weight 0, an exact zero contribution -- so
    // the graph's shape never changes and ggml captures it as a CUDA graph.
    // The per-expert path built ~50 nodes and launched ~50 kernels per layer,
    // 23 us per expert of almost pure CPU time; this is a handful of nodes.
    struct moe_graph {
        ggml_context * ctx = nullptr;
        ggml_cgraph  * gf  = nullptr;
        ggml_gallocr_t ga  = nullptr;
    };
    std::vector<moe_graph> gM_;
    ggml_tensor * t_gids_ = nullptr, * t_gw_ = nullptr;
    // Pinned host staging for the per-layer id/weight uploads, so they can be
    // queued asynchronously behind the promotion copies instead of waiting
    // for them on the copy engine (measured 0.1 ms per layer, 1 s per 200
    // tokens, once the VRAM tier started promoting freely).
    ggml_context *        pctx_ = nullptr;
    ggml_backend_buffer_t pbuf_ = nullptr;
    ggml_tensor *         p_gids_ = nullptr, * p_gw_ = nullptr;
    // In-graph VRAM MoE: per-layer residency tables on the device (slot and
    // mask by expert id), their pinned staging, and the version each holds.
    bool                  moe_in_graph_ = false;
    ggml_context *        vctx_ = nullptr;   ggml_backend_buffer_t vbuf_ = nullptr;
    std::vector<ggml_tensor *> t_vslot_, t_vmask_;
    ggml_tensor *         p_vslot_ = nullptr, * p_vmask_ = nullptr;
    std::vector<uint64_t> vslot_ver_;
    int                   n_late_ = 0;
    int64_t               late_w_ = 0;   // row width of t_gids_/t_gw_   // slots of the late fold (experts promoted this token)
    ggml_tensor *         t_rscale_ = nullptr;           // skip_miss: [1] gate renormalisation of the routed sum
    bool                  deferred_wait_ = false;        // skip_miss: a fetch's reads are in flight, not yet waited for
public:
    // For a stall watchdog: what the expert cache is blocked on (racy reads).
    int    cache_wait_state() const { return ec_.wait_state(); }
    size_t cache_wait_count() const { return ec_.wait_count(); }
private:
    bool build_moe_gpu_graph(uint32_t il);

    ggml_context *        wctx_ = nullptr;   ggml_backend_buffer_t wbuf_ = nullptr;
    // The work tensors above are sized for decode and the short-prompt batch.
    // A streamed prefill needs them at n_batch: that set lives only while one
    // runs (prefill_enter / prefill_leave), in the VRAM the expert tier's
    // dynamic buffer gives up for the duration. The decode pointers are saved
    // here while the prefill set is swapped in.
    struct work_set {
        ggml_tensor * res[2] = {nullptr, nullptr};
        ggml_tensor * cur = nullptr, * emb = nullptr, * sh = nullptr, * pg = nullptr, * pc = nullptr, * ple = nullptr;
        ggml_tensor * inject = nullptr, * sel = nullptr, * w = nullptr, * tok = nullptr, * pos = nullptr, * plei = nullptr;
    };
    work_set              dec_ws_;
    ggml_context *        pwctx_ = nullptr;  ggml_backend_buffer_t pwbuf_ = nullptr;
    ggml_gallocr_t        galloc_pf_dyn_ = nullptr;
    bool                  in_prefill_ = false;
    uint64_t              tier_epoch_seen_ = 0;
    bool prefill_enter(std::string & err);
    void sync_tier_epoch();                // rebuild the replayed graphs if the tier moved
    bool client_lent_ = false;             // vram_lend_begin() without a prefill since
    void prefill_leave();
    void save_work_set(work_set & ws) const;
    void load_work_set(const work_set & ws);
    ggml_tensor *         h_tok_ = nullptr, * h_emb_ = nullptr;   // host: token ids, gathered embeddings
    ggml_context *        hctx_ = nullptr;   ggml_backend_buffer_t hbuf_ = nullptr;
    ggml_gallocr_t        galloc_pf_ = nullptr;

    // VRAM scratch for the cache-served batched prefill: cache_batch_chunk
    // slots per part, sized for the largest layer, viewed per layer at that
    // layer's natural slice stride.
    ggml_backend_buffer_t scr_buf_  = nullptr;
    ggml_context *        scr_ctx_  = nullptr;
    ggml_tensor *         scr_xfer_ = nullptr;
    uint8_t *             scr_base_ = nullptr;
    size_t                scr_part_off_[EXPERT_NPARTS] = {0, 0, 0};
    uint32_t              scr_slots_ = 0;

    // Absolute position -> row in ov_ (n_embd floats each).
    std::vector<int32_t>  ov_pos_;
    std::vector<float>    ov_;

    // persistent, device side
    ggml_tensor * res_[2] = {nullptr, nullptr};
    ggml_tensor * t_cur_ = nullptr, * t_inject_ = nullptr, * t_sel_ = nullptr, * t_w_ = nullptr;
    ggml_tensor * t_emb_ = nullptr;   // token embeddings, before the hc repeat
    ggml_tensor * t_selnext_ = nullptr, * t_selnext2_ = nullptr;
    ggml_tensor * t_specscore_ = nullptr;   // logits of the predicted candidates, best first
    // Decode readback pack: one F32 tensor aliasing the first rows of t_cur_ that
    // holds, in order, this layer's FFN input (n_embd), the gate weights (U), the
    // expert ids as floats (U), the predicted ids as floats (K) and their logits
    // (K), so the host reads everything graph A produced in ONE device sync
    // instead of five. Exact: ids below 2^24 convert to F32 and back exactly.
    ggml_tensor *        t_pack_ = nullptr;
    int64_t              pack_n_ = 0;
    std::vector<float>   pack_host_;
    std::vector<int32_t> pred_next_;          // unpacked prediction, consumed by issue_prefetch
    std::vector<float>   scores_next_;
    std::vector<uint8_t> gA_pack_;            // per layer: the cached graph writes the pack
    ggml_tensor *        p_pc_ = nullptr;     // pinned staging for the CPU partial's async upload
    ggml_tensor *        t_hcmean_ = nullptr; // F32 [hc]: 1/hc each, for the fused stream mean (GPU graphs)
    // MTP draft head (experiment).
    model_index   mi_mtp_;
    weights       wm_;                        // the nextn block's dense part on the GPU
    weights       wmh_;                       // its routed experts in host memory, computed on the CPU per draft
    // Persistent tensors of the split head: device outputs of the first half, host
    // inputs/outputs of the CPU MoE, the partial's device landing.
    ggml_context * mctx_ = nullptr; ggml_backend_buffer_t mbuf_ = nullptr;
    ggml_tensor  * t_m_res_ = nullptr, * t_m_cur_ = nullptr, * t_m_inject_ = nullptr, * t_m_sel_ = nullptr,
                 * t_m_w_ = nullptr, * t_m_sh_ = nullptr, * t_m_pc_ = nullptr;
    ggml_context * mhctx_ = nullptr; ggml_backend_buffer_t mhbuf_ = nullptr;
    ggml_tensor  * h_m_cur_ = nullptr, * h_m_ids_ = nullptr, * h_m_w_ = nullptr, * h_m_partial_ = nullptr;
    state         st_mtp_;                    // its KV: one attention layer at n_ctx
    hparams       hpm_;                       // the MTP file's hparams, the block typed as attention
    bool          mtp_on_ = false, mtp_have_h_ = false, mtp_kv_valid_ = true;
    ggml_tensor * t_hlast_ = nullptr;         // F32 [n_embd, hc, Bd]: the wide residual after the last layer, per position of the last eval
    ggml_tensor * t_mtp_pos_ = nullptr;       // I32 [4*Bd]: the draft's positions
    ggml_tensor * t_mtp_mask_ = nullptr;      // F16 [2*(n_ctx+2)]: the two-row causal mask of a two-position draft, viewed [n_kv, 2]
    int64_t       mtp_h_rows_ = 0;            // rows of t_hlast_ the last eval filled
    int32_t       mtp_draft_ = -1, mtp_draft_top_[3] = { -1, -1, -1 }, mtp_draft2_ = -1;
    int32_t       mtp_drafts_[MTP_MAX_DRAFTS] = { -1, -1, -1 }; int mtp_n_drafts_ = 0;
    std::vector<float> mtp_logits_k_[MTP_MAX_DRAFTS]; bool mtp_have_logits_k_[MTP_MAX_DRAFTS] = { false, false, false };
    std::vector<float> mtp_logits2_; bool mtp_have_logits2_ = false;   // the second draft's distribution, when logits are wanted
    bool          mtp_want_logits_ = false, mtp_have_logits_ = false;
    std::vector<float> mtp_logits_;
    // Run the head for n positions starting at `pos`, reading rows h_row.. of
    // t_hlast_ and e_row.. of t_emb_; `actual` (may be null) are the tokens at
    // positions pos+2.. for scoring, n_actual of them.
    bool mtp_draft(int64_t pos, int64_t n, int64_t h_row, ggml_tensor * e_src, int64_t e_row, const int32_t * actual, int64_t n_actual, std::string & err,
                   ggml_tensor * h_src = nullptr);
    ggml_tensor * t_m_hres_ = nullptr;        // F32 [n_embd, hc, Bd]: the head's residual after its MoE fold, per drafted position
    int64_t       mtp_hres_rows_ = 0, mtp_last_pos_ = -1;   // rows the last draft filled; the last position it drafted
    ggml_tensor * t_sh_ = nullptr, * t_pg_ = nullptr, * t_pc_ = nullptr, * t_ple_ = nullptr;
    ggml_tensor * inp_tok_ = nullptr, * inp_pos_ = nullptr, * inp_ple_ = nullptr;
    // persistent, host side
    ggml_tensor * h_cur_ = nullptr, * h_partial_ = nullptr, * h_ple_ = nullptr, * h_ple_idx_ = nullptr;
    ggml_tensor * h_wd_ = nullptr;   // F32 [n_embd, 2U, 2]: every CPU expert's weighted output row of a decode step, summed once in selection order

    int32_t  n_past_  = 0;
    int64_t  n_vocab_ = 0;
    bool     have_expert_map_ = false;

    std::vector<float>   logits_, xfer_, zeros_;
    std::vector<int32_t> sel_, ids_;
    std::vector<float>   wgt_;
    std::vector<int32_t> pred_;   // last layer's prediction for this one
    std::vector<float>   spec_scores_, pred_margin_;   // their logits and margins to the cut-off
    std::vector<uint8_t> spec_block_mask_;             // by predicted layer
    // Two-ahead predictions in flight: pred2_a_ was made two layers back (and is
    // scored against this layer), pred2_b_ one layer back.
    std::vector<int32_t> pred2_a_, pred2_b_;

    // Per-step scratch (rule 3): sized once in init_step_scratch, then only cleared and
    // refilled -- a step's vectors never allocate. Each has one user at a time.
    std::unique_ptr<moe_layer>    ml_;              // decode MoE state, reset per layer
    std::unique_ptr<cbatch_layer> cl_;              // cache-served batch state, reset per layer
    std::vector<expert_handle>  eh_;                // a step's union of experts (step_ctx::eh)
    std::vector<int32_t>        pos_host_;          // set_positions
    std::vector<int32_t>        ple_rows_host_;     // upload_tok_ple, spec_l0_inputs
    std::vector<int32_t>        vt_slot_;           // upload_vtable without pinned staging
    std::vector<float>          vt_mask_;
    std::vector<int32_t>        sc_sid_, sc_pk_, sc_pj_, sc_gids_;   // CPU/GPU expert ids, pairs, slot tables
    std::vector<float>          sc_sw_, sc_gw_, sc_ww_;              // their gate weights
    std::vector<int>            sc_hot_, sc_cold_;                   // part_rows split by source file
    std::vector<ggml_tensor *>  sc_tg_, sc_tu_, sc_td_;              // moe_expert_graph's aliased blocks
    std::vector<uint32_t>       pf1_, pf2_;                          // issue_prefetch's L+1 / L+2 sets
    std::vector<uint16_t>       mtp_mask_host_;     // mtp_mask
    std::vector<int32_t>        mtp_ids_host_, mtp_top_;             // mtp_cpu_moe, mtp_readback
    std::vector<float>          mtp_w_host_, mtp_keep_;              // mtp_cpu_moe, mtp_draft_next
    void init_step_scratch();
};

} // namespace qwfn
