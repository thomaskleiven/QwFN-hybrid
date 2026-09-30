// qwfn-gen -- prefill a prompt, then decode greedily, tracking position.
//
// Works in token ids so it can be compared against llama.cpp directly without
// needing a tokenizer.

#include "qwfn_check.h"
#include "qwfn_engine.h"
#include "qwfn_model.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace qwfn;

struct gen_opts {
    std::vector<int32_t> prompt, replay;
    std::string cold_path, save_replay;   // --save-replay FILE: the generated ids, one per line, for a later --replay-file
    bool want_ppl = false;   // with --replay-file: mean NLL of the replayed tokens (a quality number)
    bool pair_test = false, rollback_test = false;   // exercise the multi-token decode step without the head
    int  multi_test = 2;                              // tokens per step for --multi-test / --rollback-test
    int n_gen = 16;
    engine_config cfg;
};

// Appends the ids of a file (separated by spaces, commas or newlines). False if it cannot be opened.
static bool append_ids(const char * path, std::vector<int32_t> & out) {
    assert_that(path != nullptr, "flag value is an argument");
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    const size_t n0 = out.size();
    int v; while (fscanf(f, "%d%*[ ,\n\t\r]", &v) == 1) out.push_back(v);
    fclose(f);
    assert_that(out.size() >= n0, "ids are only appended");
    return true;
}

// Engine flags, part 1. True if argv[i] was one (i then points at its last argument).
static bool parse_engine_flag(const std::string & a, int argc, char ** argv, int & i, engine_config & cfg) {
    assert_that(i >= 2 && i < argc, "flag index lies inside argv");
    assert_that(a == argv[i], "flag text is the current argument");
    auto next = [&]() { return argv[++i]; };
    if (a == "--ctx"     && i + 1 < argc) { cfg.n_ctx = (uint32_t) qwfn::arg_int("--ctx", next(), 1, 2147483647); return true; }
    if (a == "--batch"   && i + 1 < argc) { cfg.n_batch = (uint32_t) qwfn::arg_int("--batch", next(), 1, 1 << 20); return true; }
    if (a == "--ram"     && i + 1 < argc) { cfg.ram_bytes = (size_t)(qwfn::arg_float("--ram", next(), 0, 1e6) * 1e9); return true; }
    if (a == "--vram"    && i + 1 < argc) { cfg.vram_bytes = (size_t)(qwfn::arg_float("--vram", next(), 0, 1e6) * 1e9); return true; }
    if (a == "--threads" && i + 1 < argc) { cfg.n_threads = (int) qwfn::arg_int("--threads", next(), 1, 1024); return true; }
    if (a == "--cpu")    { cfg.use_gpu = false; return true; }
    if (a == "--io-threads" && i + 1 < argc) { cfg.io_threads = true; cfg.io_workers = (unsigned) qwfn::arg_int("--io-threads", next(), 1, 1024); return true; }
    if (a == "--io-uring") { cfg.io_threads = false; return true; }
    if (a == "--state-host" && i + 1 < argc) {   // none | idx | kv | kv,idx
        std::string v = next();
        cfg.idx_host = v.find("idx") != std::string::npos;
        cfg.kv_host  = v.find("kv")  != std::string::npos;
        return true;
    }
    if (a == "--kv" && i + 1 < argc) { std::string v = next();
        cfg.type_k = cfg.type_v = (v == "q8_0") ? GGML_TYPE_Q8_0 :
                                  (v == "q4_0") ? GGML_TYPE_Q4_0 : GGML_TYPE_F16; return true; }
    if (a == "--no-reuse") { cfg.reuse_graphs = false; return true; }
    if (a == "--no-speculate") { cfg.speculate = false; return true; }
    if (a == "--skip-miss") { cfg.skip_miss = true; return true; }
    if (a == "--reserve" && i + 1 < argc) { cfg.vram_reserve = (size_t) qwfn::arg_float("--reserve", next(), 0, 1e7) * (1ull << 20); return true; }
    if (a == "--spec-depth" && i + 1 < argc) { cfg.speculate_depth = (uint32_t) qwfn::arg_int("--spec-depth", next(), 0, QWFN_SPEC_MAX);
        if (cfg.speculate_depth == 0) { cfg.speculate = false; }
        return true; }
    if (a == "--spec-ahead" && i + 1 < argc) { cfg.speculate_ahead = (uint32_t) qwfn::arg_int("--spec-ahead", next(), 0, 16); return true; }
    if (a == "--spec-depth2" && i + 1 < argc) { cfg.speculate_depth2 = (uint32_t) qwfn::arg_int("--spec-depth2", next(), 0, QWFN_SPEC_MAX); return true; }
    if (a == "--spec-margin" && i + 1 < argc) { cfg.spec_margin = (float) qwfn::arg_float("--spec-margin", next(), 0, 1000); return true; }
    if (a == "--spec-gate-inflight" && i + 1 < argc) { cfg.spec_gate_inflight = (uint32_t) qwfn::arg_int("--spec-gate-inflight", next(), 0, 1 << 20); return true; }
    if (a == "--spec-block") { cfg.spec_block = true; return true; }
    if (a == "--spec-block-layers" && i + 1 < argc) { cfg.spec_block = true; cfg.spec_block_layers = next(); return true; }
    if (a == "--mtp" && i + 1 < argc) { cfg.mtp_path = next(); cfg.rollback_snapshots = true; return true; }
    return false;
}

// Engine flags, part 2.
static bool parse_tuning_flag(const std::string & a, int argc, char ** argv, int & i, engine_config & cfg) {
    assert_that(i >= 2 && i < argc, "flag index lies inside argv");
    assert_that(a == argv[i], "flag text is the current argument");
    auto next = [&]() { return argv[++i]; };
    if (a == "--mtp-drafts" && i + 1 < argc) { cfg.mtp_drafts = (uint32_t) qwfn::arg_int("--mtp-drafts", next(), 1, engine::MTP_MAX_DRAFTS); return true; }
    if (a == "--ram-frac" && i + 1 < argc) { cfg.ram_frac = qwfn::arg_float("--ram-frac", next(), 0, 1); return true; }
    // 0 forces the batched prefill path at every size. Reference runs want
    // this: token-by-token prefill is a different (equally valid) summation
    // order, and this model turns that into different tokens.
    if (a == "--prefill-decode-max" && i + 1 < argc) { cfg.prefill_decode_max = (uint32_t) qwfn::arg_int("--prefill-decode-max", next(), 0, 1 << 20); return true; }
    if (a == "--gate-drop" && i + 1 < argc) { cfg.gate_drop = (float) qwfn::arg_float("--gate-drop", next(), 0, 1); return true; }
    if (a == "--prefill-chunk" && i + 1 < argc) { cfg.prefill_chunk = (uint32_t) qwfn::arg_int("--prefill-chunk", next(), 0, 1 << 20); return true; }
    if (a == "--vram-reserve" && i + 1 < argc) { cfg.vram_reserve = (size_t)(qwfn::arg_float("--vram-reserve", next(), 0, 1e7) * 1e6); return true; }
    if (a == "--no-prefill-overlap") { cfg.prefill_overlap = false; return true; }
    if (a == "--promote" && i + 1 < argc) { cfg.promote_per_layer = (uint32_t) qwfn::arg_int("--promote", next(), 0, 1 << 16); return true; }
    if (a == "--evict" && i + 1 < argc) { std::string v = next();
        cfg.evict_policy = v == "lfu" ? 1 : v == "hybrid" ? 2 : 0; return true; }
    if (a == "--no-qsa") { cfg.use_qsa = false; return true; }
    if (a == "--prefill-cpu") { cfg.prefill_on_gpu = false; return true; }
    if (a == "--prefill-gpu") { cfg.prefill_on_gpu = true; return true; }  // FAST BUT BROKEN >T~200
    if (a == "--ubatch-kv" && i + 1 < argc) { cfg.ubatch_kv_product = (uint64_t)(qwfn::arg_float("--ubatch-kv", next(), 0, 1e7) * 1e6); return true; }
    if (a == "--indexer-top-k" && i + 1 < argc) { cfg.indexer_top_k = (uint32_t) qwfn::arg_int("--indexer-top-k", next(), 0, 1 << 20); return true; }
    return false;
}

// Run flags: prompt, replay, tests. 1 if argv[i] was one, 0 if not, -1 on an error (already reported).
static int parse_run_flag(const std::string & a, int argc, char ** argv, int & i, gen_opts & o) {
    assert_that(i >= 2 && i < argc, "flag index lies inside argv");
    assert_that(a == argv[i], "flag text is the current argument");
    auto next = [&]() { return argv[++i]; };
    engine_config & cfg = o.cfg;
    if (a == "--gen"     && i + 1 < argc) { o.n_gen = (int) qwfn::arg_int("--gen", next(), 0, 1 << 24); return 1; }
    if (a == "--ppl") { o.want_ppl = true; return 1; }
    if (a == "--pair-test") { o.pair_test = true; return 1; }          // decode the replay two tokens per step
    if (a == "--multi-test" && i + 1 < argc) { o.pair_test = true; o.multi_test = (int) qwfn::arg_int("--multi-test", next(), 1, engine::MTP_MAX_DRAFTS + 1); cfg.mtp_drafts = (uint32_t) std::max(1, o.multi_test - 1); return 1; }   // K tokens per step from the replay
    if (a == "--rollback-test") { o.rollback_test = true; cfg.rollback_snapshots = true; if (cfg.mtp_drafts < (uint32_t) std::max(1, o.multi_test - 1)) cfg.mtp_drafts = (uint32_t) std::max(1, o.multi_test - 1); return 1; }  // every token as the first of a pair with a wrong second, then roll back
    if (a == "--save-replay" && i + 1 < argc) { o.save_replay = next(); return 1; }
    if (a == "--replay-file" && i + 1 < argc) {
        // Feed these ids as the "generated" tokens instead of sampling, so
        // every configuration sees identical routing. GPU decode is
        // nondeterministic, and comparing different generated texts was
        // measured to produce 1.5 tok/s of phantom variance.
        if (!append_ids(next(), o.replay)) { fprintf(stderr, "error: cannot open replay file\n"); return -1; }
        return 1;
    }
    if (a == "--cold" && i + 1 < argc) { o.cold_path = next(); cfg.use_cold_tier = true; return 1; }
    if (a == "--prompt-file" && i + 1 < argc) {
        // Long contexts blow past ARG_MAX on the command line.
        if (!append_ids(next(), o.prompt)) { fprintf(stderr, "error: cannot open prompt file\n"); return -1; }
        return 1;
    }
    if (a == "--prompt"  && i + 1 < argc) {
        std::string t = next(); size_t p = 0;
        while (p < t.size()) {
            size_t c = t.find(',', p); if (c == std::string::npos) c = t.size();
            o.prompt.push_back((int32_t) qwfn::arg_int("--prompt id", t.substr(p, c - p).c_str(), 0, 2147483647)); p = c + 1;
        }
        return 1;
    }
    return 0;
}

// Unknown flags and flags missing their value are ignored. False on an error (already reported).
static bool parse_args(int argc, char ** argv, gen_opts & o) {
    assert_that(argc >= 2, "caller checked the model argument");
    o.cfg.n_ctx = 4096; o.cfg.n_batch = 128; o.cfg.ram_bytes = 8e9; o.cfg.vram_bytes = 0;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (parse_engine_flag(a, argc, argv, i, o.cfg) || parse_tuning_flag(a, argc, argv, i, o.cfg)) continue;
        if (parse_run_flag(a, argc, argv, i, o) < 0) return false;
    }
    if (o.prompt.empty()) o.prompt = { 9707, 11, 1879, 0 };
    assert_that(!o.prompt.empty(), "a default prompt stands in for none");
    return true;
}

static int argmax(const engine & eng, const float * l) {
    assert_that(l != nullptr, "logits present");
    assert_that(eng.n_vocab() > 0, "engine has a vocabulary");
    int b = 0; for (int64_t v = 1; v < eng.n_vocab(); v++) if (l[v] > l[b]) b = (int) v; return b;
}

struct spec_stats {
    uint64_t steps = 0, acc = 0, single = 0;
    std::vector<uint64_t> accepted_at = std::vector<uint64_t>(engine::MTP_MAX_DRAFTS + 1, 0);   // verify steps that kept exactly j drafts
};

// The speculative loop's position: `next` is the token to feed, replay index i.
struct spec_state {
    int i = 0, produced = 0, rb_at = 1;
    int32_t next = 0;
    std::vector<int32_t> drafts;
};

// One qwfn-gen run on an initialised engine: prefill, decode, diagnostics, statistics.
struct gen_run {
    engine & eng;
    gen_opts & o;
    std::string & err;
    std::vector<int32_t> hist;
    const float * lg = nullptr;
    bool use_mtp = false;
    double nll_sum = 0.0; int nll_n = 0;
    spec_stats sp;
    std::chrono::steady_clock::time_point t0;
    // Decode progress by segment (QWFN_SEGMENTS=N).
    int seg_n = 0, seg_next = 0;
    std::chrono::steady_clock::time_point seg_t;
    expert_cache_stats seg_st;

    gen_run(engine & e, gen_opts & op, std::string & er) : eng(e), o(op), err(er) {}

    bool prefill();
    bool decode_begin();
    void score(const float * l, int32_t tok);
    void segment(int done);
    bool decode_plain();
    int32_t next_after(const float * l, int k);
    void spec_l0(const int32_t * toks, int n);
    bool collect_drafts(spec_state & st);
    int single_step(spec_state & st);
    int verify_step(spec_state & st);
    bool decode_speculative();
    bool second_prompt(const char * path);
    void print_run_stats();
    void print_io_profile();
    void print_cache_stats();
    void print_spec_stats();
    void print_prediction_stats();
    void print_io_stats();
};

// ---- prefill in ubatches ----------------------------------------------
bool gen_run::prefill() {
    assert_that(!o.prompt.empty(), "prompt is never empty");
    hist = o.prompt;
    const auto ts = std::chrono::steady_clock::now();
    int32_t done = 0;
    while (done < (int32_t) o.prompt.size()) {
        const int32_t take = std::min<int32_t>(o.cfg.n_batch, (int32_t) o.prompt.size() - done);
        lg = eng.eval(hist.data(), done + take, take, err);
        if (!lg) { fprintf(stderr, "prefill failed: %s\n", err.c_str()); return false; }
        done += take;
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
    printf("prefill: %zu tokens in %.2f s  (%.1f tok/s)\n", o.prompt.size(), dt, o.prompt.size() / dt);
    assert_that(lg != nullptr, "a non-empty prefill leaves logits");
    return true;
}

// Start of the greedy decode: header, clock, and the checks and counters both loops share.
bool gen_run::decode_begin() {
    assert_that(lg != nullptr, "prefill left logits");
    printf(o.replay.empty() ? "generated:" : "replaying:");
    if (!o.replay.empty()) o.n_gen = std::min<int>(o.n_gen, (int) o.replay.size());
    t0 = std::chrono::steady_clock::now();
    use_mtp = !o.cfg.mtp_path.empty();
    if ((o.pair_test || o.rollback_test) && o.replay.empty()) { fprintf(stderr, "--pair-test / --rollback-test need --replay-file\n"); return false; }
    // Decode progress by segment (QWFN_SEGMENTS=N): tok/s and the tiers' share over
    // each N tokens, to see how long the decode after a prompt takes to warm up.
    seg_n = (int) env_int("QWFN_SEGMENTS", 0, 0, 1 << 24);
    seg_next = seg_n; seg_t = t0; seg_st = eng.cache_stats();
    assert_that(o.replay.empty() || o.n_gen <= (int) o.replay.size(), "a replay bounds the decode");
    return true;
}

void gen_run::score(const float * l, int32_t tok) {   // -log softmax(l)[tok]
    assert_that(l != nullptr, "logits present");
    assert_that(eng.n_vocab() > 0, "engine has a vocabulary");
    float mx = l[0]; for (int64_t v = 1; v < eng.n_vocab(); v++) mx = std::max(mx, l[v]);
    double z = 0.0; for (int64_t v = 0; v < eng.n_vocab(); v++) z += std::exp((double) l[v] - mx);
    const double v = -((double) l[tok] - mx - std::log(z));
    nll_sum += v; nll_n++;
    // QWFN_TOPK_DUMP=FILE: the top-K (QWFN_TOPK, default 32) log-probabilities per scored
    // position, "i id:lp ...", the format tools/llama_nll.cpp --topk writes (for KL).
    static FILE * topk_f = getenv("QWFN_TOPK_DUMP") ? fopen(getenv("QWFN_TOPK_DUMP"), "w") : nullptr;
    if (topk_f) {
        static const int K = (int) env_int("QWFN_TOPK", 32, 1, 1 << 16);
        if (K > eng.n_vocab()) { fprintf(stderr, "error: QWFN_TOPK=%d exceeds the vocabulary (%lld)\n", K, (long long) eng.n_vocab()); exit(2); }
        assert_that(K >= 1, "QWFN_TOPK at least 1");
        static std::vector<int> idx; idx.resize(eng.n_vocab());
        for (int64_t k = 0; k < eng.n_vocab(); k++) idx[k] = (int) k;
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
        const double lz = mx + std::log(z);
        fprintf(topk_f, "%d", nll_n - 1);
        for (int k = 0; k < K; k++) fprintf(topk_f, " %d:%.6f", idx[k], (double) l[idx[k]] - lz);
        fprintf(topk_f, "\n"); fflush(topk_f);
    }
    static const bool verbose = getenv("QWFN_PPL_VERBOSE") != nullptr;
    if (verbose) { int am = 0; for (int64_t k = 1; k < eng.n_vocab(); k++) if (l[k] > l[am]) am = (int) k;
        fprintf(stderr, "[nll] #%d tok %d: %.6f argmax %d\n", nll_n, tok, v, am); }
}

void gen_run::segment(int done) {
    if (seg_n <= 0 || done < seg_next) return;
    assert_that(seg_next >= seg_n, "a segment ends at least one segment in");
    const auto now = std::chrono::steady_clock::now(); const expert_cache_stats & s = eng.cache_stats();
    const double dts = std::chrono::duration<double>(now - seg_t).count();
    const uint64_t lk = s.lookups - seg_st.lookups, gh = s.gpu_hits - seg_st.gpu_hits, hh = s.hits - seg_st.hits;
    fprintf(stderr, "[segment] tokens %d-%d: %.2f tok/s, %.1f%% from VRAM, %.1f%% hit, %.2f GB read\n",
            seg_next - seg_n, done, (done - (seg_next - seg_n)) / dts, 100.0 * gh / std::max<uint64_t>(1, lk),
            100.0 * hh / std::max<uint64_t>(1, lk), (s.bytes_from_disk - seg_st.bytes_from_disk) / 1e9);
    seg_t = now; seg_st = s; seg_next += seg_n;
    assert_that(seg_next > seg_n, "segments advance");
}

bool gen_run::decode_plain() {
    assert_that(!use_mtp && !o.pair_test && !o.rollback_test, "plain loop selected");
    assert_that(lg != nullptr, "prefill left logits");
    for (int i = 0; i < o.n_gen; i++) {
        int best = 0;
        if (o.replay.empty()) best = argmax(eng, lg);
        else { best = o.replay[i]; if (o.want_ppl) score(lg, best); }
        printf(" %d", best);
        fflush(stdout);
        hist.push_back(best);
        segment(i + 1);
        lg = eng.eval(hist.data(), (int32_t) hist.size(), 1, err);
        if (!lg) { fprintf(stderr, "\ndecode failed: %s\n", err.c_str()); return false; }
    }
    return true;
}

int32_t gen_run::next_after(const float * l, int k) {   // the token at replay index k, or argmax
    if (o.replay.empty()) return argmax(eng, l);
    assert_that(k >= 0, "replay index is not negative");
    assert_that(l != nullptr, "logits present");
    return k < (int) o.replay.size() ? o.replay[k] : -1;
}

// Layer 0's reads for the tokens the next eval will take, issued as soon as
// they are known: the bonus token before the head runs, the pair once it has.
void gen_run::spec_l0(const int32_t * toks, int n) {
    if (!use_mtp) return;   // no lead without the head: the eval follows at once
    assert_that(toks != nullptr && n > 0, "tokens to lead with");
    const size_t n0 = hist.size();
    for (int k = 0; k < n; k++) hist.push_back(toks[k]);
    eng.spec_layer0(hist.data(), (int32_t) hist.size(), n, err);
    for (int k = 0; k < n; k++) hist.pop_back();
    assert_that(hist.size() == n0, "history restored after the lead");
}

// The drafts for this step (the head's, or the replay's for the tests), trimmed to the room left.
bool gen_run::collect_drafts(spec_state & st) {
    assert_that(st.produced < o.n_gen, "called while tokens remain");
    st.drafts.clear();
    if (o.pair_test || o.rollback_test) {
        // At most MTP_MAX_DRAFTS: a verify step carries no more, and the per-step
        // statistics are sized for it.
        const int K = std::min(std::max(1, o.multi_test - 1), (int) engine::MTP_MAX_DRAFTS);
        for (int k = 1; k <= K; k++) {
            if (st.i + k >= (int) o.replay.size()) break;
            const int32_t d = (o.rollback_test && k == st.rb_at)
                ? (int32_t) ((o.replay[st.i + k] + env_int("QWFN_RB_WRONG", 1, 0, 1 << 24)) % eng.n_vocab())
                : o.replay[st.i + k];
            st.drafts.push_back(d);
        }
    } else if (use_mtp && eng.mtp_draft_id() >= 0) {
        if (!eng.mtp_draft_more((int) o.cfg.mtp_drafts, err)) { fprintf(stderr, "\ndraft: %s\n", err.c_str()); return false; }
        for (int k = 0; k < eng.mtp_draft_count(); k++) st.drafts.push_back(eng.mtp_draft_k(k));
    }
    // Room for the drafts: within the requested count and the replay.
    while (!st.drafts.empty() && (st.produced + (int) st.drafts.size() >= o.n_gen || (!o.replay.empty() && st.i + (int) st.drafts.size() >= (int) o.replay.size())))
        st.drafts.pop_back();
    assert_that(st.produced + (int) st.drafts.size() < o.n_gen || st.drafts.empty(), "drafts fit the requested count");
    return true;
}

// A step without drafts. 1 to go on, 0 when the replay is exhausted, -1 on an error (reported).
int gen_run::single_step(spec_state & st) {
    assert_that(st.drafts.empty(), "single step has no drafts");
    hist.push_back(st.next);
    lg = eng.eval_decode(hist.data(), (int32_t) hist.size(), 1, err);
    if (!lg) { fprintf(stderr, "\ndecode failed: %s\n", err.c_str()); return -1; }
    printf(" %d", st.next); fflush(stdout); st.produced++; sp.single++;
    const int32_t y = next_after(lg, st.i + 1);
    if (o.want_ppl && !o.replay.empty() && st.i + 1 < o.n_gen) score(lg, o.replay[st.i + 1]);
    st.i += 1;
    if (y < 0) return 0;
    spec_l0(&y, 1);
    if (use_mtp && !eng.mtp_step(&y, 1, err)) { fprintf(stderr, "\nmtp: %s\n", err.c_str()); return -1; }
    st.next = y;
    assert_that(st.produced <= st.i, "every produced token advanced the replay index");
    return 1;
}

// The verify step: a token and K drafts evaluated as one step of K+1 positions; the trunk's pick
// after each position decides how many drafts stand, the rest is rolled back. Returns as single_step.
int gen_run::verify_step(spec_state & st) {
    assert_that(!st.drafts.empty(), "verify step has drafts");
    const int K = (int) st.drafts.size();
    {
        std::vector<int32_t> step; step.push_back(st.next); for (int32_t d : st.drafts) step.push_back(d);
        spec_l0(step.data(), (int) step.size());
        for (int32_t s : step) hist.push_back(s);
    }
    if (!eng.eval_decode(hist.data(), (int32_t) hist.size(), K + 1, err)) { fprintf(stderr, "\ndecode failed: %s\n", err.c_str()); return -1; }
    printf(" %d", st.next); fflush(stdout); st.produced++;
    int j = 0; int32_t y = -1;
    for (j = 0; j < K; j++) {
        const float * lj = eng.logits_pos(j);
        y = next_after(lj, st.i + 1 + j);
        if (o.want_ppl && !o.replay.empty() && st.i + 1 + j < o.n_gen) score(lj, o.replay[st.i + 1 + j]);
        const bool ok = o.rollback_test ? (j + 1 != st.rb_at) : o.pair_test ? true : (y == st.drafts[j]);
        if (!ok) break;
        printf(" %d", st.drafts[j]); fflush(stdout); st.produced++;
    }
    sp.steps++; sp.acc += j; sp.accepted_at[j]++;
    std::vector<int32_t> fed(st.drafts.begin(), st.drafts.begin() + j);
    if (j == K) {
        const float * lK = eng.logits_pos(K);
        y = next_after(lK, st.i + 1 + K);
        if (o.want_ppl && !o.replay.empty() && st.i + 1 + K < o.n_gen) score(lK, o.replay[st.i + 1 + K]);
        st.i += K + 1;
    } else {
        if (!eng.rollback_n(K - j, err)) { fprintf(stderr, "\nrollback: %s\n", err.c_str()); return -1; }
        hist.resize(hist.size() - (size_t) (K - j));
        st.i += j + 1;
    }
    segment(st.i);
    if (y < 0) return 0;
    fed.push_back(y);
    spec_l0(&y, 1);
    if (use_mtp && !eng.mtp_step(fed.data(), (int) fed.size(), err)) { fprintf(stderr, "\nmtp: %s\n", err.c_str()); return -1; }
    st.next = y;
    assert_that(st.produced <= st.i, "every produced token advanced the replay index");
    return 1;
}

// Speculative loop. `next` is the token to feed; the head's draft for the
// token after it (or, in the tests, the replay's own next token / a wrong
// one) rides along as the second of a pair. Position 0's logits verify the
// draft; position 1's are the next token's if it is accepted. The replay
// is the ground truth for acceptance when replaying; argmax when not.
// NLL covers replay[0..n_gen) as in the plain loop.
bool gen_run::decode_speculative() {
    assert_that(use_mtp || o.pair_test || o.rollback_test, "speculative loop selected");
    assert_that(lg != nullptr, "prefill left logits");
    spec_state st;
    st.next = o.replay.empty() ? argmax(eng, lg) : o.replay[0];
    if (o.want_ppl && !o.replay.empty()) score(lg, o.replay[0]);
    if (use_mtp && !eng.mtp_step(&st.next, 1, err)) { fprintf(stderr, "\nmtp: %s\n", err.c_str()); return false; }
    st.drafts.reserve(engine::MTP_MAX_DRAFTS);
    st.rb_at = (int) env_int("QWFN_RB_AT", 1, 0, engine::MTP_MAX_DRAFTS);   // --rollback-test: the draft that is wrong (1-based)
    while (st.produced < o.n_gen) {
        if (!collect_drafts(st)) return false;
        const int rc = st.drafts.empty() ? single_step(st) : verify_step(st);
        if (rc < 0) return false;
        if (rc == 0) break;
    }
    o.n_gen = st.produced;
    return true;
}

// QWFN_SECOND_PROMPT=FILE: append a second prompt (token ids) to the warm engine, as a tool
// result arrives in a running session, and time its prefill.
bool gen_run::second_prompt(const char * path) {
    assert_that(path != nullptr, "called with the variable set");
    std::vector<int32_t> p2; FILE * f = fopen(path, "r"); int v;
    while (f && fscanf(f, "%d", &v) == 1) p2.push_back(v);
    if (f) fclose(f);
    const uint64_t rd0 = eng.prefill_bytes_read(), fr0 = eng.prefill_bytes_from_ram();
    const double g0 = eng.t_pf_graphA, r0 = eng.t_pf_read, m0 = eng.t_pf_moe, w0 = eng.t_warm, b0 = eng.prefill_reader_busy();
    const auto ts = std::chrono::steady_clock::now();
    const float * l2 = nullptr;
    for (size_t done = 0; done < p2.size(); ) {
        const size_t take = std::min<size_t>(o.cfg.n_batch, p2.size() - done);
        hist.insert(hist.end(), p2.begin() + done, p2.begin() + done + take);
        l2 = eng.eval(hist.data(), (int32_t) hist.size(), (int32_t) take, err);
        if (!l2) { fprintf(stderr, "second prompt: %s\n", err.c_str()); return false; }
        done += take;
    }
    const double d2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
    assert_that(hist.size() >= p2.size(), "the second prompt was appended to the history");
    // FNV-1a over the final logits' bytes: a bit-exact fingerprint for regression checks.
    uint64_t h = 1469598103934665603ull;
    const uint8_t * lb = (const uint8_t *) l2;
    for (size_t k = 0; l2 && k < (size_t) eng.n_vocab() * sizeof(float); k++) { h ^= lb[k]; h *= 1099511628211ull; }
    printf("second prompt logits fnv1a %016llx argmax %d\n", (unsigned long long) h, l2 ? argmax(eng, l2) : -1);
    printf("second prompt: %zu tokens at n_past %zu in %.2f s  (%.1f tok/s) | dense %.2f s, reads %.2f s, MoE %.2f s, warm-up %.2f s"
           " | %.2f GB read, %.2f GB from the RAM tier | reader busy %.2f s\n", p2.size(), hist.size() - p2.size(), d2, p2.size() / d2,
           eng.t_pf_graphA - g0, eng.t_pf_read - r0, eng.t_pf_moe - m0, eng.t_warm - w0,
           (eng.prefill_bytes_read() - rd0) / 1e9, (eng.prefill_bytes_from_ram() - fr0) / 1e9, eng.prefill_reader_busy() - b0);
    return true;
}

void gen_run::print_run_stats() {
    assert_that(hist.size() >= o.prompt.size(), "history starts with the prompt");
    assert_that(nll_n >= 0, "scored-token count");
    if (eng.n_replay)
        printf("cached graph replays: %llu; per replay alloc %.0f us, launch %.0f us, wait %.0f us\n", (unsigned long long) eng.n_replay,
               eng.t_replay_alloc / eng.n_replay * 1e6, eng.t_replay_launch / eng.n_replay * 1e6, eng.t_replay_wait / eng.n_replay * 1e6);
    if (eng.t_eval_decode > 0)
        printf("step profile: eval %.2f s = pre %.2f + graphA %.2f + CPU MoE %.2f + io %.2f + head %.2f + rest %.2f | draft head %.2f s\n",
               eng.t_eval_decode, eng.t_pre, eng.t_layerA, eng.t_moe_cpu, eng.t_io - eng.t_io_cbatch, eng.t_head,
               eng.t_eval_decode - eng.t_pre - eng.t_layerA - eng.t_moe_cpu - (eng.t_io - eng.t_io_cbatch) - eng.t_head, eng.t_mtp);
    if (!o.save_replay.empty()) {
        FILE * f = fopen(o.save_replay.c_str(), "wb");
        if (f) { for (size_t i = hist.size() - o.n_gen; i < hist.size(); i++) fprintf(f, "%d\n", hist[i]); fclose(f);
                 printf("saved %d generated ids to %s\n", o.n_gen, o.save_replay.c_str()); }
        else fprintf(stderr, "cannot write %s\n", o.save_replay.c_str());
    }
    if (nll_n) printf("replay NLL: %.4f per token (ppl %.2f) over %d tokens\n", nll_sum / nll_n, std::exp(nll_sum / nll_n), nll_n);
    if (eng.n_exp_skipped) printf("skipped experts: %llu (misses computed without)\n", (unsigned long long) eng.n_exp_skipped);
    if (eng.n_spec_l0) printf("layer-0 speculation: %llu calls, %.2f ms each\n", (unsigned long long) eng.n_spec_l0, 1e3 * eng.t_spec_l0 / eng.n_spec_l0);
    if (eng.t_mtp_pre + eng.t_mtp_moe + eng.t_mtp_post > 0)
        printf("head split: dense half %.2f s | CPU experts + transfers %.2f s | second half + LM head + argmax %.2f s\n", eng.t_mtp_pre, eng.t_mtp_moe, eng.t_mtp_post);
    if (getenv("QWFN_IO_PROFILE") && !eng.prof_io_end.empty() && o.n_gen > 0) print_io_profile();
    if (eng.n_exp_dropped) printf("dropped experts: %llu (gate below --gate-drop; %.1f%% of routed)\n", (unsigned long long) eng.n_exp_dropped,
                                  100.0 * eng.n_exp_dropped / std::max<uint64_t>(1, eng.n_exp_dropped + eng.n_exp_gpu + eng.n_exp_cpu + eng.n_exp_skipped));
}

// Where the decode loop waits for reads, layer by layer, per decoded token.
void gen_run::print_io_profile() {
    assert_that(o.n_gen > 0 && !eng.prof_io_end.empty(), "called with a profile and decoded tokens");
    assert_that(eng.prof_io_begin.size() == eng.prof_io_end.size() && eng.prof_reads.size() == eng.prof_io_end.size(),
                "per-layer io profile vectors are sized together");
    const int n_gen = o.n_gen;
    printf("io profile (ms per token; begin = wait when the layer's fetch is issued, end = wait for its reads after the ready pass, reads = demand reads issued by the fetch):\n");
    double tb = 0, te = 0; uint64_t tr = 0;
    for (size_t il = 0; il < eng.prof_io_end.size(); il++) {
        tb += eng.prof_io_begin[il]; te += eng.prof_io_end[il]; tr += eng.prof_reads[il];
        printf("  L%-2zu %s begin %5.2f end %5.2f reads %5.2f%s", il, eng.is_attn_layer((uint32_t) il) ? "A" : "R",
               1e3 * eng.prof_io_begin[il] / n_gen, 1e3 * eng.prof_io_end[il] / n_gen, (double) eng.prof_reads[il] / n_gen, (il % 3 == 2) ? "\n" : " |");
    }
    printf("\n  total begin %.1f ms, end %.1f ms, %.1f demand reads per token; layer 0 begin %.2f ms\n",
           1e3 * tb / n_gen, 1e3 * te / n_gen, (double) tr / n_gen, 1e3 * eng.prof_io_begin[0] / n_gen);
}

void gen_run::print_cache_stats() {
    const auto & s = eng.cache_stats();
    assert_that(s.hit_rate() >= 0.0, "hit rate is not negative");
    assert_that(s.gpu_rate() >= 0.0, "VRAM rate is not negative");
    printf("expert cache: %.1f%% hit, %.1f%% from VRAM, %.2f GB from disk\n",
           100.0 * s.hit_rate(), 100.0 * s.gpu_rate(), s.bytes_from_disk / 1e9);
    printf("              tiers: %llu promotions, %llu RAM evictions, %llu cold-file reads, %llu upgrades\n",
           (unsigned long long) s.promotions, (unsigned long long) s.evictions,
           (unsigned long long) s.cold_tier_reads, (unsigned long long) s.upgrades);
    {
        const auto c = eng.ram_census();
        printf("              RAM tier at end: %llu slots = %llu hot + %llu cold (%llu of them hot-worthy) + %llu empty; %llu prefetched unused, %llu in flight; %llu experts marked hot-worthy\n",
               (unsigned long long) c.slots, (unsigned long long) c.hot, (unsigned long long) c.cold, (unsigned long long) c.cold_hotw,
               (unsigned long long) c.empty, (unsigned long long) c.speculative, (unsigned long long) c.inflight, (unsigned long long) c.hotw_marked);
    }
    printf("decode split: graphA(GPU) %.2f s | MoE gpu %.2f s (%llu experts, sync-wait %.2f s) | MoE cpu %.2f s (%llu experts) | io %.2f s\n",
           eng.t_layerA, eng.t_moe_gpu, (unsigned long long) eng.n_exp_gpu, eng.t_moe_gpu_sync,
           eng.t_moe_cpu, (unsigned long long) eng.n_exp_cpu, eng.t_io);
    if (eng.n_exp_cpu && eng.n_exp_gpu)
        printf("              per expert: gpu %.0f us, cpu %.0f us  (%.1fx)\n",
               eng.t_moe_gpu / eng.n_exp_gpu * 1e6, eng.t_moe_cpu / eng.n_exp_cpu * 1e6,
               (eng.t_moe_cpu / eng.n_exp_cpu) / (eng.t_moe_gpu / eng.n_exp_gpu));
    if (eng.n_layerA_rec && eng.n_layerA_attn)
        printf("              graphA per decode layer: recurrent %.0f us (x%llu), attention %.0f us (x%llu)"
               " [attention: build+alloc %.0f us, gpu %.0f us; per-token inputs %.0f us]\n",
               eng.t_layerA_rec / eng.n_layerA_rec * 1e6, (unsigned long long) eng.n_layerA_rec,
               eng.t_layerA_attn / eng.n_layerA_attn * 1e6, (unsigned long long) eng.n_layerA_attn,
               eng.t_attn_build / eng.n_layerA_attn * 1e6, eng.t_attn_compute / eng.n_layerA_attn * 1e6,
               eng.n_decode ? eng.t_inputs / eng.n_decode * 1e6 : 0.0);
    printf("prefetch: %.1f%% of the next layer's experts predicted correctly",
           eng.pred_total ? 100.0 * eng.pred_hits / eng.pred_total : 0.0);
    if (eng.pred2_total)
        printf(" (two ahead: %.1f%%)", 100.0 * eng.pred2_hits / eng.pred2_total);
    printf(" | %llu issued, %llu used (%.1f%%), %llu wasted\n",
           (unsigned long long) s.pf_issued, (unsigned long long) s.pf_used,
           s.pf_issued ? 100.0 * s.pf_used / s.pf_issued : 0.0,
           (unsigned long long) s.pf_wasted);
}

void gen_run::print_spec_stats() {
    assert_that(sp.accepted_at.size() == (size_t) engine::MTP_MAX_DRAFTS + 1, "one bucket per possible draft count");
    assert_that(sp.steps > 0 || sp.acc == 0, "accepted drafts come from verify steps");
    if (sp.steps) {
        printf("verify steps by drafts kept:");
        for (size_t j = 0; j < sp.accepted_at.size(); j++) if (sp.accepted_at[j]) printf(" %zu:%llu", j, (unsigned long long) sp.accepted_at[j]);
        printf("  (tokens per step %.2f)\n", (double) (sp.steps + sp.acc) / sp.steps);
    }
    if (sp.steps || sp.single)
        printf("speculative: %llu pair steps, %llu accepted (%.1f%%), %llu single steps, %llu rollbacks (%.3f s), head %.2f s\n",
               (unsigned long long) sp.steps, (unsigned long long) sp.acc, sp.steps ? 100.0 * sp.acc / sp.steps : 0.0,
               (unsigned long long) sp.single, (unsigned long long) eng.n_rollback, eng.t_rollback, eng.t_mtp);
    if (eng.mtp_n || eng.mtp_prompt_n)
        printf("mtp draft: %llu decode drafts scored, %llu accepted (%.1f%%), top-3 %.1f%% | prompt: %llu scored, %.1f%% accepted | %.2f s in the head\n",
               (unsigned long long) eng.mtp_n, (unsigned long long) eng.mtp_acc,
               eng.mtp_n ? 100.0 * eng.mtp_acc / eng.mtp_n : 0.0, eng.mtp_n ? 100.0 * eng.mtp_top3 / eng.mtp_n : 0.0,
               (unsigned long long) eng.mtp_prompt_n, eng.mtp_prompt_n ? 100.0 * eng.mtp_prompt_acc / eng.mtp_prompt_n : 0.0, eng.t_mtp);
    if (eng.pf_gated)
        printf("prefetch gate: %llu predicted candidates not read (margin < %.2f%s)\n",
               (unsigned long long) eng.pf_gated, o.cfg.spec_margin,
               o.cfg.spec_gate_inflight ? ", only after that many recent reads" : "");
}

// Precision by predicted rank, then by confidence margin, then recall by layer.
void gen_run::print_prediction_stats() {
    assert_that(eng.pred_hits_layer.size() == eng.pred_total_layer.size(), "per-layer prediction counters are sized together");
    printf("prediction by rank, precision %%:");
    for (int k = 0; k < (int) QWFN_SPEC_MAX; k++)
        if (eng.rank_total[k]) printf(" r%d %.0f", k + 1, 100.0 * eng.rank_hits[k] / eng.rank_total[k]);
    printf("\n");
    unsigned long long mt = 0; for (int b = 0; b < engine::SPEC_MARGIN_BUCKETS; b++) mt += eng.margin_total[b];
    printf("prediction by margin [edge+) P(correct)%% / share%%:");
    for (int b = 0; b < engine::SPEC_MARGIN_BUCKETS; b++)
        if (eng.margin_total[b])
            printf(" [%.2g) %.0f/%.1f", engine::margin_edge(b), 100.0 * eng.margin_hits[b] / eng.margin_total[b],
                   mt ? 100.0 * eng.margin_total[b] / mt : 0.0);
    printf("\n");
    std::vector<std::pair<double, int>> acc;
    for (size_t l = 0; l < eng.pred_total_layer.size(); l++)
        if (eng.pred_total_layer[l]) acc.push_back({ 100.0 * eng.pred_hits_layer[l] / eng.pred_total_layer[l], (int) l });
    std::sort(acc.begin(), acc.end());
    assert_that(acc.size() <= eng.pred_total_layer.size(), "at most one entry per layer");
    printf("prediction by layer, worst 10:");
    for (size_t i = 0; i < acc.size() && i < 10; i++) printf(" L%d %.0f", acc[i].second, acc[i].first);
    if (acc.size() > 10) printf("  | best: L%d %.0f", acc.back().second, acc.back().first);
    printf("\n");
}

void gen_run::print_io_stats() {
    const auto & s = eng.cache_stats();
    const auto & io = eng.cache_io();
    assert_that(io.which() == io_engine::backend::threads || io.which() == io_engine::backend::uring, "a known io backend");
    assert_that(&s == &eng.cache_stats(), "stats are the engine's live counters");
    printf("io breakdown: submit %.2f s, promote %.2f s, wait %.2f s | %llu bursts, %llu reads, "
           "%.1f reads/burst, %.0f KiB/read\n",
           s.t_submit, s.t_promote, s.t_wait,
           (unsigned long long) s.n_bursts, (unsigned long long) s.n_reads,
           s.n_bursts ? (double) s.n_reads / s.n_bursts : 0.0,
           s.n_reads ? s.bytes_from_disk / (double) s.n_reads / 1024.0 : 0.0);
    printf("             wait-only bandwidth: %.2f GB/s   (device peak measured 7.0 GB/s)\n",
           s.t_wait > 0 ? s.bytes_from_disk / s.t_wait / 1e9 : 0.0);
    printf("             io_engine: backend=%s, O_DIRECT=%d, %llu reads, %.2f GB actually read, "
           "prep %.2f s, io_uring_submit %.2f s\n",
           io.which() == io_engine::backend::threads ? "threads" : "uring",
           (int) io.direct_io(), (unsigned long long) io.stat_reads,
           io.stat_bytes / 1e9, io.stat_t_prep, io.stat_t_submit_syscall);
    printf("engine time: prefill %.2f s / %lld tok, decode %.2f s / %lld tok, expert io %.2f s\n",
           eng.t_prefill, (long long) eng.n_prefill, eng.t_decode, (long long) eng.n_decode, eng.t_io);
    if (eng.n_prefill > 0 && (eng.t_pf_graphA > 0 || eng.t_pf_moe > 0))
        printf("prefill split: dense graphs %.2f s | expert reads (blocking) %.2f s | MoE %.2f s | warm-up %.2f s  (of %.2f s)\n",
               eng.t_pf_graphA, eng.t_pf_read, eng.t_pf_moe, eng.t_warm, eng.t_prefill);
    if (s.warm_admitted || s.warm_promoted)
        printf("cache warm-up from prefill: %llu blocks into RAM, %llu on to VRAM, %.2f s\n",
               (unsigned long long) s.warm_admitted, (unsigned long long) s.warm_promoted, eng.t_warm);
    if (eng.prefill_bytes_read() || eng.prefill_bytes_from_ram())
        printf("streamed sweeps: %.2f GB of experts read, %.2f GB taken from the RAM tier\n",
               eng.prefill_bytes_read() / 1e9, eng.prefill_bytes_from_ram() / 1e9);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: qwfn-gen <shard.gguf> [--prompt id,id,...] [--gen N] [--ctx N]\n"
            "                [--ram GB] [--vram GB] [--batch N] [--threads N] [--cpu] [--no-qsa]\n");
        return 1;
    }
    gen_opts o;
    if (!parse_args(argc, argv, o)) return 1;
    assert_that(!o.prompt.empty(), "parse_args supplies a default prompt");

    model_index mi;
    std::string err;
    if (!mi.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    model_index cold;
    model_index * coldp = nullptr;
    if (!o.cold_path.empty()) {
        if (cold.load(o.cold_path, err)) { coldp = &cold; printf("cold tier: %s\n", o.cold_path.c_str()); }
        else fprintf(stderr, "cold tier unavailable: %s\n", err.c_str());
    }

    engine eng;
    if (!eng.init(&mi, coldp, o.cfg, qwfn::backend_dir(), err)) {
        fprintf(stderr, "engine init: %s\n", err.c_str()); return 1;
    }
    printf("%s\n\n", eng.memory_summary().c_str());

    gen_run r{ eng, o, err };
    if (!r.prefill()) return 1;

    // ---- greedy decode ------------------------------------------------------
    if (!r.decode_begin()) return 1;
    const bool spec = r.use_mtp || o.pair_test || o.rollback_test;
    if (!(spec ? r.decode_speculative() : r.decode_plain())) return 1;
    assert_that(r.hist.size() >= o.prompt.size(), "history starts with the prompt");
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - r.t0).count();
    printf("\n\ndecode: %d tokens in %.2f s  (%.2f tok/s), n_past=%d\n", o.n_gen, dt, o.n_gen / dt, eng.n_past());
    if (const char * sp = getenv("QWFN_SECOND_PROMPT"))
        if (!r.second_prompt(sp)) return 1;
    r.print_run_stats();
    r.print_cache_stats();
    r.print_spec_stats();
    r.print_prediction_stats();
    r.print_io_stats();
    return 0;
}
