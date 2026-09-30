// Session snapshot check: logits after save -> unrelated prompt -> load must match
// logits of a straight continuation (to the GPU's run-to-run noise, measured here too).
//   qwfn-snaptest <shard.gguf> <promptA ids> <promptB ids> <continuation ids> [n]
#include "qwfn_check.h"
#include "qwfn_engine.h"
#include "qwfn_model.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace qwfn;

static std::vector<int32_t> read_ids(const char * p) {
    assert_that(p != nullptr, "id file path is set");
    std::vector<int32_t> v; FILE * f = fopen(p, "rb"); int x;
    if (f) { while (fscanf(f, "%d%*[ ,\n\t\r]", &x) == 1) v.push_back(x); fclose(f); }
    assert_that(f != nullptr || v.empty(), "an unreadable file yields no ids");
    return v;
}

static void compare(const char * what, const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b,
                    int64_t V) {
    assert_that(a.size() == b.size(), "both runs score the same continuation");
    assert_that(a.empty() || (int64_t) a[0].size() == V, "one logit row per vocabulary entry");
    double maxd = 0; int agree = 0;
    for (size_t i = 0; i < a.size(); i++) {
        int aa = 0, bb = 0;
        for (int64_t v = 0; v < V; v++) { maxd = std::max(maxd, (double) std::fabs(a[i][v] - b[i][v])); if (a[i][v] > a[i][aa]) aa = (int) v; if (b[i][v] > b[i][bb]) bb = (int) v; }
        agree += aa == bb;
    }
    printf("%s: %zu positions, max |logit diff| %.4f, argmax agreement %d/%zu\n", what, a.size(), maxd, agree, a.size());
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model A B cont [n]\n", argv[0]); return 2; }
    std::vector<int32_t> A = read_ids(argv[2]), B = read_ids(argv[3]), C = read_ids(argv[4]);
    const int n = argc > 5 ? (int) qwfn::arg_int("n", argv[5], 0, 1 << 24) : 16;
    C.resize(std::min<size_t>(C.size(), n));
    engine_config cfg; cfg.n_ctx = 16384; cfg.n_batch = 4096; cfg.ram_bytes = 12e9; cfg.vram_bytes = 12e9;
    cfg.type_k = cfg.type_v = GGML_TYPE_Q8_0;
    if (const char * m = getenv("SNAP_MTP")) { cfg.mtp_path = m; cfg.rollback_snapshots = true; }
    model_index mi; std::string err;
    if (!mi.load(argv[1], err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    engine eng;
    if (!eng.init(&mi, nullptr, cfg, "", err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const int64_t V = eng.n_vocab();
    assert_that(V > 0, "engine reports a vocabulary");
    auto prefill = [&](const std::vector<int32_t> & p) {
        for (size_t done = 0; done < p.size();) {
            const int32_t take = (int32_t) std::min<size_t>(cfg.n_batch, p.size() - done);
            if (!eng.eval(p.data(), (int32_t) (done + take), take, err)) { fprintf(stderr, "eval: %s\n", err.c_str()); exit(1); }
            done += take;
        }
    };
    auto continue_logits = [&](std::vector<int32_t> hist) {
        std::vector<std::vector<float>> out;
        for (int32_t t : C) {
            hist.push_back(t);
            const float * l = eng.eval(hist.data(), (int32_t) hist.size(), 1, err);
            if (!l) { fprintf(stderr, "eval: %s\n", err.c_str()); exit(1); }
            out.emplace_back(l, l + V);
        }
        assert_that(out.size() == C.size(), "one logit row per continuation token");
        return out;
    };
    eng.reset(); prefill(A); auto ref1 = continue_logits(A);
    eng.reset(); prefill(A); auto ref2 = continue_logits(A);
    compare("straight vs straight (GPU noise)", ref1, ref2, V);
    eng.reset(); prefill(A);
    engine::state_snapshot snap;
    if (!eng.snapshot_save(snap)) { fprintf(stderr, "save failed\n"); return 1; }
    printf("snapshot: %d positions, %.1f MB\n", snap.n_past, snap.bytes() / 1e6);
    eng.reset(); prefill(B); continue_logits(B);   // an unrelated request in between
    if (!eng.snapshot_load(snap)) { fprintf(stderr, "load failed\n"); return 1; }
    auto got = continue_logits(A);
    compare("restored vs straight", got, ref1, V);
    if (getenv("SNAP_MTP")) {   // the draft head must stay usable after a restore and a multi-token batch
        eng.reset(); prefill(A);
        engine::state_snapshot s2; eng.snapshot_save(s2);
        eng.reset(); prefill(B);
        eng.snapshot_load(s2);
        std::vector<int32_t> h = A; h.insert(h.end(), C.begin(), C.begin() + 4);
        if (!eng.eval(h.data(), (int32_t) h.size(), 4, err)) { fprintf(stderr, "eval: %s\n", err.c_str()); return 1; }
        printf("draft head after restore + 4-token batch: %s\n", eng.mtp_ready() ? "ready (OK)" : "NOT ready (drafting off)");
        if (!eng.mtp_ready()) return 1;
    }
    return 0;
}
