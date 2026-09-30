// llama-nll -- the llama.cpp reference for teacher-forced parity: per-token NLL (and top-k
// log-probabilities) of a continuation under llama.cpp's own qwen4exp graph. Routed experts stay
// in host memory (the -ot exps=CPU layout), everything else on the GPU.
//
//   llama-nll MODEL PROMPT_IDS REPLAY_IDS [options]
//     --n N            score the first N continuation tokens (default: all)
//     --kv TYPE        KV cache type, f16 | q8_0 | q4_0 (default q8_0, as the engine runs)
//     --fa on|off      flash attention (default on)
//     --ubatch U       score the continuation U tokens per decode call after the prompt
//                      (default 512: one prefill-shaped batch; 1 = decode-shaped)
//     --dump FILE      per-token lines "i token nll argmax"
//     --topk FILE [K]  per-token top-K log-probabilities "i id:lp id:lp ..." (K default 32)
//     --generate N     instead of scoring, write a greedy continuation of N tokens to REPLAY_IDS
#include "llama.h"
#include "ggml-backend.h"
#include "../src/qwfn_check.h"   // header-only; llama-nll does not link qwfn_core
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using qwfn::assert_that;

static std::vector<llama_token> read_ids(const char * p) {
    assert_that(p != nullptr, "id file path is set");
    std::vector<llama_token> v; FILE * f = fopen(p, "rb"); int x;
    if (f) { while (fscanf(f, "%d%*[ ,\n\t\r]", &x) == 1) v.push_back(x); fclose(f); }
    assert_that(f != nullptr || v.empty(), "an unreadable file yields no ids");
    return v;
}

static ggml_type kv_type(const std::string & s) {
    static const struct { const char * name; ggml_type type; } k_types[] = {
        { "f16", GGML_TYPE_F16 }, { "q8_0", GGML_TYPE_Q8_0 }, { "q4_0", GGML_TYPE_Q4_0 } };
    for (const auto & k : k_types) {
        assert_that(k.name != nullptr, "KV type table entry has a name");
        assert_that(k.type < GGML_TYPE_COUNT, "KV type table entry is a ggml type");
        if (s == k.name) return k.type;
    }
    fprintf(stderr, "unknown --kv %s\n", s.c_str()); exit(2);
}

struct nll_opts {
    const char * model_path = nullptr, * prompt_path = nullptr, * replay_path = nullptr;
    int n_score = -1, ubatch = 512, topk = 32, gen = 0;
    std::string kv = "q8_0", dump_path, topk_path;
    bool fa = true;
};

// Options after the three positional arguments. Returns 0, or the exit code of a usage error.
static int parse_args(int argc, char ** argv, nll_opts & o) {
    assert_that(argc >= 4, "caller checked the positional arguments");
    o.model_path = argv[1]; o.prompt_path = argv[2]; o.replay_path = argv[3];
    for (int i = 4; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
        if (a == "--n") o.n_score = (int) qwfn::arg_int("--n", next().c_str(), -1, 2147483647);
        else if (a == "--kv") o.kv = next();
        else if (a == "--fa") o.fa = next() == "on";
        else if (a == "--ubatch") o.ubatch = (int) qwfn::arg_int("--ubatch", next().c_str(), 1, 512);   // n_batch is 512
        else if (a == "--dump") o.dump_path = next();
        else if (a == "--topk") { o.topk_path = next(); if (i + 1 < argc && argv[i + 1][0] != '-') o.topk = (int) qwfn::arg_int("--topk K", argv[++i], 1, 1 << 20); }
        else if (a == "--generate") o.gen = (int) qwfn::arg_int("--generate", next().c_str(), 0, 1 << 24);
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    assert_that(o.ubatch >= 1, "--ubatch is clamped to at least 1");
    return 0;
}

struct llama_handles {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int nv = 0;
};

// Model with routed experts on the host, then a context for P prompt + C continuation tokens.
static bool load_llama(const nll_opts & o, int P, int C, llama_handles & h) {
    assert_that(o.model_path != nullptr, "model path parsed");
    assert_that(P > 0, "prompt checked non-empty");
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    static llama_model_tensor_buft_override ov[2] = { { "exps", ggml_backend_cpu_buffer_type() }, { nullptr, nullptr } };
    mp.tensor_buft_overrides = ov;
    h.model = llama_model_load_from_file(o.model_path, mp);
    if (!h.model) { fprintf(stderr, "load failed\n"); return false; }
    h.vocab = llama_model_get_vocab(h.model);
    h.nv = llama_vocab_n_tokens(h.vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = P + C + 16; cp.n_batch = 512; cp.n_ubatch = 512; cp.no_perf = false;
    cp.type_k = cp.type_v = kv_type(o.kv);
    cp.flash_attn_type = o.fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    h.ctx = llama_init_from_model(h.model, cp);
    if (!h.ctx) { fprintf(stderr, "context init failed\n"); return false; }
    return true;
}

// Receives the logits run() asks for: greedy mode keeps the argmax, scoring mode accumulates the
// teacher-forced NLL (and writes the optional dump / top-k lines). A struct instead of a callback (rule 9).
struct logits_sink {
    bool greedy = false;
    int nv = 0, P = 0, topk = 0, last = -1;
    const std::vector<llama_token> * seq = nullptr;
    FILE * dump = nullptr, * tkf = nullptr;
    double nll = 0; int cnt = 0, top1 = 0;
    std::vector<int> idx;

    void take(int pos, const float * l) {   // logits at pos predict seq[pos+1]
        assert_that(l != nullptr && nv > 0, "logits for a loaded vocabulary");
        assert_that(seq != nullptr, "sink bound to a sequence");
        if (greedy) { last = (int) (std::max_element(l, l + nv) - l); return; }
        if (pos < P - 1 || pos + 1 >= (int) seq->size()) return;
        const llama_token t = (*seq)[pos + 1];
        float mx = l[0]; int am = 0; for (int v = 1; v < nv; v++) if (l[v] > mx) { mx = l[v]; am = v; }
        double z = 0; for (int v = 0; v < nv; v++) z += std::exp((double) l[v] - mx);
        const double lz = mx + std::log(z), val = -((double) l[t] - lz);
        if (dump) fprintf(dump, "%d %d %.6f %d\n", cnt, t, val, am);
        if (tkf) write_topk(l, lz);
        nll += val; cnt++; top1 += am == t;
    }

    void write_topk(const float * l, double lz) {
        assert_that(topk >= 1 && topk <= nv, "top-k fits the vocabulary");
        assert_that((int) idx.size() == nv, "index scratch sized to the vocabulary");
        for (int v = 0; v < nv; v++) idx[v] = v;
        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
        fprintf(tkf, "%d", cnt);
        for (int k = 0; k < topk; k++) fprintf(tkf, " %d:%.6f", idx[k], (double) l[idx[k]] - lz);
        fprintf(tkf, "\n");
    }
};

// Decode tokens [from, to) of `seq` in calls of `step` tokens; logits for positions >= want_from go to `sink`.
static void run(llama_context * ctx, const std::vector<llama_token> & seq, int from, int to, int step, int want_from,
                logits_sink & sink) {
    assert_that(ctx != nullptr, "context initialised");
    assert_that(from >= 0 && to <= (int) seq.size(), "decode range lies inside the sequence");
    llama_batch b = llama_batch_init(std::max(step, 1), 0, 1);
    for (int s = from; s < to; s += step) {
        const int e = std::min(to, s + step);
        b.n_tokens = e - s;
        for (int i = s; i < e; i++) {
            const int k = i - s;
            b.token[k] = seq[i]; b.pos[k] = i; b.n_seq_id[k] = 1; b.seq_id[k][0] = 0;
            b.logits[k] = i >= want_from;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at %d\n", s); exit(1); }
        for (int i = s; i < e; i++) if (i >= want_from) sink.take(i, llama_get_logits_ith(ctx, i - s));
    }
    llama_batch_free(b);
}

// Greedy continuation of o.gen tokens, decode-shaped, written to the replay path.
static int generate(const nll_opts & o, const llama_handles & h, const std::vector<llama_token> & prompt) {
    assert_that(o.gen != 0, "generate mode");
    assert_that(h.nv > 0, "vocabulary loaded");
    const int P = (int) prompt.size();
    std::vector<llama_token> seq = prompt, out;
    logits_sink g; g.greedy = true; g.nv = h.nv; g.seq = &seq;
    run(h.ctx, seq, 0, P, 512, P - 1, g);
    while ((int) out.size() < o.gen) {
        out.push_back(g.last); seq.push_back(g.last);
        if (llama_vocab_is_eog(h.vocab, g.last)) break;
        run(h.ctx, seq, (int) seq.size() - 1, (int) seq.size(), 1, (int) seq.size() - 1, g);
    }
    FILE * f = fopen(o.replay_path, "w"); for (llama_token t : out) fprintf(f, "%d\n", t); fclose(f);
    printf("llama.cpp greedy continuation: %zu tokens -> %s (kv %s, fa %s)\n", out.size(), o.replay_path, o.kv.c_str(), o.fa ? "on" : "off");
    llama_free(h.ctx); llama_model_free(h.model); return 0;
}

// Teacher-forced NLL of `replay` after `prompt`, with the optional per-token dump and top-k files.
static void score_replay(const nll_opts & o, const llama_handles & h, const std::vector<llama_token> & prompt,
                         const std::vector<llama_token> & replay) {
    assert_that(!prompt.empty() && !replay.empty(), "scoring needs a prompt and a continuation");
    assert_that(h.ctx != nullptr && h.nv > 0, "context and vocabulary loaded");
    const int P = (int) prompt.size();
    std::vector<llama_token> seq = prompt; seq.insert(seq.end(), replay.begin(), replay.end());
    logits_sink sc; sc.nv = h.nv; sc.P = P; sc.topk = o.topk; sc.seq = &seq;
    if (!o.topk_path.empty() && o.topk > h.nv) { fprintf(stderr, "--topk K %d exceeds the vocabulary (%d)\n", o.topk, h.nv); exit(2); }
    sc.dump = o.dump_path.empty() ? nullptr : fopen(o.dump_path.c_str(), "w");
    sc.tkf  = o.topk_path.empty() ? nullptr : fopen(o.topk_path.c_str(), "w");
    sc.idx.resize(h.nv);
    run(h.ctx, seq, 0, P, 512, P - 1, sc);                                  // the prompt: prefill-shaped
    run(h.ctx, seq, P, (int) seq.size() - 1, o.ubatch, P, sc);               // the continuation, ubatch tokens per call
    if (sc.dump) fclose(sc.dump);
    if (sc.tkf) fclose(sc.tkf);
    const double nll = sc.nll; const int cnt = sc.cnt, top1 = sc.top1;
    printf("llama.cpp replay NLL: %.4f per token over %d tokens (greedy agreement %.1f%%) [kv %s, fa %s, ubatch %d]\n",
           nll / cnt, cnt, 100.0 * top1 / cnt, o.kv.c_str(), o.fa ? "on" : "off", o.ubatch);
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s MODEL PROMPT_IDS REPLAY_IDS [--n N] [--kv T] [--fa on|off] [--ubatch U] [--dump F] [--topk F [K]] [--generate N]\n", argv[0]); return 2; }
    nll_opts o;
    if (const int rc = parse_args(argc, argv, o)) return rc;
    std::vector<llama_token> prompt = read_ids(o.prompt_path);
    std::vector<llama_token> replay = o.gen ? std::vector<llama_token>() : read_ids(o.replay_path);
    if (prompt.empty() || (!o.gen && replay.empty())) { fprintf(stderr, "empty prompt or continuation\n"); return 2; }
    if (!o.gen && o.n_score > 0 && (size_t) o.n_score < replay.size()) replay.resize(o.n_score);
    assert_that(!prompt.empty(), "prompt checked non-empty");
    assert_that(o.gen || !replay.empty(), "scoring has a continuation");

    const int P = (int) prompt.size(), C = o.gen ? o.gen : (int) replay.size();
    llama_handles h;
    if (!load_llama(o, P, C, h)) return 1;
    if (o.gen) return generate(o, h, prompt);
    score_replay(o, h, prompt, replay);
    llama_free(h.ctx); llama_model_free(h.model);
}
