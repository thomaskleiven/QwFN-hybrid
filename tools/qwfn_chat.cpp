// qwfn-chat -- type text, get text. Interactive or one-shot.
//
// The engine's boundary is token ids; this is the only thing above it. The chat
// framing follows the template embedded in the GGUF (tokenizer.chat_template,
// 9,993 chars) rather than a guess at it. Two details from that template matter
// and are easy to get wrong:
//
//   * The generation prompt pre-fills the model INTO the think block:
//     "<|im_start|>assistant\n<think>\n". Thinking is the default; turning it
//     off means emitting a pre-closed "<think>\n\n</think>\n\n" instead, not
//     omitting the block.
//   * Reasoning effort is not a sampler knob -- it is a system message the
//     template injects. xhigh (the template's own default) and low have text;
//     medium deliberately has none.
//
// History is kept as tokens and only ever appended to, never re-rendered. What
// the model generates after our "<think>\n" prefix is exactly what the template
// would have rendered for a completed assistant turn, so appending its output
// verbatim reproduces the template for the next turn. That is also why context
// survives across turns for free: the engine tracks n_past, so turn N prefills
// only its own new tokens, not the conversation so far.

#include "qwfn_check.h"
#include "qwfn_engine.h"
#include "qwfn_model.h"
#include "qwfn_vocab.h"
#include "qwfn_vision.h"
#include "qwfn_template.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace qwfn;

using clk = std::chrono::steady_clock;
static double since(clk::time_point t) {
    return std::chrono::duration<double>(clk::now() - t).count();
}


// ---- incremental UTF-8 ------------------------------------------------------
// A multi-byte character can straddle two tokens, so bytes are buffered and only
// complete sequences are printed. Without this, streaming output mangles any
// non-ASCII text at token boundaries -- which any real HTML or source file hits.
struct utf8_stream {
    std::string pending;

    std::string feed(const std::string & bytes) {
        pending += bytes;
        size_t cut = pending.size();
        for (size_t back = 0; back < 4 && back < pending.size(); back++) {
            const size_t i = pending.size() - 1 - back;
            const unsigned char c = (unsigned char) pending[i];
            if ((c & 0xC0) == 0x80) continue;                 // continuation
            const size_t need = (c & 0x80) == 0x00 ? 1
                              : (c & 0xE0) == 0xC0 ? 2
                              : (c & 0xF0) == 0xE0 ? 3
                              : (c & 0xF8) == 0xF0 ? 4 : 1;
            if (i + need > pending.size()) cut = i;            // incomplete tail
            break;
        }
        assert_that(cut <= pending.size(), "the cut lies inside the buffer");
        std::string out = pending.substr(0, cut);
        pending.erase(0, cut);
        assert_that(pending.size() <= 3, "only an incomplete sequence stays pending");
        return out;
    }
};

struct sampler {
    float temp = 0.0f, top_p = 0.95f;
    int   top_k = 40;
    std::mt19937 rng{0xC0FFEEu};

    int pick(const float * lg, int64_t n) {
        assert_that(lg != nullptr, "logits present");
        assert_that(n > 0, "a vocabulary to pick from");
        if (temp <= 0.0f) {                                    // greedy
            int best = 0;
            for (int64_t v = 1; v < n; v++) if (lg[v] > lg[best]) best = (int) v;
            return best;
        }
        const int k = (int) std::min<int64_t>(top_k > 0 ? top_k : n, n);
        std::vector<int> idx(n);
        for (int64_t v = 0; v < n; v++) idx[v] = (int) v;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return lg[a] > lg[b]; });
        idx.resize(k);

        const float mx = lg[idx[0]];
        std::vector<float> p(k);
        double sum = 0;
        for (int i = 0; i < k; i++) { p[i] = std::exp((lg[idx[i]] - mx) / temp); sum += p[i]; }
        for (int i = 0; i < k; i++) p[i] = (float) (p[i] / sum);

        double cum = 0;                                        // nucleus
        int keep = k;
        for (int i = 0; i < k; i++) { cum += p[i]; if (cum >= top_p) { keep = i + 1; break; } }

        std::uniform_real_distribution<double> U(0.0, cum);
        double r = U(rng), acc = 0;
        for (int i = 0; i < keep; i++) { acc += p[i]; if (r <= acc) return idx[i]; }
        return idx[0];
    }
};

// ---- attachments ------------------------------------------------------------
struct attachment { std::string path, body; size_t n_tok = 0; };

// An image already run through the vision tower: the embeddings are held until
// the next message, then spliced over its <|image_pad|> placeholders.
struct pending_image {
    std::string        path;
    std::vector<float> emb;
    int n_tok = 0, gw = 0, gh = 0;
};

// Refuse binaries rather than feed the model megabytes of mojibake.
static bool looks_binary(const std::string & s) {
    const size_t n = std::min<size_t>(s.size(), 8192);
    assert_that(n <= s.size() && n <= 8192, "scan window inside the text and the cap");
    for (size_t i = 0; i < n; i++) if (s[i] == '\0') return true;
    assert_that(n == 0 || s[n - 1] != '\0', "no NUL in the scanned window");
    return false;
}

static std::string trim(const std::string & s);

// Paths as a person actually produces them at a console: "~/x.html", a path
// dragged into the terminal (which arrives quoted, or with spaces backslash-
// escaped), or a bare relative path. fopen() understands none of the first three.
static std::string expand_path(std::string p) {
    const size_t n_in = p.size();
    p = trim(p);
    assert_that(p.size() <= n_in, "trimming never grows the path");
    if (p.size() >= 2 && ((p.front() == '"'  && p.back() == '"') ||
                          (p.front() == '\'' && p.back() == '\''))) {
        p = p.substr(1, p.size() - 2);          // dropped-in quoted path
    } else {
        std::string un;                          // "foo\ bar.html" -> "foo bar.html"
        for (size_t i = 0; i < p.size(); i++) {
            if (p[i] == '\\' && i + 1 < p.size()) { un += p[++i]; continue; }
            un += p[i];
        }
        p = un;
    }
    assert_that(p.size() <= n_in, "unquoting and unescaping never grow the path");
    if (p == "~" || p.rfind("~/", 0) == 0) {
        const char * home = getenv("HOME");
        if (home) p = std::string(home) + p.substr(1);
    }
    return p;
}

static bool read_file(const std::string & path, std::string & out, std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    assert_that(!path.empty(), "an empty path cannot be opened");
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); err = "cannot size " + path; return false; }
    out.resize((size_t) sz);
    const size_t got = sz ? fread(out.data(), 1, (size_t) sz, f) : 0;
    fclose(f);
    assert_that(got <= (size_t) sz, "fread reads at most the requested size");
    out.resize(got);
    if (looks_binary(out)) { err = path + " looks binary; only text files can be attached"; return false; }
    return true;
}

static std::string trim(const std::string & s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    const size_t b = s.find_last_not_of(" \t\r\n");
    assert_that(b != std::string::npos, "a non-blank character exists");
    assert_that(b >= a, "the last non-blank is not before the first");
    return s.substr(a, b - a + 1);
}

static const char k_usage[] =
    "usage: qwfn-chat <shard.gguf> [options]\n"
    "\n"
    "  -p, --prompt TEXT   one-shot: answer TEXT and exit (default: interactive)\n"
    "  -f, --file PATH     attach a text file to the first message (repeatable)\n"
    "  -i, --image PATH    attach an image (needs --mmproj) (repeatable)\n"
    "      --mmproj PATH   vision projector gguf; enables images (encoded on the CPU, --threads)\n"
    "      --system TEXT   system message\n"
    "      --think LEVEL   xhigh (default) | medium | low | off\n"
    "      --hide-think    generate reasoning but do not print it\n"
    "      --temp F        0 = greedy (default)   --top-p F   --top-k N   --seed N\n"
    "      --max N         max tokens per reply (default 512; 0 = fill the context)\n"
    "      --ignore-eos    keep generating past end-of-turn (long-context testing)\n"
    "      --ctx N         context (default 8192)      --batch N   (default 1024)\n"
    "      --ram GB        --vram GB   --threads N   --cpu   --kv f16|q8_0|q4_0 (default q8_0)\n"
    "      --ubatch-kv M   cap on n_kv*ubatch in millions for the prefill graph (default 96)\n"
    "      --ram-frac F    MemAvailable share the RAM tier may take (default 0.60)\n"
    "      --spec-ahead N  predict 1 or 2 layers ahead (default 2)\n"
    "      --no-prefill-overlap   single prefill staging buffer, saves ~1.8 GB RAM\n"
    "\n"
    "Interactive commands:\n"
    "  /file PATH   attach a text file to the next message (rest of line = path)\n"
    "  /image PATH  attach an image to the next message (needs --mmproj)\n"
    "  /files       list what is attached      /drop   discard attachments\n"
    "  /stats       session totals so far      /reset  clear the conversation\n"
    "  /quit        exit\n";

struct chat_opts {
    std::string one_shot, system_msg, effort = "xhigh", mmproj_path;
    std::vector<std::string> startup_files, startup_images;
    bool hide_think = false, interactive = true, ignore_eos = false;
    int  max_gen = 512;                    // 0 = until the context is full
    sampler smp;
    engine_config cfg;
};

// Chat and sampling flags. True if argv[i] was one (i then points at its last argument).
static bool parse_chat_flag(const std::string & a, int argc, char ** argv, int & i, chat_opts & o) {
    assert_that(i >= 2 && i < argc, "flag index lies inside argv");
    assert_that(a == argv[i], "flag text is the current argument");
    auto next = [&]() { return argv[++i]; };
    if ((a == "-p" || a == "--prompt") && i + 1 < argc) { o.one_shot = next(); o.interactive = false; return true; }
    if ((a == "-f" || a == "--file")   && i + 1 < argc) { o.startup_files.push_back(next()); return true; }
    if (a == "--system" && i + 1 < argc) { o.system_msg = next(); return true; }
    if (a == "--mmproj" && i + 1 < argc) { o.mmproj_path = next(); return true; }
    if ((a == "-i" || a == "--image") && i + 1 < argc) { o.startup_images.push_back(next()); return true; }
    if (a == "--think"  && i + 1 < argc) { o.effort = next(); return true; }
    if (a == "--hide-think") { o.hide_think = true; return true; }
    if (a == "--ignore-eos") { o.ignore_eos = true; return true; }
    if (a == "--temp"   && i + 1 < argc) { o.smp.temp  = (float) qwfn::arg_float("--temp", next(), 0, 100); return true; }
    if (a == "--top-p"  && i + 1 < argc) { o.smp.top_p = (float) qwfn::arg_float("--top-p", next(), 0, 1); return true; }
    if (a == "--top-k"  && i + 1 < argc) { o.smp.top_k = (int) qwfn::arg_int("--top-k", next(), 0, 1 << 24); return true; }
    if (a == "--seed"   && i + 1 < argc) { o.smp.rng.seed((unsigned) qwfn::arg_int("--seed", next(), -2147483648LL, 4294967295LL)); return true; }
    if (a == "--max"    && i + 1 < argc) { o.max_gen = (int) qwfn::arg_int("--max", next(), -1, 1 << 30); return true; }
    return false;
}

// Engine flags. True if argv[i] was one (i then points at its last argument).
static bool parse_engine_flag(const std::string & a, int argc, char ** argv, int & i, engine_config & cfg) {
    assert_that(i >= 2 && i < argc, "flag index lies inside argv");
    assert_that(a == argv[i], "flag text is the current argument");
    auto next = [&]() { return argv[++i]; };
    if (a == "--ctx"    && i + 1 < argc) { cfg.n_ctx = (uint32_t) qwfn::arg_int("--ctx", next(), 1, 2147483647); return true; }
    if (a == "--batch"  && i + 1 < argc) { cfg.n_batch = (uint32_t) qwfn::arg_int("--batch", next(), 1, 1 << 20); return true; }
    if (a == "--ubatch-kv" && i + 1 < argc) { cfg.ubatch_kv_product = (uint64_t)(qwfn::arg_float("--ubatch-kv", next(), 0, 1e7) * 1e6); return true; }
    if (a == "--indexer-top-k" && i + 1 < argc) { cfg.indexer_top_k = (uint32_t) qwfn::arg_int("--indexer-top-k", next(), 0, 1 << 20); return true; }
    if (a == "--ram"    && i + 1 < argc) { cfg.ram_bytes = (size_t)(qwfn::arg_float("--ram", next(), 0, 1e6) * 1e9); return true; }
    if (a == "--vram"   && i + 1 < argc) { cfg.vram_bytes = (size_t)(qwfn::arg_float("--vram", next(), 0, 1e6) * 1e9); return true; }
    if (a == "--threads"&& i + 1 < argc) { cfg.n_threads = (int) qwfn::arg_int("--threads", next(), 1, 1024); return true; }
    if (a == "--ram-frac" && i + 1 < argc) { cfg.ram_frac = qwfn::arg_float("--ram-frac", next(), 0, 1); return true; }
    if (a == "--spec-ahead" && i + 1 < argc) { cfg.speculate_ahead = (uint32_t) qwfn::arg_int("--spec-ahead", next(), 0, 16); return true; }
    if (a == "--no-prefill-overlap") { cfg.prefill_overlap = false; return true; }
    if (a == "--cpu")   { cfg.use_gpu = false; return true; }
    if (a == "--no-qsa") { cfg.use_qsa = false; return true; }
    if (a == "--prefill-cpu") { cfg.prefill_on_gpu = false; return true; }
    if (a == "--prefill-gpu") { cfg.prefill_on_gpu = true; return true; }  // FAST BUT BROKEN >T~200
    if (a == "--skip-miss") { cfg.skip_miss = true; return true; }
    if (a == "--reserve" && i + 1 < argc) { cfg.vram_reserve = (size_t) qwfn::arg_float("--reserve", next(), 0, 1e7) * (1ull << 20); return true; }
    if (a == "--spec-depth" && i + 1 < argc) { cfg.speculate_depth = (uint32_t) qwfn::arg_int("--spec-depth", argv[++i], 0, QWFN_SPEC_MAX);
        if (cfg.speculate_depth == 0) { cfg.speculate = false; }
        return true; }
    if (a == "--spec-depth2" && i + 1 < argc) { cfg.speculate_depth2 = (uint32_t) qwfn::arg_int("--spec-depth2", argv[++i], 0, QWFN_SPEC_MAX); return true; }
    if (a == "--spec-margin" && i + 1 < argc) { cfg.spec_margin = (float) qwfn::arg_float("--spec-margin", next(), 0, 1000); return true; }
    if (a == "--spec-gate-inflight" && i + 1 < argc) { cfg.spec_gate_inflight = (uint32_t) qwfn::arg_int("--spec-gate-inflight", next(), 0, 1 << 20); return true; }
    if (a == "--spec-block") { cfg.spec_block = true; return true; }
    if (a == "--spec-block-layers" && i + 1 < argc) { cfg.spec_block = true; cfg.spec_block_layers = next(); return true; }
    if (a == "--state-host" && i + 1 < argc) {   // none | idx | kv | kv,idx
        std::string v = next();
        cfg.idx_host = v.find("idx") != std::string::npos;
        cfg.kv_host  = v.find("kv")  != std::string::npos;
        return true;
    }
    if (a == "--kv" && i + 1 < argc) {
        std::string v = next();
        cfg.type_k = cfg.type_v = (v == "q8_0") ? GGML_TYPE_Q8_0 :
                                  (v == "q4_0") ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
        return true;
    }
    return false;
}

// False on an unknown option (already reported).
static bool parse_args(int argc, char ** argv, chat_opts & o) {
    assert_that(argc >= 2, "caller checked the model argument");
    engine_config & cfg = o.cfg;
    // vram defaults high on purpose: the tier self-tunes down to whatever the
    // device can spare (state and dense core are allocated first), and without
    // it every routed expert computes on the CPU at 3.2x the cost. --vram 0
    // still disables it. --batch 1024: prefill throughput scales linearly with
    // the ubatch and the input arena is capped by ubatch_kv_product anyway.
    // --batch 4096: the prefill streams each layer's experts once per batch,
    // so a bigger batch is proportionally fewer 53 GB sweeps per prompt; the
    // compute runs in 2048-token chunks regardless. Costs ~0.7 GB of VRAM in
    // work buffers against 1024. Measured: 32K tokens in 101 s at 4096.
    cfg.n_ctx = 8192; cfg.n_batch = 4096; cfg.ram_bytes = 8e9; cfg.vram_bytes = 12e9;
    // q8_0 KV by default: the intended use is 128K+ context, where an f16 cache
    // costs 1.5 GB of expert slots and measured slower (KV type comparison on the reference machine).
    cfg.type_k = cfg.type_v = GGML_TYPE_Q8_0;

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (parse_chat_flag(a, argc, argv, i, o) || parse_engine_flag(a, argc, argv, i, cfg)) continue;
        fprintf(stderr, "unknown option: %s\n", a.c_str());
        return false;
    }
    assert_that(o.interactive || argc >= 4, "one-shot mode came from a -p TEXT pair");
    return true;
}

// The decode of one reply: what was produced and how it ended.
struct reply_state {
    int n = 0;
    bool hit_wall = false, printed_any = false, in_think = false;
    utf8_stream us;
    clk::time_point td;
    double last_status = 0, t_dec = 0;
};

// A conversation: tokenizer, model, engine and the token history, one turn at a time.
struct chat_session {
    chat_opts & o;
    const bool thinking;
    const bool color;
    std::string err;
    qwfn::vocab vb;
    model_index mi;
    engine eng;
    qwfn::vision_encoder vis;
    int32_t tok_image_pad = -1;
    std::vector<int32_t> hist;
    int32_t fed = 0;                       // how much of hist the engine has seen
    std::vector<attachment> pending;       // attached, not yet sent
    std::vector<pending_image> images;     // encoded, not yet sent
    // ---- session totals -----------------------------------------------------
    int64_t s_pre_tok = 0, s_dec_tok = 0;
    double  s_pre_t = 0,   s_dec_t = 0;
    int     s_turns = 0;

    explicit chat_session(chat_opts & opts)
        : o(opts), thinking(opts.effort != "off"), color(isatty(fileno(stdout))) {}

    void ansi(const char * seq, bool flush = false) const;
    void dim()        const { ansi("\033[2m"); }
    void undim()      const { ansi("\033[0m"); }
    // Erase the live status line. Without the erase, a shorter update leaves the
    // tail of the previous one on screen.
    void clear_line() const { ansi("\r\033[K", true); }

    bool init(const char * model_path);
    void attach_image(const std::string & raw);
    void attach(const std::string & raw);
    void reset_conversation();
    std::vector<int32_t> turn_tokens(const std::string & user,
                                     std::vector<std::pair<int32_t, const pending_image *>> & splices);
    bool prefill_turn(const float * & lg, int32_t & pre_total, double & t_pre);
    void show_piece(const std::string & piece, reply_state & r);
    void decode_reply(const float * lg, int budget, reply_state & r);
    void finish_turn(reply_state & r, int budget, int32_t pre_total, double t_pre);
    bool turn(const std::string & user);
    void session_summary();
    int  command(const std::string & cmd);
    int  interactive_loop();
};

// Terminal escape sequences, only when stdout is a terminal.
void chat_session::ansi(const char * seq, bool flush) const {
    assert_that(seq != nullptr, "an escape sequence");
    assert_that(seq[0] == '\033' || seq[0] == '\r', "starts like an escape sequence");
    if (!color) return;
    printf("%s", seq);
    if (flush) fflush(stdout);
}

bool chat_session::init(const char * model_path) {
    assert_that(model_path != nullptr, "model path from argv");
    assert_that(hist.empty() && tok_image_pad < 0, "a fresh session");
    fprintf(stderr, "loading tokenizer...\n");
    if (!vb.load(model_path, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return false; }

    if (!mi.load(model_path, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return false; }

    if (!eng.init(&mi, nullptr, o.cfg, qwfn::backend_dir(), err)) {
        fprintf(stderr, "engine init: %s\n", err.c_str()); return false;
    }
    fprintf(stderr, "%s\n", eng.memory_summary().c_str());

    if (!o.mmproj_path.empty()) {
        // On the CPU backend, as the server runs it: no VRAM for the projector.
        ggml_backend_t vis_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!vis_backend || !vis.load(o.mmproj_path, vis_backend, ggml_backend_get_default_buffer_type(vis_backend), err)) {
            fprintf(stderr, "vision: %s\n", vis_backend ? err.c_str() : "no CPU backend"); return false;
        }
        vis.set_n_threads(o.cfg.n_threads);
        const auto ip = vb.encode("<|image_pad|>", false, true);
        if (ip.size() != 1) { fprintf(stderr, "vision: <|image_pad|> is not a single token\n"); return false; }
        tok_image_pad = ip[0];
    }
    return true;
}

void chat_session::attach_image(const std::string & raw) {
    if (!vis.loaded()) { printf("  images need --mmproj\n"); return; }
    assert_that(tok_image_pad >= 0, "a loaded projector has its pad token");
    const std::string path = expand_path(raw);
    qwfn::image_u8 img;
    std::string e;
    if (!img.load(path, e)) { printf("  %s\n", e.c_str()); return; }
    pending_image pi; pi.path = path;
    const auto t0 = clk::now();
    // The projector's weights are staged onto the device for the encode; the
    // tier's dynamic buffer is where they go, as for the server.
    if (vis.weights_on_host()) eng.vram_lend_begin();
    if (!vis.encode(img, pi.emb, pi.n_tok, pi.gw, pi.gh, e)) { eng.vram_lend_end(); printf("  %s\n", e.c_str()); return; }
    printf("  attached %s  (%dx%d -> %dx%d grid, %d image tokens, %.2f s)\n",
           path.c_str(), img.nx, img.ny, pi.gw, pi.gh, pi.n_tok, since(t0));
    const size_t n0 = images.size();
    images.push_back(std::move(pi));
    assert_that(images.size() == n0 + 1, "the image is queued");
}

void chat_session::attach(const std::string & raw) {
    const std::string path = expand_path(raw);
    std::string body, e;
    if (!read_file(path, body, e)) { printf("  %s\n", e.c_str()); return; }
    assert_that(!looks_binary(body), "read_file refuses binaries");
    attachment at{path, body, 0};
    at.n_tok = vb.encode(body, false, false).size();
    const int32_t room = (int32_t) o.cfg.n_ctx - (int32_t) hist.size();
    printf("  attached %s  (%zu bytes, ~%zu tokens", path.c_str(), body.size(), at.n_tok);
    if ((int32_t) at.n_tok >= room)
        printf(") -- WARNING: bigger than the %d tokens of context left\n", room);
    else
        printf(", %.0f%% of remaining context)\n", 100.0 * at.n_tok / (room > 0 ? room : 1));
    const size_t n0 = pending.size();
    pending.push_back(std::move(at));
    assert_that(pending.size() == n0 + 1, "the file is queued");
}

void chat_session::reset_conversation() {
    eng.reset();
    hist.clear();
    fed = 0;
    // The system block, built exactly as the template does.
    const std::string pre = build_system_block(o.effort, o.system_msg);
    if (!pre.empty()) {
        auto t = vb.encode(pre, false, true);
        hist.insert(hist.end(), t.begin(), t.end());
    }
    assert_that(fed == 0, "the engine has seen nothing yet");
    assert_that(eng.n_past() == 0, "a reset engine holds no tokens");
}

// The user turn's tokens: attachments fenced in front of the message, images as pad runs.
std::vector<int32_t> chat_session::turn_tokens(const std::string & user,
                                               std::vector<std::pair<int32_t, const pending_image *>> & splices) {
    assert_that(splices.empty(), "caller passes an empty splice list");
    // Attachments ride in front of the message, fenced so the model can tell
    // file content from instructions.
    std::string body;
    for (const auto & a : pending) {
        body += "<file path=\"" + a.path + "\">\n" + a.body;
        if (!a.body.empty() && a.body.back() != '\n') body += "\n";
        body += "</file>\n\n";
    }
    pending.clear();
    body += user;

    // Framing and content are tokenized SEPARATELY. parse_special scans the
    // whole string, so encoding them together lets a file that merely
    // contains the literal text "<|im_end|>" turn into a real control token
    // and forge a turn boundary mid-attachment. (This file contains those
    // literals, so `-f tools/qwfn_chat.cpp` used to break itself.) Content
    // is encoded with parse_special=false and can only ever be plain text.
    const std::string tt = qwfn::turn_tail(thinking);

    std::vector<int32_t> t;
    auto app = [&](const std::vector<int32_t> & v) { t.insert(t.end(), v.begin(), v.end()); };
    app(vb.encode("<|im_start|>user\n", false, true));

    // Images go in ahead of the text as <|vision_start|> <|image_pad|>xN
    // <|vision_end|>. The pads are real tokens so the sequence and the PLE
    // window stay well formed; only their embeddings get replaced.
    for (const auto & pi : images) {
        app(vb.encode("<|vision_start|>", false, true));
        splices.emplace_back((int32_t) t.size(), &pi);
        t.insert(t.end(), (size_t) pi.n_tok, tok_image_pad);
        app(vb.encode("<|vision_end|>", false, true));
    }
    if (!body.empty()) app(vb.encode(body, false, /*parse_special=*/false));
    app(vb.encode(tt, false, true));
    assert_that(splices.size() == images.size(), "one splice per image");
    return t;
}

// ---- prefill, with a live rate --------------------------------------
bool chat_session::prefill_turn(const float * & lg, int32_t & pre_total, double & t_pre) {
    assert_that(fed < (int32_t) hist.size(), "the turn left tokens to prefill");
    assert_that(lg == nullptr, "no logits before the prefill");
    const auto tp = clk::now();
    const int32_t pre_start = fed;
    pre_total = (int32_t) hist.size() - pre_start;
    while (fed < (int32_t) hist.size()) {
        const int32_t take = std::min<int32_t>(o.cfg.n_batch, (int32_t) hist.size() - fed);
        lg = eng.eval(hist.data(), fed + take, take, err);
        if (!lg) { fprintf(stderr, "\nprefill failed: %s\n", err.c_str()); return false; }
        fed += take;
        // Only worth showing when there is more than one ubatch to do --
        // otherwise it flashes once and is gone.
        if (color && pre_total > (int32_t) o.cfg.n_batch) {
            const double el = since(tp);
            printf("\r\033[2m[prefill %d/%d tok  %.1f tok/s]\033[0m\033[K",
                   fed - pre_start, pre_total, (fed - pre_start) / (el > 0 ? el : 1e-9));
            fflush(stdout);
        }
    }
    t_pre = since(tp);
    clear_line();
    s_pre_tok += pre_total; s_pre_t += t_pre;
    return true;
}

// Print one sampled piece, or track the end of the reasoning block.
void chat_session::show_piece(const std::string & piece, reply_state & r) {
    assert_that(r.n >= 0, "token count");
    assert_that(!r.in_think || thinking, "reasoning only when thinking is on");
    const bool suppressed = r.in_think && o.hide_think;
    if (r.in_think && piece.find("</think>") != std::string::npos) {
        r.in_think = false;
        if (o.hide_think) clear_line();
        else { undim(); printf("\n\n"); }
        r.us.pending.clear();
    } else if (!suppressed) {
        const std::string out = r.us.feed(piece);
        if (!out.empty()) { printf("%s", out.c_str()); fflush(stdout); r.printed_any = true; }
    } else if (color) {
        // Nothing is being printed, so the line is free for a live rate.
        // Hidden reasoning runs for minutes; without this it looks hung.
        const double el = since(r.td);
        if (el - r.last_status > 0.2) {
            r.last_status = el;
            printf("\r\033[2m[reasoning %d tok  %.1f tok/s]\033[0m\033[K",
                   r.n, r.n / (el > 0 ? el : 1e-9));
            fflush(stdout);
        }
    }
}

// ---- decode ---------------------------------------------------------
void chat_session::decode_reply(const float * lg, int budget, reply_state & r) {
    assert_that(lg != nullptr, "prefill left logits");
    assert_that(budget >= 0, "the budget is clamped at zero");
    r.td = clk::now();
    if (r.in_think && !o.hide_think) { dim(); printf("[thinking] "); }
    fflush(stdout);

    for (; r.n < budget; r.n++) {
        const int tok = o.smp.pick(lg, eng.n_vocab());
        if (vb.is_eog(tok)) {
            hist.push_back(tok); r.n++;   // not evaluated yet; fed re-syncs below
            // Ignoring end-of-turn is how a long-context run actually reaches
            // the context wall; otherwise the model stops after a few hundred
            // tokens and nothing is exercised.
            if (!o.ignore_eos) break;
            if ((int32_t) hist.size() + 1 > (int32_t) o.cfg.n_ctx) { r.hit_wall = true; break; }
            lg = eng.eval(hist.data(), (int32_t) hist.size(), 1, err);
            if (!lg) { r.hit_wall = true; break; }
            fed = (int32_t) hist.size();
            continue;
        }

        const std::string piece = vb.piece(tok, false);
        hist.push_back(tok);
        show_piece(piece, r);

        if ((int32_t) hist.size() + 1 > (int32_t) o.cfg.n_ctx) { r.hit_wall = true; r.n++; break; }
        lg = eng.eval(hist.data(), (int32_t) hist.size(), 1, err);
        if (!lg) { r.hit_wall = true; r.n++; break; }
        fed = (int32_t) hist.size();
    }
    r.t_dec = since(r.td);
    if (r.in_think && o.hide_think) clear_line();
    const std::string tail = r.us.feed("");
    if (!tail.empty()) printf("%s", tail.c_str());
    s_dec_tok += r.n; s_dec_t += r.t_dec; s_turns++;
}

void chat_session::finish_turn(reply_state & r, int budget, int32_t pre_total, double t_pre) {
    assert_that(s_turns > 0, "the decode counted this turn");
    // Close the turn so the next one continues the template correctly.
    //
    // The subtle case is running out of budget while still inside <think>.
    // Closing that with a bare <|im_end|> leaves an unterminated think block
    // in the history; the next turn then sees a malformed assistant turn and
    // reasons forever without ever emitting an answer.
    std::string closing;
    if (r.in_think) closing += "\n</think>\n\n";
    if (hist.empty() || !vb.is_eog(hist.back())) closing += "<|im_end|>\n";
    else                                         closing += "\n";
    auto e = vb.encode(closing, false, true);
    hist.insert(hist.end(), e.begin(), e.end());

    // Re-sync from the engine rather than computing an index into hist.
    // The sampled end-of-turn token is appended but never evaluated, so any
    // hand-computed offset here is one too high; the next prefill then skips
    // <|im_end|> entirely and the model sees an unterminated assistant turn,
    // drifting one token further every turn. n_past is the only authority on
    // what the KV actually holds.
    fed = eng.n_past();

    const bool wall = r.hit_wall || (r.n >= budget && (o.max_gen <= 0 || budget < o.max_gen));
    const bool truncated = !wall && o.max_gen > 0 && r.n >= o.max_gen;
    assert_that(!(wall && truncated), "a reply ends one way");

    undim(); printf("\n"); dim();
    if (wall)
        printf("[context full at %d tokens -- stopped cleanly; /reset to continue]\n", eng.n_past());
    else if (truncated)
        printf(r.in_think
               ? "[cut off at --max %d while still reasoning -- raise --max, or use --think low/off]\n"
               : "[cut off at --max %d]\n", o.max_gen);
    else if (o.hide_think && !r.printed_any)
        printf("[the whole reply was reasoning; nothing to show with --hide-think]\n");

    printf("[prefill %d tok %.1f tok/s | generated %d tok in %.1f s (%.1f tok/s) | ctx %d/%d]",
           pre_total, pre_total / (t_pre > 0 ? t_pre : 1e-9),
           r.n, r.t_dec, r.n / (r.t_dec > 0 ? r.t_dec : 1e-9),
           eng.n_past(), (int) o.cfg.n_ctx);
    undim(); printf("\n");
}

// One user turn -> streamed reply. Returns false on a fatal engine error.
bool chat_session::turn(const std::string & user) {
    assert_that(images.empty() || vis.loaded(), "images are only encoded with a projector");
    assert_that(images.empty() || tok_image_pad >= 0, "image pads have a token");
    std::vector<std::pair<int32_t, const pending_image *>> splices;  // offset in t
    const std::vector<int32_t> t = turn_tokens(user, splices);
    const int32_t n_pre = (int32_t) t.size();

    if ((int32_t) hist.size() + n_pre >= (int32_t) o.cfg.n_ctx) {
        printf("  that message needs %d tokens but only %d are left of %d. /reset first.\n",
               n_pre, (int32_t) o.cfg.n_ctx - (int32_t) hist.size(), (int) o.cfg.n_ctx);
        return true;
    }
    const int32_t t_base = (int32_t) hist.size();
    hist.insert(hist.end(), t.begin(), t.end());
    for (const auto & sp : splices)
        eng.set_embeddings(t_base + sp.first, sp.second->emb.data(), sp.second->n_tok);
    images.clear();

    const float * lg = nullptr;
    int32_t pre_total = 0;
    double t_pre = 0;
    if (!prefill_turn(lg, pre_total, t_pre)) return false;

    // Budget the reply against the context wall. Generating into it makes
    // eval() fail, and before this it took the whole process down with it.
    const int32_t room   = (int32_t) o.cfg.n_ctx - (int32_t) hist.size() - 2;
    const int     budget = o.max_gen > 0 ? std::min<int>(o.max_gen, std::max<int32_t>(room, 0))
                                         : std::max<int32_t>(room, 0);
    reply_state r;
    r.in_think = thinking;
    decode_reply(lg, budget, r);
    finish_turn(r, budget, pre_total, t_pre);
    return true;
}

void chat_session::session_summary() {
    if (s_turns == 0) return;
    assert_that(s_pre_tok >= 0 && s_dec_tok >= 0, "token totals");
    assert_that(s_pre_t >= 0 && s_dec_t >= 0, "time totals");
    dim();
    printf("\nsession: %d repl%s\n", s_turns, s_turns == 1 ? "y" : "ies");
    if (eng.pred_total) printf("  prefetch   %.1f%% of the next layer's experts predicted (%llu scored)\n",
           100.0 * eng.pred_hits / eng.pred_total, (unsigned long long) eng.pred_total);
    printf("  prefill    %lld tok in %.1f s   (%.1f tok/s avg)\n",
           (long long) s_pre_tok, s_pre_t, s_pre_tok / (s_pre_t > 0 ? s_pre_t : 1e-9));
    printf("  generated  %lld tok in %.1f s   (%.1f tok/s avg)\n",
           (long long) s_dec_tok, s_dec_t, s_dec_t > 0 ? s_dec_tok / s_dec_t : 0.0);
    printf("  total      %.1f s of model time, context %d/%d\n",
           s_pre_t + s_dec_t, eng.n_past(), (int) o.cfg.n_ctx);
    undim();
}

// An interactive command. 0: not a command (a message), 1: handled, 2: quit.
int chat_session::command(const std::string & cmd) {
    assert_that(!cmd.empty(), "blank lines are skipped by the caller");
    assert_that(cmd == trim(cmd), "the caller trims the line");
    if (cmd == "/quit" || cmd == "/exit") return 2;
    if (cmd == "/reset") {
        reset_conversation(); pending.clear(); images.clear(); eng.clear_embeddings();
        printf("(conversation cleared)\n");
        return 1;
    }
    if (cmd == "/drop")  { pending.clear(); images.clear(); printf("(attachments discarded)\n"); return 1; }
    if (cmd == "/stats") { session_summary(); return 1; }
    if (cmd == "/files") {
        if (pending.empty()) printf("(nothing attached)\n");
        for (const auto & a : pending)
            printf("  %s  (%zu bytes, ~%zu tokens)\n", a.path.c_str(), a.body.size(), a.n_tok);
        return 1;
    }
    if (cmd.rfind("/image", 0) == 0 && (cmd.size() == 6 || cmd[6] == ' ')) {
        const std::string path = trim(cmd.substr(6));
        if (path.empty()) printf("  usage: /image PATH\n");
        else attach_image(path);
        return 1;
    }
    if (cmd.rfind("/file", 0) == 0 && (cmd.size() == 5 || cmd[5] == ' ')) {
        // Rest of the line is the path, so paths containing spaces work.
        const std::string path = trim(cmd.substr(5));
        if (path.empty()) printf("  usage: /file PATH\n");
        else attach(path);
        return 1;
    }
    if (cmd[0] == '/') { printf("  unknown command: %s\n", cmd.c_str()); return 1; }
    return 0;
}

int chat_session::interactive_loop() {
    assert_that(o.interactive, "interactive mode");
    assert_that(fed <= (int32_t) hist.size(), "the engine has not seen past the history");
    const bool stdin_tty = isatty(fileno(stdin));
    printf("\nqwfn-chat ready. /file PATH attaches a file, /reset clears, /quit exits.\n");
    if (!stdin_tty) printf("(reading piped input)\n");
    std::string line;
    bool any = false;
    // rule 2 deviation: the interactive REPL runs until /quit or end of input, see docs/CODING_RULES.md
    for (;;) {
        printf("\n");
        ansi("\033[1m");
        printf("> ");
        ansi("\033[0m");
        fflush(stdout);
        if (!std::getline(std::cin, line)) {
            if (!any && !stdin_tty) {
                printf("\n\nNo input on stdin. The interactive prompt needs a terminal --\n"
                       "run this from a shell, or pipe messages in:\n"
                       "  printf 'hello\\n/quit\\n' | qwfn-chat MODEL --ram 12\n"
                       "or ask one question and exit:\n"
                       "  qwfn-chat MODEL --ram 12 -p \"your question\"\n");
            }
            break;
        }
        const std::string cmd = trim(line);
        if (cmd.empty()) continue;
        const int c = command(cmd);
        if (c == 2) break;
        if (c == 1) continue;

        printf("\n");
        any = true;
        if (!turn(line)) return 1;
    }
    session_summary();
    printf("\n");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fputs(k_usage, stderr);
        return 1;
    }
    chat_opts o;
    if (!parse_args(argc, argv, o)) return 1;
    if (!effort_valid(o.effort)) {
        fprintf(stderr, "--think must be xhigh, medium, low or off\n"); return 1;
    }
    chat_session s(o);
    if (!s.init(argv[1])) return 1;
    s.reset_conversation();
    assert_that(s.fed == 0, "a fresh conversation");

    for (const auto & f : o.startup_files)  s.attach(f);
    for (const auto & f : o.startup_images) s.attach_image(f);

    if (!o.interactive) {
        if (!s.turn(o.one_shot)) return 1;
        s.session_summary();
        return 0;
    }
    assert_that(o.interactive, "one-shot mode returned above");
    return s.interactive_loop();
}
