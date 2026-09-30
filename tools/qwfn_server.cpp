// qwfn-server -- an HTTP front end speaking the OpenAI chat API and the
// Anthropic Messages API (Claude Code).
//
// One engine, one request at a time. That is not a simplification to be fixed
// later: the engine holds a single KV cache plus DeltaNet recurrent state and
// short-conv history, all of which are sequential accumulations over one
// sequence. Two interleaved conversations would corrupt each other, so requests
// take a mutex and run to completion.
//
// The one optimisation that IS sound here is prefix continuation. A harness
// replays the whole conversation every turn, and if the new token sequence is an
// exact EXTENSION of what the engine has already consumed, we can feed only the
// tail -- position only ever moves forward, so the recurrent state stays valid.
// Anything else (edited history, a new conversation, a regenerate) resets and
// re-prefills, because the recurrent layers cannot be rewound: unlike a KV cache
// you cannot simply forget the tail of a scan.

#include "qwfn_engine.h"
#include "qwfn_model.h"
#include "qwfn_vocab.h"
#include "qwfn_vision.h"
#include "qwfn_template.h"
#include "qwfn_check.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <sys/stat.h>
#include <execinfo.h>
#include <csignal>
#include <atomic>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <vector>

using namespace qwfn;
using json = nlohmann::ordered_json;
using clk  = std::chrono::steady_clock;

// Session snapshots (saved sequences, shared-prefix checkpoints): on unless QWFN_SNAPSHOTS is
// 0, off, false or no.
static bool snapshots_on() {
    static const bool on = [] {
        const char * v = getenv("QWFN_SNAPSHOTS");
        if (!v) return true;
        const std::string s = v;
        qwfn::assert_that(s.size() < (1u << 20), "an environment value of sane length");
        return !(s == "0" || s == "off" || s == "false" || s == "no");
    }();
    qwfn::assert_that(on || getenv("QWFN_SNAPSHOTS") != nullptr, "snapshots are off only when asked");
    return on;
}

static double since(clk::time_point t) {
    return std::chrono::duration<double>(clk::now() - t).count();
}
static int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
// A value the JSON parser produced (the checks below assert it of JSON arguments: whatever a
// client sent, the parser never yields a discarded or binary value).
static bool parsed(const json & j) { return !j.is_discarded() && !j.is_binary(); }

// ---- base64, for data: image URLs -------------------------------------------
static bool b64_decode(const std::string & in, std::vector<uint8_t> & out) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    int acc = 0, bits = 0;
    out.clear();
    for (char c : in) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        const int v = val(c);
        if (v < 0) return false;
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((uint8_t) ((acc >> bits) & 0xFF)); }
    }
    qwfn::assert_that(bits >= 0 && bits < 8, "b64: fewer than 8 bits left over");
    qwfn::assert_that(out.size() <= in.size(), "b64: decoded no longer than encoded");
    return true;
}

// A numeric field of a request object within [lo, hi]. Absent: `out` untouched. False, with `err`,
// on another type or out of range; the conversion of a valid value is json's own get<T>().
template <typename T>
static bool num_field(const json & j, const char * key, double lo, double hi, T & out, std::string & err) {
    qwfn::assert_that(key != nullptr, "num_field: a key");
    qwfn::assert_that(lo <= hi, "num_field: a range");
    if (!j.is_object() || !j.contains(key)) return true;
    const json & v = j[key];
    const double d = v.is_number() ? v.get<double>() : NAN;
    if (!(d >= lo && d <= hi)) {
        char buf[160];
        snprintf(buf, sizeof buf, "%s must be a number in [%g, %g]", key, lo, hi);
        err = buf;
        return false;
    }
    out = v.get<T>();
    return true;
}

// Sampling settings. The two presets are the model's own: the GGUF embeds the
// thinking one (temp 1.0, top_p 0.95, top_k 20); the model card wants temp 0.7,
// top_p 0.8, top_k 20, presence_penalty 1.5 when thinking is off. A request
// overrides any field; POST /props changes the presets for everyone.
struct sampling {
    float temp = 1.0f, top_p = 0.95f, min_p = 0.0f;
    int   top_k = 20;
    float presence_penalty = 0.0f, frequency_penalty = 0.0f, repeat_penalty = 1.0f;
    int   repeat_last_n = 64;
    json to_json() const {
        return {{"temperature", temp}, {"top_p", top_p}, {"top_k", top_k}, {"min_p", min_p},
                {"presence_penalty", presence_penalty}, {"frequency_penalty", frequency_penalty},
                {"repeat_penalty", repeat_penalty}, {"repeat_last_n", repeat_last_n}};
    }
    // Take whatever fields `j` carries (a request body or a /props update). False, with `err`, on
    // a field of the wrong type or out of range; the settings may then be partly updated, so a
    // caller applies them to a copy. The bounds keep every penalised logit finite: a
    // repeat_penalty of 0 (or a denormal) divides to inf and the softmax turns to NaN.
    bool from_json(const json & j, std::string & err) {
        qwfn::assert_that(parsed(j), "sampling: a parsed JSON value");
        qwfn::assert_that(j.is_object() || !j.contains("temperature"), "sampling: only an object carries fields");
        const double F = FLT_MAX, I = INT_MAX;
        return num_field(j, "temperature", -F, F, temp, err) && num_field(j, "top_p", -F, F, top_p, err)
            && num_field(j, "top_k", -1, I, top_k, err) && num_field(j, "min_p", -F, F, min_p, err)
            && num_field(j, "presence_penalty", -1000, 1000, presence_penalty, err)
            && num_field(j, "frequency_penalty", -1000, 1000, frequency_penalty, err)
            && num_field(j, "repeat_penalty", 1e-3, 1000, repeat_penalty, err)
            && num_field(j, "repeat_last_n", -1, I, repeat_last_n, err);
    }
};
static sampling preset_thinking()     { sampling s; return s; }
static sampling preset_non_thinking() { sampling s; s.temp = 0.7f; s.top_p = 0.8f; s.top_k = 20; s.presence_penalty = 1.5f; return s; }

struct sampler {
    sampling     cfg;
    std::mt19937 rng{0xC0FFEEu};
    std::vector<int32_t> gen;   // tokens generated so far, for the penalties

    // The k largest logits, descending. A partial_sort over the whole vocabulary
    // (248K entries) cost 1.5-3 ms per call and a sampled pair step makes three
    // or four of them; instead two linear passes -- the max, then every logit
    // within 40 temperatures of it (a relative probability of e^-40, nothing
    // at float precision) -- and a partial_sort over the few that survive.
    // Same k best, same order, whenever at least k survive; when fewer do,
    // the ones left out had zero probability anyway.
    void top_indices(const float * lg, int64_t n, int k, std::vector<int> & idx) const {
        qwfn::assert_that(lg != nullptr && n > 0, "top_indices: logits");
        float mx = lg[0];
        for (int64_t v = 1; v < n; v++) mx = std::max(mx, lg[v]);
        const float margin = 40.0f * std::max(cfg.temp, 1e-3f);
        const float floor_ = mx - margin;
        idx.clear();
        for (int64_t v = 0; v < n; v++) if (lg[v] >= floor_) idx.push_back((int) v);
        const int kk = (int) std::min<size_t>((size_t) k, idx.size());
        qwfn::assert_that(kk >= 0 && (size_t) kk <= idx.size(), "top_indices: k within the survivors");
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        idx.resize(kk);
    }

    // The logits with the repetition penalties applied, over the last repeat_last_n
    // GENERATED tokens (llama.cpp semantics): `lg_in` itself when none applies, else
    // a copy in `pen`.
    const float * penalised(const float * lg_in, int64_t n, std::vector<float> & pen) const {
        qwfn::assert_that(lg_in != nullptr && n > 0, "penalties: logits");
        const bool penalise = cfg.repeat_last_n != 0 && !gen.empty() &&
            (cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f || cfg.repeat_penalty != 1.0f);
        if (!penalise) return lg_in;
        pen.assign(lg_in, lg_in + n);
        const size_t from = cfg.repeat_last_n > 0 && gen.size() > (size_t) cfg.repeat_last_n ? gen.size() - cfg.repeat_last_n : 0;
        qwfn::assert_that(from < gen.size(), "penalties: a window inside the generated tokens");
        std::unordered_map<int32_t, int> cnt;
        for (size_t i = from; i < gen.size(); i++) cnt[gen[i]]++;
        for (const auto & [t, c] : cnt) {
            if (t < 0 || t >= n) continue;
            float & v = pen[t];
            if (cfg.repeat_penalty != 1.0f) v = v > 0 ? v / cfg.repeat_penalty : v * cfg.repeat_penalty;
            v -= cfg.presence_penalty + cfg.frequency_penalty * c;
        }
        return pen.data();
    }

    // Top-k, temperature, min-p, top-p over `lg` (temperature > 0): the candidates in
    // `idx`, their probabilities in `p`; returns how many are kept, `cum` their mass.
    int candidates(const float * lg, int64_t n, std::vector<int> & idx, std::vector<float> & p, double & cum) const {
        qwfn::assert_that(cfg.temp > 0.0f, "candidates: sampling at temperature");
        top_indices(lg, n, (int) std::min<int64_t>(cfg.top_k > 0 ? cfg.top_k : n, n), idx);
        const int k = (int) idx.size();
        const float mx = lg[idx[0]];
        p.assign(k, 0.0f);
        double sum = 0;
        for (int i = 0; i < k; i++) { p[i] = std::exp((lg[idx[i]] - mx) / cfg.temp); sum += p[i]; }
        for (int i = 0; i < k; i++) p[i] = (float) (p[i] / sum);
        int keep = k;
        if (cfg.min_p > 0.0f) { const float floor_ = cfg.min_p * p[0]; keep = 1; while (keep < k && p[keep] >= floor_) keep++; }
        cum = 0; int keep_p = keep;
        for (int i = 0; i < keep; i++) { cum += p[i]; if (cum >= cfg.top_p) { keep_p = i + 1; break; } }
        qwfn::assert_that(keep_p >= 0 && keep_p <= k, "candidates: kept within the candidates");
        return keep_p;
    }

    // The distribution pick() samples from -- penalties, top-k, temperature,
    // min-p, top-p -- as (token, probability) over the kept candidates. Empty
    // at temperature 0 (greedy: pick() takes the argmax).
    std::vector<std::pair<int32_t, float>> dist(const float * lg_in, int64_t n) {
        std::vector<std::pair<int32_t, float>> out;
        if (cfg.temp <= 0.0f) return out;
        std::vector<float> pen;
        const float * lg = penalised(lg_in, n, pen);
        std::vector<int> idx;
        std::vector<float> p;
        double cum = 0;
        const int keep = candidates(lg, n, idx, p, cum);
        out.reserve(keep);
        for (int i = 0; i < keep; i++) out.emplace_back(idx[i], (float) (p[i] / cum));
        qwfn::assert_that(out.size() <= idx.size(), "dist: no more tokens than candidates");
        qwfn::assert_that(lg != nullptr, "dist: logits");
        return out;
    }
    int32_t sample(const std::vector<std::pair<int32_t, float>> & d) {
        std::uniform_real_distribution<double> U(0.0, 1.0);
        double r = U(rng), acc = 0;
        qwfn::assert_that(r >= 0.0 && r <= 1.0, "sample: a draw in [0, 1]");
        for (const auto & [t, p] : d) { acc += p; if (r <= acc) return t; }
        // Requests are validated (repeat_penalty >= 0.001 etc.), but float overflow can still
        // make a weight NaN: the fallback below handles it, so only a real negative sum is a bug.
        qwfn::assert_that(!(acc < 0.0), "sample: probabilities are not negative");
        return d.empty() ? 0 : d.back().first;
    }
    static float prob_of(const std::vector<std::pair<int32_t, float>> & d, int32_t t) {
        qwfn::assert_that(t >= 0, "prob_of: a token id");
        qwfn::assert_that(d.empty() || d.front().first >= 0, "prob_of: token ids in the distribution");
        for (const auto & [x, p] : d) if (x == t) return p;
        return 0.0f;
    }

    int pick(const float * lg_in, int64_t n) {
        std::vector<float> pen;
        const float * lg = penalised(lg_in, n, pen);
        if (cfg.temp <= 0.0f) {
            int best = 0;
            for (int64_t v = 1; v < n; v++) if (lg[v] > lg[best]) best = (int) v;
            qwfn::assert_that(best >= 0 && best < n, "pick: the argmax is a token");
            return best;
        }
        std::vector<int> idx;
        std::vector<float> p;
        double cum = 0;
        const int keep = candidates(lg, n, idx, p, cum);
        qwfn::assert_that(keep <= (int) idx.size(), "pick: kept within the candidates");
        std::uniform_real_distribution<double> U(0.0, cum);
        double r = U(rng), acc = 0;
        for (int i = 0; i < keep; i++) { acc += p[i]; if (r <= acc) return idx[i]; }
        return idx[0];
    }
};

// What a harness polls for a live counter: written by the generating thread
// per token, read by /stats, /metrics and /slots without the engine mutex.
// ---- tool calling ------------------------------------------------------------
// The model's own template (Qwen3.8 Flash Next): functions listed as JSON in
// the system turn, calls emitted as
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
// and results fed back as a user turn of <tool_response> blocks. The server
// translates OpenAI `tools` / `tool_calls` / role "tool" both ways.
static const char * TOOLS_INSTRUCTIONS =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n"
    "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

static std::string render_tools_block(const json & tools) {
    qwfn::assert_that(parsed(tools), "tools block: a parsed JSON value");
    if (!tools.is_array() || tools.empty()) return {};
    std::string s = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const auto & t : tools) s += "\n" + t.dump();
    s += "\n</tools>";
    s += TOOLS_INSTRUCTIONS;
    qwfn::assert_that(s.size() > strlen(TOOLS_INSTRUCTIONS), "tools block: the instructions included");
    return s;
}

// OpenAI arguments come as a JSON string (or, from some clients, an object).
static json tool_args_object(const json & fn) {
    qwfn::assert_that(parsed(fn), "tool arguments: a parsed JSON value");
    qwfn::assert_that(fn.is_object() || !fn.contains("arguments"), "tool arguments: only an object carries them");
    if (!fn.contains("arguments")) return json::object();
    const json & a = fn["arguments"];
    if (a.is_object()) return a;
    if (a.is_string()) { try { json o = json::parse(a.get<std::string>()); if (o.is_object()) return o; } catch (...) {} }
    return json::object();
}

// A stable key for "is this the reply we just produced": names and arguments.
// Arguments compare as sorted-key JSON: a harness replays them re-encoded
// (Unsloth Studio sorts the keys), and a key order the model never chose must
// not cost the conversation its prefix continuation.
static std::string tool_calls_key(const json & tool_calls) {
    qwfn::assert_that(parsed(tool_calls), "tool calls key: a parsed JSON value");
    if (!tool_calls.is_array()) return {};
    std::string k;
    for (const auto & tc : tool_calls) {
        const json & fn = tc.contains("function") ? tc["function"] : tc;
        k += fn.value("name", "") + "(" + nlohmann::json(tool_args_object(fn)).dump() + ");";
    }
    qwfn::assert_that(k.empty() || k.back() == ';', "tool calls key: one entry per call");
    return k;
}

// The reply text a client echoes back may differ from what the server kept in
// trailing whitespace: a streamed reply emitted "OK\n\n" before its tool block
// was recognised while parse_tool_calls() keeps the text trimmed, and clients
// trim on their own. Matching modulo trailing whitespace is what keeps the
// prefix continuation alive across a tool-calling turn.
static bool same_reply_text(const std::string & a, const std::string & b) {
    size_t na = a.size(), nb = b.size();
    while (na > 0 && (a[na - 1] == '\n' || a[na - 1] == ' ' || a[na - 1] == '\r' || a[na - 1] == '\t')) na--;
    while (nb > 0 && (b[nb - 1] == '\n' || b[nb - 1] == ' ' || b[nb - 1] == '\r' || b[nb - 1] == '\t')) nb--;
    qwfn::assert_that(na <= a.size(), "same_reply_text: trimmed within a");
    qwfn::assert_that(nb <= b.size(), "same_reply_text: trimmed within b");
    return na == nb && a.compare(0, na, b, 0, nb) == 0;
}

// Parse every complete <tool_call> block in `text`. Parameter values are typed
// by the tool's schema (a "string" stays a string; anything else is parsed as
// JSON when it parses). Returns the text before the first block.
// A call starts only at the template's full marker: a bare "<tool_call>" in
// prose ("I already sent a <tool_call>...") is text, not a call.
static const char * TC_OPEN = "<tool_call>";
static const char * TC_MARK = "<tool_call>\n<function=";
static const char * TC_CLOSE = "</tool_call>";
static size_t find_tool_block(const std::string & text, size_t from) {
    const std::string mark = TC_MARK;
    qwfn::assert_that(mark.size() > strlen(TC_OPEN), "find_tool_block: the marker extends the tag");
    size_t p = text.find(TC_OPEN, from);
    while (p != std::string::npos && text.compare(p, mark.size(), mark) != 0) p = text.find(TC_OPEN, p + 1);
    qwfn::assert_that(p == std::string::npos || p >= from, "find_tool_block: found at or after the start");
    return p;
}
// parameters.properties of the tool named `name`, for typing (null when it has none).
static json tool_schema(const json & tools, const std::string & name) {
    qwfn::assert_that(parsed(tools), "tool schema: a parsed JSON value");
    json schema;
    if (tools.is_array())
        for (const auto & t : tools) {
            const json & fn = t.contains("function") ? t["function"] : t;
            if (fn.value("name", "") == name && fn.contains("parameters") && fn["parameters"].contains("properties")) schema = fn["parameters"]["properties"];
        }
    qwfn::assert_that(parsed(schema), "tool schema: a JSON value");
    return schema;
}
// A parameter value as the batch parser and the streamer both type it: its schema type
// "string" keeps it a string; anything else is parsed as JSON when it parses (and, with
// no schema type, when it is not a bare JSON string).
static json typed_value(std::string val, const std::string & ty) {
    const size_t n0 = val.size();
    if (!val.empty() && val.back() == '\n') val.pop_back();
    qwfn::assert_that(val.size() + 1 >= n0, "typed_value: at most the one newline dropped");
    json v = val;
    if (ty != "string") { try { json p = json::parse(val); if (ty.empty() ? !p.is_string() : true) v = p; } catch (...) {} }
    qwfn::assert_that(parsed(v), "typed_value: a JSON value");
    return v;
}
// The <parameter=KEY>VALUE</parameter> pairs of one block, from `q` on.
static json parse_params(const std::string & block, size_t q, const json & schema) {
    qwfn::assert_that(q <= block.size(), "parse_params: start within the block");
    json args = json::object();
    // Every pass consumes a "<parameter=" tag, so the block's length bounds the passes.
    for (size_t pass = 0; pass <= block.size(); pass++) {
        const size_t ps = block.find("<parameter=", q);
        if (ps == std::string::npos) break;
        const size_t pe = block.find('>', ps);
        if (pe == std::string::npos) break;
        const std::string key = block.substr(ps + 11, pe - ps - 11);
        size_t vs = pe + 1;
        if (vs < block.size() && block[vs] == '\n') vs++;
        // A value runs to </parameter>; one the model never closed runs to the
        // end of its function -- the same cut the streamed parser makes.
        size_t ve = block.find("</parameter>", vs);
        const bool closed = ve != std::string::npos;
        if (!closed) { ve = block.find("</function>", vs); if (ve == std::string::npos) ve = block.size(); }
        qwfn::assert_that(vs <= ve && ve <= block.size(), "parse_params: a value within the block");
        const std::string ty = schema.is_object() && schema.contains(key) ? schema[key].value("type", "") : "";
        args[key] = typed_value(block.substr(vs, ve - vs), ty);
        if (!closed) break;
        q = ve + 12;
    }
    return args;
}
static std::string parse_tool_calls(const std::string & text, const json & tools, json & calls, size_t * consumed = nullptr) {
    calls = json::array();
    const std::string OPEN = TC_OPEN, CLOSE = TC_CLOSE;
    size_t first = find_tool_block(text, 0);
    std::string before = text.substr(0, first == std::string::npos ? text.size() : first);
    size_t pos = first, end_consumed = first == std::string::npos ? text.size() : first;
    while (pos != std::string::npos) {
        // A block ends at </tool_call>. The last block of a reply that the model
        // closed with </function> and then ended (no </tool_call>) counts too:
        // the call is complete, only the wrapper is missing.
        size_t close = text.find(CLOSE, pos), after = close == std::string::npos ? 0 : close + CLOSE.size();
        if (close == std::string::npos) {
            const size_t fc = text.find("</function>", pos);
            if (fc == std::string::npos || text.find(OPEN, fc) != std::string::npos) break;
            close = fc + 11; after = text.size();
        }
        qwfn::assert_that(pos + OPEN.size() <= close, "parse_tool_calls: a block ends after its tag");
        const std::string block = text.substr(pos + OPEN.size(), close - pos - OPEN.size());
        end_consumed = after;
        pos = find_tool_block(text, end_consumed);
        const size_t f = block.find("<function=");
        if (f == std::string::npos) continue;
        const size_t fe = block.find('>', f);
        if (fe == std::string::npos) continue;
        const std::string name = block.substr(f + 10, fe - f - 10);
        const json args = parse_params(block, fe + 1, tool_schema(tools, name));
        calls.push_back(json{{"id", "call_" + std::to_string(calls.size()) + "_" + std::to_string(now_unix() % 100000)},
                             {"type", "function"},
                             {"function", {{"name", name}, {"arguments", args.dump()}}}});
    }
    qwfn::assert_that(end_consumed <= text.size(), "parse_tool_calls: consumed within the text");
    if (consumed) *consumed = end_consumed;
    while (!before.empty() && (before.back() == '\n' || before.back() == ' ')) before.pop_back();
    return before;
}

// Token pieces can end inside a multi-byte UTF-8 character (an em dash, CJK,
// an emoji split over two tokens). nlohmann::json::dump() throws on invalid
// UTF-8, and an exception out of a streaming provider makes httplib close the
// connection with no trailer and no log line -- "peer closed connection
// without sending complete message body" on the client. So deltas hold back an
// incomplete tail until the next piece completes it.
static size_t utf8_incomplete_tail(const std::string & s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= 3 && back <= n; back++) {
        const unsigned char c = (unsigned char) s[n - back];
        if ((c & 0xC0) == 0x80) continue;          // continuation byte: keep looking for the lead
        const size_t need = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        qwfn::assert_that(need >= 1 && need <= 4, "utf8: a sequence is 1 to 4 bytes");
        const size_t held = back < need ? back : 0;  // lead byte found: is the sequence complete?
        qwfn::assert_that(held <= 3 && held <= n, "utf8: a held tail is at most 3 bytes");
        return held;
    }
    return 0;
}
static std::string json_dump(const json & j) {   // never throws on bad UTF-8
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}
// The inside of a JSON string literal for `s`, escaped exactly as json::dump does it.
static std::string json_escape(const std::string & s) {
    const std::string d = json_dump(json(s));
    return d.substr(1, d.size() - 2);
}
static uint64_t hash_bytes(const void * p, size_t n) {   // FNV-1a
    qwfn::assert_that(p != nullptr || n == 0, "hash_bytes: data for a non-empty range");
    qwfn::assert_that(n <= SIZE_MAX / 2, "hash_bytes: a sane length");
    uint64_t h = 1469598103934665603ull;
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// Streams one <tool_call> block as OpenAI tool_calls deltas WHILE the model is
// writing it. The opening delta (index, id, name) goes out as soon as
// "<function=NAME>" is complete; the arguments follow as a JSON object built up
// in fragments, so the concatenation the client accumulates is what the batch
// parser would have produced. A parameter the tool's schema types as "string"
// (which is what carries code and file contents) streams as it is written,
// JSON-escaped; one typed "array" or "object" (an edit list) streams as the raw
// JSON the model writes; any other parameter is held to its </parameter> and
// typed like the batch parser types it.
//
// Before this the whole block was held until </tool_call>: a 4,000-token file
// written into one parameter meant five minutes of silence on the wire, longer
// than a harness's read timeout (Unsloth Studio's proxy: 300 s), and the
// generation was cut off from the client's side with the work half done.
struct stream_events;   // the route's writers (below); a tool_streamer sends its deltas through them
struct tool_streamer {
    const json & tools;
    size_t block_start;                 // index of "<tool_call>" in the text
    int    index;                       // tool_calls[].index for this call
    std::string call_id;
    stream_events & ev;                 // where the tool_calls deltas go
    bool emit(const json & d);          // one tool_calls delta; false = write failed (defined with stream_events)

    enum { HEADER, PARAMS, VALUE_STR, VALUE_HELD, CLOSED } st = HEADER;
    enum step { NEXT, DONE, FAILED };   // a state's outcome: parse on, return true, return false
    bool   raw = false;                 // VALUE_STR streams the value unquoted and unescaped (array/object)
    bool   opened = false;              // the opening delta has been sent
    size_t pos = 0;                     // parse cursor into the text
    json   schema;                      // parameters.properties of the tool, for typing
    std::string key, ty;                // the parameter being written and its schema type
    int    n_params = 0;
    size_t val_start = 0, val_emitted = 0;

    tool_streamer(const json & t, size_t start, int idx, std::string id, stream_events & e)
        : tools(t), block_start(start), index(idx), call_id(std::move(id)), ev(e) {}

    bool args(const std::string & frag) {
        return emit(json{{"index", index}, {"function", {{"arguments", frag}}}});
    }
    bool close_object() {   // the end of the arguments object
        qwfn::assert_that(st != CLOSED, "tool stream: the object is closed once");
        qwfn::assert_that(n_params >= 0, "tool stream: a parameter count");
        st = CLOSED;
        return !opened || args(n_params == 0 ? "{}" : "}");
    }
    // The earliest closer at or after `from`: </parameter>, </function> or </tool_call>.
    static size_t next_closer(const std::string & t, size_t from, size_t & len) {
        static const char * C[3] = {"</parameter>", "</function>", "</tool_call>"};
        qwfn::assert_that(from <= t.size(), "tool stream: a cursor within the text");
        size_t best = std::string::npos; len = 0;
        for (const char * c : C) {
            const size_t p = t.find(c, from);
            if (p != std::string::npos && (best == std::string::npos || p < best)) { best = p; len = strlen(c); }
        }
        qwfn::assert_that(best == std::string::npos || len > 0, "tool stream: a closer has a length");
        return best;
    }
    json typed(std::string val) const { return typed_value(std::move(val), ty); }

    // "<function=NAME>": the opening delta once the name and the byte after it are in.
    step feed_header(const std::string & t) {
        qwfn::assert_that(st == HEADER && !opened, "tool stream: the header comes first");
        const size_t f = block_start + strlen(TC_MARK);
        const size_t fe = t.find('>', f), tc = t.find(TC_CLOSE, f);
        if (tc != std::string::npos && (fe == std::string::npos || tc < fe)) { st = CLOSED; return DONE; }   // no name: not a call
        if (fe == std::string::npos || fe + 1 >= t.size()) return DONE;   // need the name, and the byte after it
        const std::string name = t.substr(f, fe - f);
        schema = tool_schema(tools, name);
        pos = fe + 1; if (t[pos] == '\n') pos++;
        opened = true; st = PARAMS;
        qwfn::assert_that(pos <= t.size(), "tool stream: the cursor within the text");
        if (!emit(json{{"index", index}, {"id", call_id}, {"type", "function"},
                       {"function", {{"name", name}, {"arguments", ""}}}})) return FAILED;
        return NEXT;
    }
    // Between parameters: the next <parameter=KEY>, the end of the call, or text to skip.
    step feed_params(const std::string & t) {
        qwfn::assert_that(st == PARAMS && opened, "tool stream: between parameters of an open call");
        qwfn::assert_that(pos <= t.size(), "tool stream: the cursor within the text");
        size_t q = pos;
        while (q < t.size() && isspace((unsigned char) t[q])) q++;
        if (q >= t.size()) return DONE;
        if (t.compare(q, 11, "<parameter=") == 0) {
            const size_t pe = t.find('>', q);
            if (pe == std::string::npos || pe + 1 >= t.size()) return DONE;   // need the key, and the byte after it
            key = t.substr(q + 11, pe - q - 11);
            pos = pe + 1; if (t[pos] == '\n') pos++;
            val_start = val_emitted = pos;
            ty = schema.is_object() && schema.contains(key) ? schema[key].value("type", "") : "";
            std::string frag = (n_params == 0 ? "{" : ",") + json_dump(json(key)) + ":";
            n_params++;
            raw = ty == "array" || ty == "object";
            if (ty == "string") { frag += "\""; st = VALUE_STR; }
            else if (raw)       { st = VALUE_STR; }
            else                { st = VALUE_HELD; }
            return args(frag) ? NEXT : FAILED;
        }
        if (t.compare(q, 11, "</function>") == 0) { pos = q + 11; return close_object() ? DONE : FAILED; }
        if (t.compare(q, 12, TC_CLOSE) == 0)      { pos = q;      return close_object() ? DONE : FAILED; }
        // Something else. Still a possible prefix of one of the tags: wait.
        // Otherwise skip to the next tag, as the batch parser's find() does.
        const size_t avail = t.size() - q;
        for (const char * tag : {"<parameter=", "</function>", "</tool_call>"})
            if (avail < strlen(tag) && strncmp(tag, t.c_str() + q, avail) == 0) return DONE;
        size_t best = t.find("<parameter=", q); size_t len = 0;
        const size_t c = next_closer(t, q, len);
        if (c != std::string::npos && (best == std::string::npos || c < best)) best = c;
        if (best == std::string::npos) return DONE;
        // A stray closer right here (only </parameter> can be: the others were handled
        // above): step over it, or pos would never advance and the loop would not end.
        pos = best == q ? q + len : best;
        return NEXT;
    }
    // A value that streams as it is written (a string, escaped; an array or object, raw).
    step feed_value_str(const std::string & t) {
        qwfn::assert_that(st == VALUE_STR, "tool stream: inside a streamed value");
        qwfn::assert_that(val_start <= val_emitted && val_emitted <= t.size(), "tool stream: emitted within the value");
        size_t len = 0;
        const size_t c = next_closer(t, val_emitted, len);
        if (c != std::string::npos) {
            size_t vend = c;
            if (vend > val_start && t[vend - 1] == '\n') vend--;
            std::string frag = vend > val_emitted ? (raw ? t.substr(val_emitted, vend - val_emitted) : json_escape(t.substr(val_emitted, vend - val_emitted))) : std::string();
            if (!raw) frag += "\"";
            if (!frag.empty() && !args(frag)) return FAILED;
            if (t.compare(c, len, "</parameter>") == 0) { pos = c + len; st = PARAMS; return NEXT; }
            pos = t.compare(c, len, "</function>") == 0 ? c + len : c;
            return close_object() ? DONE : FAILED;
        }
        // No closer yet: emit what cannot still become one.
        const size_t avail = t.size() - val_emitted;
        size_t hold = 0;
        for (size_t k = std::min<size_t>(13, avail); k > 0 && !hold; k--)
            for (const char * tag : {"\n</parameter>", "</parameter>", "\n</function>", "</function>", "\n</tool_call>", "</tool_call>"})
                if (strncmp(tag, t.c_str() + t.size() - k, k) == 0) { hold = k; break; }
        const size_t upto = t.size() - hold;
        if (upto > val_emitted) {
            const std::string piece = t.substr(val_emitted, upto - val_emitted);
            if (!args(raw ? piece : json_escape(piece))) return FAILED;
            val_emitted = upto;
        }
        return DONE;
    }
    // A value held to its closer, then typed like the batch parser types it.
    step feed_value_held(const std::string & t) {
        qwfn::assert_that(st == VALUE_HELD, "tool stream: inside a held value");
        qwfn::assert_that(val_start <= t.size(), "tool stream: the value starts within the text");
        size_t len = 0;
        const size_t c = next_closer(t, val_start, len);
        if (c == std::string::npos) return DONE;
        if (!args(json_dump(typed(t.substr(val_start, c - val_start))))) return FAILED;
        if (t.compare(c, len, "</parameter>") == 0) { pos = c + len; st = PARAMS; return NEXT; }
        pos = t.compare(c, len, "</function>") == 0 ? c + len : c;
        return close_object() ? DONE : FAILED;
    }
    // Feed the text so far (the whole accumulated content, block_start-relative
    // positions are absolute into it). Returns false only on a write failure.
    bool feed(const std::string & t) {
        qwfn::assert_that(block_start <= t.size(), "tool stream: the block starts within the text");
        // A NEXT advances pos, or moves PARAMS to a value state with pos past the key:
        // at most two passes per byte of text.
        const size_t cap = 2 * t.size() + 4;
        for (size_t pass = 0; pass < cap; pass++) {
            qwfn::assert_that(st >= HEADER && st <= CLOSED, "tool stream: a known state");
            const step r = st == CLOSED     ? DONE
                         : st == HEADER     ? feed_header(t)
                         : st == PARAMS     ? feed_params(t)
                         : st == VALUE_STR  ? feed_value_str(t)
                         :                    feed_value_held(t);
            if (r != NEXT) return r == DONE;
        }
        qwfn::assert_that(false, "tool stream: the parse advances on every pass");
        return false;
    }
};

// Backtraces without a debugger (ptrace is restricted on this machine): a
// fatal signal prints the dying thread's stack; SIGUSR2, sent by the stall
// watchdog to the generating thread, prints where it is stuck.
static void print_backtrace(const char * why) {
    qwfn::assert_that(why != nullptr, "backtrace: a reason");
    void * frames[64];
    const int n = backtrace(frames, 64);
    qwfn::assert_that(n >= 0 && n <= 64, "backtrace: frames within the buffer");
    char head[160];
    const int hl = snprintf(head, sizeof head, "\n[qwfn-server] === %s: backtrace of thread %lu (%d frames) ===\n", why, (unsigned long) pthread_self(), n);
    (void) !write(2, head, hl);
    backtrace_symbols_fd(frames, n, 2);
}
static void on_fatal(int sig) {
    qwfn::assert_that(sig > 0, "fatal signal: a signal number");
    qwfn::assert_that(sig < NSIG, "fatal signal: a known signal");
    print_backtrace(sig == SIGSEGV ? "SIGSEGV" : sig == SIGABRT ? "SIGABRT" : sig == SIGBUS ? "SIGBUS" : sig == SIGFPE ? "SIGFPE" : "fatal signal");
    signal(sig, SIG_DFL); raise(sig);
}
static void on_stall_probe(int) { print_backtrace("STALL probe (SIGUSR2)"); }

struct live_stats {
    std::mutex mu;
    bool   busy = false;
    // The current (or, when idle, the last) request: its whole prompt, the part reused
    // from the engine's prefix, the part being prefilled and the tokens generated.
    int    n_input = 0, n_cached = 0, n_prompt = 0, n_gen = 0;
    double t_prompt = 0, t_gen = 0;          // seconds, the current or last request
    // Prefill progress: new prompt tokens done so far, fractional inside a batch (the
    // engine reports each layer of a streamed batch), and the clock it runs on, so a
    // reader sees the rate move while the prefill is still going.
    bool   prefilling = false;
    double prompt_done = 0, prompt_base = 0;
    const qwfn::engine * eng = nullptr;      // its prefill_progress() refines prompt_done inside a batch
    double prompt_done_locked() const {      // mu held
        const qwfn::engine::prefill_progress_t p = eng ? eng->prefill_progress() : qwfn::engine::prefill_progress_t{};
        qwfn::assert_that(p.layers_done <= p.n_layer, "live: layers done within the model");
        qwfn::assert_that(prompt_base >= 0, "live: a batch base within the prompt");
        return p.layers_done > 0 ? prompt_base + (double) p.T * p.layers_done / p.n_layer : prompt_done;
    }
    std::chrono::steady_clock::time_point t_prompt0;
    double t_prompt_total = 0, t_gen_total = 0;
    long long n_prompt_total = 0, n_gen_total = 0, n_requests = 0, n_input_total = 0, n_cached_total = 0;
    long long n_pairs_total = 0, n_accepted_total = 0, n_drafted_total = 0;   // the draft head's verify steps, drafts accepted, drafts proposed
    int    n_past = 0;
    // The last request that completed, kept apart from the live counters so a monitor
    // can show it while the next one runs.
    struct snapshot {
        int n_input = 0, n_cached = 0, n_prompt = 0, n_gen = 0, n_pairs = 0, n_accepted = 0, n_drafted = 0;
        double t_prompt = 0, t_gen = 0; std::string finish; long long when = 0; bool ok = true;
    } last;
    bool have_last = false;
    double prompt_seconds_locked() const {   // mu held
        qwfn::assert_that(t_prompt >= 0, "live: a prompt time");
        qwfn::assert_that(prompt_done >= 0, "live: prefill progress");
        if (prefilling) return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prompt0).count();
        return t_prompt;
    }
    json timings() {   // llama.cpp's field names, plus the prompt's cached share and live progress
        std::lock_guard<std::mutex> lk(mu);
        const double tp = prompt_seconds_locked();
        const double done = prefilling ? prompt_done_locked() : (double) n_prompt;
        qwfn::assert_that(n_prompt >= 0 && n_gen >= 0, "live: token counts");
        qwfn::assert_that(tp >= 0 && t_gen >= 0, "live: elapsed times");
        return {{"prompt_n", n_prompt}, {"prompt_ms", tp * 1e3},
                {"prompt_per_second", tp > 0.05 ? done / tp : 0.0},
                {"prompt_cached_n", n_cached}, {"prompt_done_n", done}, {"prompt_input_n", n_input}, {"prefilling", prefilling},
                {"predicted_n", n_gen}, {"predicted_ms", t_gen * 1e3},
                // The first token comes straight from the prompt's logits at ~0 ms:
                // no rate until the clock has something to divide by.
                {"predicted_per_second", t_gen >= 0.1 ? n_gen / t_gen : 0.0}};
    }
    json last_json() {
        std::lock_guard<std::mutex> lk(mu);
        if (!have_last) return nullptr;
        qwfn::assert_that(!last.finish.empty(), "live: a finished request has a finish reason");
        qwfn::assert_that(last.n_gen >= 0 && last.n_prompt >= 0, "live: token counts");
        return {{"input_tokens", last.n_input}, {"cached_tokens", last.n_cached}, {"prompt_tokens", last.n_prompt}, {"generated_tokens", last.n_gen},
                {"prompt_ms", last.t_prompt * 1e3}, {"prompt_tokens_per_second", last.t_prompt > 0 ? last.n_prompt / last.t_prompt : 0.0},
                {"generation_ms", last.t_gen * 1e3}, {"tokens_per_second", last.t_gen > 0 ? last.n_gen / last.t_gen : 0.0},
                {"finish_reason", last.finish}, {"ok", last.ok}, {"completed_at", last.when},
                {"speculative", {{"pairs", last.n_pairs}, {"accepted", last.n_accepted}, {"drafted", last.n_drafted}}}};
    }
};

// ---- server state -----------------------------------------------------------
// content parts -> text, with images encoded to embeddings on the way past
struct pending_img { std::vector<float> emb; int n_tok = 0; };

struct server {
    model_index    mi;
    uint32_t       mtp_drafts = 2;   // --mtp-drafts: the most drafts a verify step carries (2: best on the 5060 Ti box)
    engine         eng;
    qwfn::vocab    vb;
    vision_encoder vis;
    ggml_backend_t vis_backend = nullptr;   // the CPU backend the projector runs on
    int32_t        tok_image_pad = -1;

    std::mutex           mu;
    std::vector<int32_t> consumed;      // exactly what the engine has evaluated
    // Sequences set aside when a request did not extend them (most recent last):
    // an agent's side request (a title, a summary) no longer costs the main
    // conversation a full prefill of its prefix. QWFN_SNAPSHOTS=0 disables.
    struct saved_seq { std::vector<int32_t> tok; engine::state_snapshot snap; };
    std::vector<saved_seq> saved;       // least recently used first
    size_t saved_cap = 0;               // bytes (QWFN_SNAP_MB, default 1024)
    // Insert with the byte cap enforced: a snapshot larger than the cap is not kept,
    // older ones go first. An equal token sequence is replaced.
    void save_seq(saved_seq && sv) {
        if (sv.snap.bytes() > saved_cap) return;
        for (size_t k = 0; k < saved.size(); k++) if (saved[k].tok == sv.tok) { saved.erase(saved.begin() + k); break; }
        saved.push_back(std::move(sv));
        size_t total = 0; for (const auto & x : saved) total += x.snap.bytes();
        while (saved.size() > 4 || total > saved_cap) { total -= saved.front().snap.bytes(); saved.erase(saved.begin()); }
        qwfn::assert_that(saved.size() <= 4, "saved sequences: at most four");
        qwfn::assert_that(total <= saved_cap, "saved sequences: within the byte cap");
    }
    // The longest saved sequence that the prompt strictly extends, -1 if none.
    int best_saved(const std::vector<int32_t> & tok, bool has_images) const {
        int best = -1;
        if (has_images) return -1;
        for (size_t k = 0; k < saved.size(); k++) {
            const auto & t = saved[k].tok;
            if (t.size() < tok.size() && std::equal(t.begin(), t.end(), tok.begin()) &&
                (best < 0 || t.size() > saved[best].tok.size())) best = (int) k;
        }
        qwfn::assert_that(best >= -1 && best < (int) saved.size(), "best_saved: an index or -1");
        qwfn::assert_that(best < 0 || saved[best].tok.size() < tok.size(), "best_saved: the prompt strictly extends it");
        return best;
    }
    // The image embeddings inside `consumed`, by position (a hash of each). The
    // pads of any two images are the same tokens, so a prompt whose TOKENS extend
    // the consumed prefix can still carry a different image at the same place --
    // an edited turn -- and that has to re-prefill, not continue.
    std::unordered_map<int32_t, uint64_t> consumed_img;

    // The last reply, kept as TOKENS. Re-tokenizing an assistant turn from its
    // text does not reproduce the tokens that were generated: the framing and
    // the content are encoded separately, and BPE merges across the seam
    // differently ("<think>\n" + "\n</think>" vs "<think>\n\n</think>"). Without
    // this, a replayed conversation never matches and every turn re-prefills the
    // whole history.
    std::vector<int32_t> last_gen;      // tokens generated last time
    std::vector<int32_t> last_prompt;   // the prompt those tokens continued
    json                 last_msgs;     // the message list that produced it
    std::string          last_content, last_reasoning, last_tool_key;
    bool                 last_thinking = true;
    std::string          model_id = "qwen3.8-flash-next";
    std::string          model_file;   // the shard the server was started with, for /props
    uint32_t             n_ctx = 0, n_batch = 0;

    // Live-adjustable defaults (GET/POST /props) and the live counter.
    std::mutex   props_mu;
    std::string  def_effort = "xhigh";
    std::string  dump_dir;                   // POST /props {"dump_requests": DIR}: every request body is written there, for harness debugging
    sampling     preset_think = preset_thinking(), preset_nothink = preset_non_thinking();
    int          def_max_tokens = 0;
    int          def_reasoning_budget = 0;   // max reasoning tokens, 0 = unlimited (see generate)
    live_stats   live;
    // The thread running the current generation, for the stall watchdog's SIGUSR2 probe.
    pthread_t         gen_thread{};
    std::atomic<bool> gen_thread_set{false};
    engine_config cfg;                       // as started, for /props

    // Unique per request: the tool-call ids a harness keys its cards and
    // replays on are derived from it, and two rounds of one conversation used
    // to hand out the same "call_0_<seconds>" id.
    std::atomic<long long> req_seq{0};
    // The draft head's acceptance per draft position, as running means across
    // requests (see drafts_wanted).
    float acc_at[engine::MTP_MAX_DRAFTS] = { 0.85f, 0.80f, 0.75f };

    // A request's prompt, as tokens plus any image embeddings to splice in.
    struct prompt {
        std::vector<int32_t> tok;
        std::vector<std::pair<int32_t, std::vector<float>>> splices;  // offset, emb
    };

};

// The template applies |trim to every message's content and reasoning (llama.cpp:
// C isspace); untrimmed, a harness's trailing "\n\n" lands before <|im_end|>, a
// layout the model never saw. It trims the rendered string, images included, so
// whitespace next to an image is kept: the cut never passes one.
static void trim_like_template(std::string & text, std::vector<std::pair<size_t, pending_img>> * imgs = nullptr) {
    const char * ws = " \t\n\r\v\f";
    size_t b = text.find_first_not_of(ws), e = text.find_last_not_of(ws);
    b = b == std::string::npos ? text.size() : b;
    e = e == std::string::npos ? 0 : e + 1;
    if (imgs && !imgs->empty()) { b = std::min(b, imgs->front().first); e = std::max(e, imgs->back().first); }
    qwfn::assert_that(!imgs || imgs->empty() || imgs->front().first <= imgs->back().first, "trim: images in text order");
    if (b >= e) { text.clear(); if (imgs) for (auto & im : *imgs) im.first = 0; return; }
    qwfn::assert_that(e <= text.size(), "trim: the cut within the text");
    text = text.substr(b, e - b);
    if (imgs) for (auto & im : *imgs) im.first -= b;
}

static std::string trimmed(std::string s) { trim_like_template(s); return s; }

static bool render_content(server & S, const json & content, std::string & text,
                           std::vector<std::pair<size_t, pending_img>> & imgs,
                           std::string & err) {
    qwfn::assert_that(parsed(content), "render_content: a parsed JSON value");
    if (content.is_string()) { text += content.get<std::string>(); return true; }
    if (!content.is_array()) { text += content.dump(); return true; }
    for (const auto & part : content) {
        const std::string ty = part.value("type", "text");
        if (ty == "text") { text += part.value("text", ""); continue; }
        if (ty == "image_url" || ty == "input_image") {
            if (!S.vis.loaded()) { err = "server was started without --mmproj; images are not supported"; return false; }
            std::string url;
            if (part.contains("image_url")) {
                url = part["image_url"].is_string() ? part["image_url"].get<std::string>()
                                                    : part["image_url"].value("url", "");
            } else url = part.value("image_url", "");
            const size_t comma = url.find(",");
            if (url.rfind("data:", 0) != 0 || comma == std::string::npos) {
                err = "only data: image URLs are supported (base64)"; return false;
            }
            std::vector<uint8_t> raw;
            if (!b64_decode(url.substr(comma + 1), raw)) { err = "bad base64 in image_url"; return false; }
            image_u8 img;
            if (!img.load_memory(raw.data(), raw.size(), err)) return false;
            pending_img pi;
            int gw = 0, gh = 0;
            const auto t0 = clk::now();
            // On the CPU: no VRAM is borrowed from the expert tier for this.
            if (!S.vis.encode(img, pi.emb, pi.n_tok, gw, gh, err)) return false;
            fprintf(stderr, "[qwfn-server] image: %zu bytes, %dx%d px -> %dx%d grid, %d tokens, encoded in %.2f s\n",
                    raw.size(), img.nx, img.ny, gw, gh, pi.n_tok, since(t0));
            imgs.emplace_back(text.size(), std::move(pi));   // marker position in text
            continue;
        }
    }
    qwfn::assert_that(imgs.empty() || imgs.back().first <= text.size(), "render_content: image markers within the text");
    return true;
}

// ---- the Anthropic Messages API ---------------------------------------------
// POST /v1/messages is what Claude Code (and the Anthropic SDKs) speak when
// ANTHROPIC_BASE_URL points here. A request is translated into the OpenAI
// shape the chat path handles -- system + messages + tools + tool_choice --
// generated by the same code, and the reply goes back as Anthropic content
// blocks and stream events. Fields with no counterpart here are accepted and
// ignored: cache_control, metadata, context_management, output_config.format,
// the beta tool fields (strict, defer_loading). Server tools (a `type` other
// than "custom", no input_schema) are dropped: nothing here would run them.
static std::string anthropic_text(const json & c) {   // a string, or the text blocks of a list
    qwfn::assert_that(parsed(c), "anthropic_text: a parsed JSON value");
    if (c.is_string()) return c.get<std::string>();
    std::string s;
    if (c.is_array())
        for (const auto & b : c)
            if (b.value("type", "") == "text") { if (!s.empty()) s += "\n\n"; s += b.value("text", ""); }
    qwfn::assert_that(c.is_array() || s.empty(), "anthropic_text: only a list has text blocks");
    return s;
}
// A list of parts that are all text collapses to one string: Claude Code sends
// its newest message as a one-block list carrying cache_control and replays it
// as a plain string, and the prefix continuation compares the messages as sent.
static json compact_parts(const json & parts) {
    qwfn::assert_that(parsed(parts), "compact_parts: a parsed JSON value");
    std::string s; bool all_text = parts.is_array();
    if (all_text) for (const auto & p : parts) { if (p.value("type", "") != "text") { all_text = false; break; } if (!s.empty()) s += "\n\n"; s += p.value("text", ""); }
    qwfn::assert_that(!all_text || parts.is_array(), "compact_parts: only a list collapses");
    return all_text ? json(s) : parts;
}
// One Anthropic content block as an OpenAI content part; null for a block
// with nothing to render here (a tool reference, a server tool's result).
static bool anthropic_part(const json & b, json & part, std::string & err) {
    qwfn::assert_that(parsed(b), "anthropic_part: a parsed JSON value");
    part = nullptr;
    const std::string ty = b.value("type", "");
    if (ty == "text") { part = json{{"type", "text"}, {"text", b.value("text", "")}}; return true; }
    const json src = b.contains("source") && b["source"].is_object() ? b["source"] : json::object();
    qwfn::assert_that(src.is_object(), "anthropic_part: a source object");
    if (ty == "image") {
        if (src.value("type", "") != "base64") { err = "image sources must be base64 (this server does not fetch URLs)"; return false; }
        part = json{{"type", "image_url"}, {"image_url", {{"url", "data:" + src.value("media_type", "image/png") + ";base64," + src.value("data", "")}}}};
        return true;
    }
    if (ty == "document") {
        // A text document reads as text; anything else (a PDF) is named, so the
        // model knows what it was not given, instead of failing the turn.
        const std::string text = src.value("type", "") == "text" ? src.value("data", "")
                               : "[document not available: " + src.value("media_type", "unknown type") + " is not supported by this server]";
        part = json{{"type", "text"}, {"text", text}};
        return true;
    }
    return true;
}
// An assistant turn: text, thinking and tool_use blocks -> content, reasoning_content, tool_calls.
static json anthropic_assistant(const json & c) {
    qwfn::assert_that(parsed(c), "anthropic_assistant: a parsed JSON value");
    std::string text, reasoning; json calls = json::array();
    if (c.is_string()) text = c.get<std::string>();
    else if (c.is_array())
        for (const auto & b : c) {
            const std::string ty = b.value("type", "");
            if (ty == "text")          text += b.value("text", "");
            else if (ty == "thinking") reasoning += b.value("thinking", "");
            else if (ty == "tool_use")
                calls.push_back(json{{"id", b.value("id", "")}, {"type", "function"},
                                     {"function", {{"name", b.value("name", "")},
                                                   {"arguments", (b.contains("input") && b["input"].is_object() ? b["input"] : json::object()).dump()}}}});
        }
    json msg{{"role", "assistant"}, {"content", text}};
    if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
    if (!calls.empty()) msg["tool_calls"] = calls;
    qwfn::assert_that(msg.is_object() && msg.contains("content"), "anthropic_assistant: a message");
    return msg;
}
// A user turn (or a system entry mid-conversation): each tool_result is
// a tool message, the other blocks one message of parts, in the order sent.
static bool anthropic_user(const std::string & role, const json & c, json & msgs, std::string & err) {
    qwfn::assert_that(parsed(c), "anthropic_user: a parsed JSON value");
    qwfn::assert_that(msgs.is_array(), "anthropic_user: a message list");
    if (c.is_string()) { msgs.push_back(json{{"role", role}, {"content", c}}); return true; }
    if (!c.is_array()) { err = "messages[].content must be a string or a list of blocks"; return false; }
    json parts = json::array();
    auto flush = [&]() { if (!parts.empty()) { msgs.push_back(json{{"role", role}, {"content", compact_parts(parts)}}); parts = json::array(); } };
    for (const auto & b : c) {
        if (b.value("type", "") == "tool_result") {
            flush();
            const json rc = b.contains("content") ? b["content"] : json("");
            json content = json::array();
            if (rc.is_string()) content = rc;
            else if (rc.is_array())
                for (const auto & rb : rc) { json p; if (!anthropic_part(rb, p, err)) return false; if (!p.is_null()) content.push_back(p); }
            msgs.push_back(json{{"role", "tool"}, {"tool_call_id", b.value("tool_use_id", "")}, {"content", compact_parts(content)}});
            continue;
        }
        json p;
        if (!anthropic_part(b, p, err)) return false;
        if (!p.is_null()) parts.push_back(p);
    }
    flush();
    return true;
}
// Custom tools as OpenAI functions, and tool_choice.
static void anthropic_tools(const json & a, json & o) {
    qwfn::assert_that(parsed(a), "anthropic_tools: a parsed JSON value");
    qwfn::assert_that(o.is_object(), "anthropic_tools: the request being built");
    if (a.contains("tools") && a["tools"].is_array()) {
        json tools = json::array();
        for (const auto & t : a["tools"]) {
            if (!t.is_object() || !t.contains("input_schema") || (t.contains("type") && t["type"] != "custom")) continue;
            json fn{{"name", t.value("name", "")}};
            if (t.contains("description")) fn["description"] = t["description"];
            fn["parameters"] = t["input_schema"];
            tools.push_back(json{{"type", "function"}, {"function", fn}});
        }
        o["tools"] = tools;
    }
    if (a.contains("tool_choice") && a["tool_choice"].is_object()) {
        const json & tc = a["tool_choice"];
        const std::string ty = tc.value("type", "auto");
        if (ty == "any")       o["tool_choice"] = "required";
        else if (ty == "none") o["tool_choice"] = "none";
        else if (ty == "tool") o["tool_choice"] = json{{"type", "function"}, {"function", {{"name", tc.value("name", "")}}}};
    }
}
// thinking and output_config.effort as reasoning_effort and reasoning_budget (see below).
static void anthropic_effort(const json & a, const std::string & def_effort, int def_budget, json & o) {
    qwfn::assert_that(parsed(a), "anthropic_effort: a parsed JSON value");
    qwfn::assert_that(o.is_object(), "anthropic_effort: the request being built");
    std::string effort;
    if (a.contains("thinking") && a["thinking"].is_object()) {
        const json & t = a["thinking"];
        const std::string ty = t.value("type", "adaptive");
        if (ty == "disabled") effort = "off";
        else if (ty == "enabled") {
            effort = def_effort != "off" ? def_effort : "xhigh";
            if (t.contains("budget_tokens") && t["budget_tokens"].is_number_integer()) {
                const int b = t["budget_tokens"].get<int>();
                if (b > 0) o["reasoning_budget"] = def_budget > 0 ? std::min(b, def_budget) : b;
            }
        }
    }
    const bool on = effort.empty() ? def_effort != "off" : effort != "off";
    if (on && a.contains("output_config") && a["output_config"].is_object() &&
        a["output_config"].contains("effort") && a["output_config"]["effort"].is_string()) {
        const std::string lv = a["output_config"]["effort"].get<std::string>();
        effort = lv == "low" ? "low" : lv == "medium" ? "medium" : lv == "max" ? "xhigh" : effort.empty() ? def_effort : effort;
    }
    if (!effort.empty()) o["reasoning_effort"] = effort;
}
// The request as the chat path takes it. Thinking: "disabled" turns it off;
// "enabled" turns it on at the server's effort (xhigh when the server's default
// is off) with budget_tokens as a cap under the server's own; "adaptive" (what
// Claude Code sends a model it does not know) leaves the server's default.
// output_config.effort low / medium / max maps onto low / medium / xhigh and
// "high" (what Claude Code sends by default) keeps the server's level; it only
// modulates a thinking that is on.
static bool anthropic_to_openai(const json & a, const std::string & def_effort, int def_budget, json & o, std::string & err) {
    qwfn::assert_that(parsed(a), "anthropic_to_openai: a parsed JSON value");
    if (!a.contains("messages") || !a["messages"].is_array()) { err = "messages: field required"; return false; }
    o = json::object();
    json msgs = json::array();
    if (a.contains("system")) {
        const std::string sys = anthropic_text(a["system"]);
        if (!sys.empty()) msgs.push_back(json{{"role", "system"}, {"content", sys}});
    }
    for (const auto & m : a["messages"]) {
        const std::string role = m.value("role", "user");
        const json c = m.contains("content") ? m["content"] : json("");
        if (role == "assistant") { msgs.push_back(anthropic_assistant(c)); continue; }
        if (!anthropic_user(role, c, msgs, err)) return false;
    }
    o["messages"] = msgs;
    anthropic_tools(a, o);
    for (const char * k : {"max_tokens", "temperature", "top_p", "top_k", "stream"}) if (a.contains(k)) o[k] = a[k];
    if (a.contains("stop_sequences")) o["stop"] = a["stop_sequences"];
    anthropic_effort(a, def_effort, def_budget, o);
    qwfn::assert_that(o["messages"].is_array(), "anthropic_to_openai: a message list");
    return true;
}
static const char * anthropic_stop_reason(const std::string & finish, const std::string & stop_seq) {
    qwfn::assert_that(!finish.empty(), "stop reason: a finish reason");
    qwfn::assert_that(finish == "stop" || finish == "length" || finish == "tool_calls", "stop reason: a known finish reason");
    if (finish == "tool_calls") return "tool_use";
    if (finish == "length")     return "max_tokens";
    return stop_seq.empty() ? "end_turn" : "stop_sequence";
}
static std::string tool_use_id(const std::string & id) {   // "call_N_stamp" -> "toolu_N_stamp"
    qwfn::assert_that(!id.empty(), "tool_use_id: a call id");
    std::string r = id.rfind("call_", 0) == 0 ? "toolu_" + id.substr(5) : id;
    qwfn::assert_that(r.size() >= id.size(), "tool_use_id: never shorter");
    return r;
}
// A thinking block carries an opaque signature the client hands back
// unchanged; nothing here verifies it, but the block is well formed with one.
static std::string thinking_signature(const std::string & t) {
    char b[32];
    snprintf(b, sizeof b, "qwfn%016llx", (unsigned long long) hash_bytes(t.data(), t.size()));
    return b;
}

// ---- prompt construction -----------------------------------------------
// Framing is tokenized with parse_special=true, message content with false,
// so a message merely containing the text "<|im_end|>" cannot forge a turn.
// A message's content as JSON for render_content: null (OpenAI tool-call
// replies carry content: null) becomes an empty string.
static json content_of(const json & m) {
    qwfn::assert_that(parsed(m), "content_of: a parsed JSON value");
    qwfn::assert_that(m.is_object() || !m.contains("content"), "content_of: only an object has content");
    if (!m.contains("content") || m["content"].is_null()) return json("");
    return m["content"];
}
// The role of message i, empty past the end.
static std::string role_at(const json & messages, size_t i) {
    qwfn::assert_that(messages.is_array(), "role_at: a message list");
    qwfn::assert_that(i <= messages.size() + 1, "role_at: at most one past the end");
    return i < messages.size() ? messages[i].value("role", "user") : std::string();
}

// Appends tokens to a prompt: framing (special tokens parsed) or plain text.
struct prompt_writer {
    server & S;
    server::prompt & P;
    void app(const std::vector<int32_t> & v) { P.tok.insert(P.tok.end(), v.begin(), v.end()); }
    void sp(const std::string & s) { app(S.vb.encode(s, false, true)); }
    void pl(const std::string & s) { app(S.vb.encode(s, false, false)); }
};

static void open_assistant(prompt_writer & W, bool thinking, const std::string & forced) {
    qwfn::assert_that(forced.empty() || forced.rfind("<tool_call>\n", 0) == 0, "open_assistant: a forced call opens a block");
    W.sp("<|im_start|>assistant\n");
    W.sp(thinking && forced.empty() ? "<think>\n" : "<think>\n\n</think>\n\n");
    // A forced call opening: <tool_call> is one of the model's own tokens
    // (tokenized as framing), the function name is text.
    if (!forced.empty()) { W.sp("<tool_call>\n"); W.pl(forced.substr(strlen("<tool_call>\n"))); }
    qwfn::assert_that(!W.P.tok.empty(), "open_assistant: a prompt");
}
// The template's rendering of an assistant message's tool calls, after its
// content. <tool_call> and </tool_call> are single tokens in this vocabulary
// (user-defined, like <think>): encoded as plain text they become BPE
// fragments the model never produced, so the history reads as foreign.
static void app_tool_calls(prompt_writer & W, const json & tcs, bool has_content) {
    qwfn::assert_that(tcs.is_array(), "app_tool_calls: a list of calls");
    const size_t n0 = W.P.tok.size();
    bool first = true;
    for (const auto & tc : tcs) {
        const json & fn = tc.contains("function") ? tc["function"] : tc;
        const std::string name = fn.value("name", "");
        if (name.empty()) continue;
        if (first && has_content) W.pl("\n\n"); else if (!first) W.pl("\n");
        W.sp("<tool_call>\n");
        std::string body = "<function=" + name + ">\n";
        const json args = tool_args_object(fn);
        for (auto it = args.begin(); it != args.end(); ++it)
            body += "<parameter=" + it.key() + ">\n" + (it.value().is_string() ? it.value().get<std::string>() : it.value().dump()) + "\n</parameter>\n";
        body += "</function>\n";
        W.pl(body);
        W.sp("</tool_call>");
        first = false;
    }
    qwfn::assert_that(W.P.tok.size() >= n0, "app_tool_calls: only appends");
}
// Parts in the order they were sent, as the template renders them:
// the text up to each image, then <|vision_start|> <|image_pad|> x N
// <|vision_end|> in its place. The pads are real tokens, so the
// sequence and the PLE n-gram window stay well formed; only their
// embeddings are replaced (see engine::set_embeddings). A tool result
// carries them the same way (the screenshot a harness's tool took).
static void app_parts(prompt_writer & W, const std::string & text, std::vector<std::pair<size_t, pending_img>> & imgs) {
    qwfn::assert_that(imgs.empty() || W.S.tok_image_pad >= 0, "app_parts: images only with the vision tower");
    size_t at = 0;
    for (auto & im : imgs) {
        if (im.first > at) W.pl(text.substr(at, im.first - at));
        at = im.first;
        W.sp("<|vision_start|>");
        W.P.splices.emplace_back((int32_t) W.P.tok.size(), std::move(im.second.emb));
        W.P.tok.insert(W.P.tok.end(), (size_t) im.second.n_tok, W.S.tok_image_pad);
        W.sp("<|vision_end|>");
    }
    qwfn::assert_that(at <= text.size(), "app_parts: images within the text");
    if (at < text.size()) W.pl(text.substr(at));
}
// A user or tool message. Tool results are grouped into one user turn
// of <tool_response> blocks, as the template does.
static bool render_user_or_tool(prompt_writer & W, const json & m, const std::string & prev_role,
                                const std::string & next_role, std::string & err) {
    qwfn::assert_that(parsed(m), "render_user_or_tool: a parsed JSON value");
    const size_t n0 = W.P.tok.size();
    const std::string role = m.value("role", "user");
    std::string text;
    std::vector<std::pair<size_t, pending_img>> imgs;
    if (!render_content(W.S, content_of(m), text, imgs, err)) return false;
    trim_like_template(text, &imgs);
    if (role == "tool") {
        if (prev_role != "tool") W.sp("<|im_start|>user");
        W.sp("\n<tool_response>\n");
        app_parts(W, text, imgs);
        W.sp("\n</tool_response>");
        if (next_role != "tool") W.sp("<|im_end|>\n");
        return true;
    }
    W.sp("<|im_start|>" + role + "\n");
    app_parts(W, text, imgs);
    W.sp("<|im_end|>\n");
    qwfn::assert_that(W.P.tok.size() > n0, "render_user_or_tool: a turn was appended");
    return true;
}

// How a prompt build that tried to continue the last reply went.
enum class cont { done, failed, rebuild };

// The prompt is the previous prompt plus the exact tokens generated plus the
// new turns after message nprev (our reply); an assistant turn among them is
// not a clean extension, and the prompt is rebuilt from the messages.
static cont continue_reply(prompt_writer & W, const json & messages, size_t nprev, bool thinking,
                           const std::string & forced, std::string & e) {
    server & S = W.S;
    server::prompt & P = W.P;
    qwfn::assert_that(nprev + 2 <= messages.size(), "continue_reply: a reply and a new turn");
    qwfn::assert_that(!S.last_gen.empty(), "continue_reply: a reply to continue");
    P.tok = S.last_prompt;
    P.tok.insert(P.tok.end(), S.last_gen.begin(), S.last_gen.end());
    for (size_t m = nprev + 1; m < messages.size(); m++) {
        const std::string role = messages[m].value("role", "user");
        if (role == "assistant") { P.tok.clear(); break; }   // not a clean extension
        if (!render_user_or_tool(W, messages[m], role_at(messages, m - 1), role_at(messages, m + 1), e)) return cont::failed;
    }
    if (!P.tok.empty()) { open_assistant(W, thinking, forced); return cont::done; }
    P.splices.clear();   // fall through to a full rebuild
    return cont::rebuild;
}
// A harness replays the whole conversation each turn. If this request is
// the previous one plus (our reply, a new user turn), the prompt is the
// previous prompt plus the exact tokens we generated plus the new turn --
// no re-tokenising, so the engine can continue from where it stopped.
static cont try_continuation(prompt_writer & W, const json & messages, bool thinking,
                             const std::string & forced, std::string & e) {
    server & S = W.S;
    qwfn::assert_that(messages.is_array(), "try_continuation: a message list");
    const size_t nprev = S.last_msgs.is_array() ? S.last_msgs.size() : 0;
    if (!(nprev > 0 && !S.last_gen.empty() && messages.size() >= nprev + 2)) return cont::rebuild;
    size_t diff = 0;
    while (diff < nprev && S.last_msgs[diff] == messages[diff]) diff++;
    qwfn::assert_that(diff <= nprev, "try_continuation: the first difference within the previous messages");
    const json & reply = messages[nprev];
    const bool text_same  = content_of(reply).is_string() && same_reply_text(content_of(reply).get<std::string>(), S.last_content);
    const bool calls_same = tool_calls_key(reply.value("tool_calls", json::array())) == S.last_tool_key;
    if (diff == nprev && reply.value("role", "") == "assistant" && text_same && calls_same)
        return continue_reply(W, messages, nprev, thinking, forced, e);
    if (diff < nprev) {
        // The harness changed something it had already sent, and the
        // recurrent state cannot be rewound to the change: the whole
        // history is re-prefilled. Named, so the cause can be found.
        fprintf(stderr, "[qwfn-server] prefix lost: message %zu of %zu (%s) is not what the previous request sent (%zu -> %zu bytes)\n",
                diff, messages.size(), messages[diff].value("role", "?").c_str(), S.last_msgs[diff].dump().size(), messages[diff].dump().size());
    } else {
        fprintf(stderr, "[qwfn-server] prefix lost: the reply at message %zu came back different from what was generated (role %s, text %s, tool calls %s)\n",
                nprev, reply.value("role", "?").c_str(), text_same ? "same" : "differs", calls_same ? "same" : "differ");
    }
    return cont::rebuild;
}
// Every leading system (or developer) message, trimmed and joined by "\n",
// as the template merges them. `first` is the message after them.
static bool leading_system(server & S, const json & messages, std::string & system_msg, size_t & first, std::string & e) {
    qwfn::assert_that(messages.is_array(), "leading_system: a message list");
    first = 0;
    for (; first < messages.size(); first++) {
        const std::string role = messages[first].value("role", "");
        if (role != "system" && role != "developer") break;
        std::string part;
        std::vector<std::pair<size_t, pending_img>> ignore;
        if (!render_content(S, messages[first].value("content", json("")), part, ignore, e)) return false;
        trim_like_template(part);
        if (!part.empty()) system_msg += (system_msg.empty() ? "" : "\n") + part;
    }
    qwfn::assert_that(first <= messages.size(), "leading_system: the first turn within the list");
    return true;
}
// An assistant message of the history.
static bool render_assistant(prompt_writer & W, const json & m, std::string & e) {
    server & S = W.S;
    qwfn::assert_that(parsed(m), "render_assistant: a parsed JSON value");
    const size_t n0 = W.P.tok.size();
    std::string text;
    std::vector<std::pair<size_t, pending_img>> imgs;
    if (!render_content(S, content_of(m), text, imgs, e)) return false;
    trim_like_template(text);
    // The template ignores reasoning that is not a string (a null from some clients).
    std::string rc = m.contains("reasoning_content") && m["reasoning_content"].is_string()
                   ? m["reasoning_content"].get<std::string>() : std::string();
    trim_like_template(rc);
    const json tcs = m.value("tool_calls", json::array());
    // If this is verbatim the reply we just produced, replay the
    // exact tokens so the engine can continue instead of re-prefilling.
    if (!S.last_gen.empty() && same_reply_text(text, S.last_content) && tool_calls_key(tcs) == S.last_tool_key &&
        (rc.empty() || rc == trimmed(S.last_reasoning))) {
        W.sp("<|im_start|>assistant\n");
        W.sp(S.last_thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
        W.P.tok.insert(W.P.tok.end(), S.last_gen.begin(), S.last_gen.end());
        return true;
    }
    // The text between <think> and </think> is one piece, as the template's
    // tokenizer sees it: apart, an empty block's "\n" + "\n" became two
    // tokens where training has "\n\n".
    W.sp("<|im_start|>assistant\n<think>");
    W.pl("\n" + rc + "\n");
    W.sp("</think>\n\n");
    if (!text.empty()) W.pl(text);
    if (tcs.is_array() && !tcs.empty()) app_tool_calls(W, tcs, !text.empty());
    W.sp("<|im_end|>\n");
    qwfn::assert_that(W.P.tok.size() > n0, "render_assistant: a turn was appended");
    return true;
}
static bool build_prompt(server & S, const json & messages, const std::string & effort,
                         bool thinking, const std::string & tools_block, const std::string & forced,
                         server::prompt & P, std::string & e) {
    qwfn::assert_that(P.tok.empty() && P.splices.empty(), "build_prompt: a fresh prompt");
    qwfn::assert_that(messages.is_array(), "build_prompt: a message list");
    prompt_writer W{S, P};
    const cont c = try_continuation(W, messages, thinking, forced, e);
    if (c != cont::rebuild) return c == cont::done;
    std::string system_msg;
    size_t first = 0;
    if (!leading_system(S, messages, system_msg, first, e)) return false;
    W.sp(build_system_block(effort, system_msg, tools_block));
    for (size_t m = first; m < messages.size(); m++) {
        const std::string role = messages[m].value("role", "user");
        if (role == "assistant") {
            if (!render_assistant(W, messages[m], e)) return false;
            continue;
        }
        if (!render_user_or_tool(W, messages[m], m > first ? role_at(messages, m - 1) : std::string(), role_at(messages, m + 1), e)) return false;
    }
    open_assistant(W, thinking, forced);
    return true;
}

// ---- generation ---------------------------------------------------------
struct gen_result {
    std::string reasoning, content, finish = "stop";
    std::string stop_seq;                                       // the stop sequence that ended the reply, when one did
    int n_input = 0, n_cached = 0, n_prompt = 0, n_gen = 0;   // whole prompt, reused prefix, prefilled, generated
    int n_pairs = 0, n_accepted = 0, n_drafted = 0;     // verify steps, drafts accepted, drafts proposed
    double t_prompt = 0, t_gen = 0;
    bool reasoning_budget_hit = false;
};
// Where generate() sends its output: a struct instead of callbacks (rule 9). Streaming: every piece
// goes to the stream driver and tick() keeps bytes moving through a silent stretch. Otherwise a piece
// only checks that the client, when there is one, is still connected; `gone` records that it left.
struct stream_driver;
struct gen_sink {
    stream_driver *          stream = nullptr;
    const httplib::Request * req = nullptr;
    bool                     gone = false;
    bool delta(const std::string & piece, bool is_reasoning);   // false: stop generating
    void tick();                                                // after every prefill batch and generated token
};
using dist_t   = std::vector<std::pair<int32_t, float>>;

// The prefix the engine can continue from: everything it holds, when the
// prompt strictly extends it (images included: the pads match any image of
// the same size, so each one's embedding is compared by hash); else 0. A prompt
// equal to it has no tail to produce logits from, so it re-prefills.
static int32_t prefix_reuse(const server & S, const server::prompt & P) {
    qwfn::assert_that(S.consumed.size() <= (size_t) INT32_MAX, "prefix: the sequence fits an int32");
    if (S.consumed.empty() || P.tok.size() <= S.consumed.size() ||
        !std::equal(S.consumed.begin(), S.consumed.end(), P.tok.begin())) return 0;
    for (const auto & sp : P.splices)
        if (sp.first < (int32_t) S.consumed.size()) {
            const auto it = S.consumed_img.find(sp.first);
            if (it == S.consumed_img.end() ||
                it->second != hash_bytes(sp.second.data(), sp.second.size() * sizeof(float))) return 0;
        }
    qwfn::assert_that(S.consumed.size() < P.tok.size(), "prefix: the prompt strictly extends it");
    return (int32_t) S.consumed.size();
}
// The prompt does not extend the engine's sequence: set that sequence aside if it
// is worth a prefill later, then restore the longest saved sequence the prompt
// extends, or start over.
static void restore_or_reset(server & S, const server::prompt & P) {
    const bool snaps_on = snapshots_on();
    S.saved_cap = (size_t) qwfn::env_int("QWFN_SNAP_MB", 1024, 0, 1 << 24) << 20;
    const auto ts0 = clk::now();
    // Set the current sequence aside if it is worth a prefill later.
    if (snaps_on && S.consumed.size() >= 2048 && S.consumed_img.empty()) {
        server::saved_seq sv; sv.tok = S.consumed;
        if (S.eng.snapshot_save(sv.snap)) S.save_seq(std::move(sv));
    }
    const int best = snaps_on ? S.best_saved(P.tok, !P.splices.empty()) : -1;
    qwfn::assert_that(best >= -1 && best < (int) S.saved.size(), "restore: a saved index or -1");
    S.eng.clear_embeddings();
    S.consumed_img.clear();
    if (best >= 0 && S.eng.snapshot_load(S.saved[best].snap)) {
        S.consumed = S.saved[best].tok;
        fprintf(stderr, "[qwfn-server] restored a saved sequence of %zu tokens (%.0f MB) in %.2f s\n",
                S.consumed.size(), S.saved[best].snap.bytes() / 1e6, std::chrono::duration<double>(clk::now() - ts0).count());
        std::rotate(S.saved.begin() + best, S.saved.begin() + best + 1, S.saved.end());   // most recently used last
    } else {
        S.eng.reset();
        S.consumed.clear();
    }
    qwfn::assert_that(S.consumed_img.empty(), "restore: no image state carried over");
}
// A fresh prefill that shares a long prefix with a saved sequence (the next
// session's system prompt and tool definitions, say) stops at the end of
// the shared part once, and that point is saved: later prompts with the same
// prefix restore it. A recurrent state cannot be cut at an arbitrary point
// after the fact, so the cut has to be made while the prefill passes it.
// Returns that point, 0 for none.
static int32_t shared_prefix_split(const server & S, const server::prompt & P) {
    if (!(snapshots_on() && P.splices.empty() && S.consumed.size() < 2048)) return 0;
    size_t L = 0;
    for (const auto & sv : S.saved) {
        size_t l = 0; const size_t m = std::min(sv.tok.size(), P.tok.size());
        while (l < m && sv.tok[l] == P.tok[l]) l++;
        L = std::max(L, l);
    }
    qwfn::assert_that(L <= P.tok.size(), "split: a common prefix within the prompt");
    bool have = false;   // already saved at exactly this point
    for (const auto & sv : S.saved) have |= sv.tok.size() == L && std::equal(sv.tok.begin(), sv.tok.end(), P.tok.begin());
    const int32_t split_at = L >= 2048 && L + 16 < P.tok.size() && L >= S.consumed.size() + 2 && !have ? (int32_t) L : 0;
    qwfn::assert_that(split_at == 0 || split_at > (int32_t) S.consumed.size(), "split: past what the engine holds");
    return split_at;
}

// From the first engine mutation on, a failure can leave the sequence state
// inconsistent: the layer graphs advance the DeltaNet scan as they run, while
// n_past_ is only bumped once the whole eval succeeds, so a mid-eval error (a
// short expert read, say) leaves some layers ahead of the token count. KV and
// the indexer survive that -- they are positional and get overwritten -- but a
// scan cannot be rewound, so the next request must not continue from it.
struct dirty_guard {
    server & st; const bool & dirty; bool ok = false;
    ~dirty_guard() {
        if (ok || !dirty) return;
        st.last_msgs = json(); st.consumed.clear(); st.consumed_img.clear();
        st.eng.reset(); st.eng.clear_embeddings();
        qwfn::assert_that(st.consumed.empty() && st.consumed_img.empty(), "dirty state: dropped");
        qwfn::assert_that(st.last_msgs.is_null(), "dirty state: no conversation to continue");
    }
};
// Whatever way a generation returns (an eval error, a client that went away), the
// counters must not say "busy" forever.
struct busy_guard { live_stats & L; ~busy_guard() { std::lock_guard<std::mutex> lk(L.mu); L.busy = false; L.prefilling = false; } };

// The image embeddings not yet evaluated go to the engine, their hashes to consumed_img.
static void apply_splices(server & S, const server::prompt & P, bool & state_dirty) {
    qwfn::assert_that(P.splices.size() <= P.tok.size(), "splices: at most one per position");
    for (const auto & sp : P.splices) {
        qwfn::assert_that(sp.first >= 0 && (size_t) sp.first < P.tok.size(), "splices: a position in the prompt");
        if (sp.first < (int32_t) S.consumed.size()) continue;   // already evaluated (and verified above)
        state_dirty = true;
        S.eng.set_embeddings(sp.first, sp.second.data(),
                             (int32_t) (sp.second.size() / 2560));
        S.consumed_img[sp.first] = hash_bytes(sp.second.data(), sp.second.size() * sizeof(float));
    }
}
static void live_begin(server & S, const gen_result & R) {
    qwfn::assert_that(R.n_input >= R.n_cached, "live: the reused prefix within the prompt");
    qwfn::assert_that(R.n_prompt >= 0, "live: a prefill length");
    std::lock_guard<std::mutex> lk(S.live.mu);
    S.live.busy = true; S.live.n_input = R.n_input; S.live.n_cached = R.n_cached; S.live.n_prompt = R.n_prompt; S.live.n_gen = 0;
    S.live.t_prompt = 0; S.live.t_gen = 0; S.live.prompt_done = 0; S.live.prompt_base = 0; S.live.prefilling = R.n_prompt > 0; S.live.t_prompt0 = clk::now(); S.live.n_requests++;
}
static void live_end(server & S, const gen_result & R) {
    qwfn::assert_that(R.n_gen >= 0, "live: a generated count");
    qwfn::assert_that(R.n_accepted <= R.n_drafted, "live: no more drafts accepted than proposed");
    std::lock_guard<std::mutex> lk(S.live.mu);
    S.live.busy = false; S.live.n_gen = R.n_gen; S.live.t_gen = R.t_gen;
    S.live.n_prompt_total += R.n_prompt; S.live.t_prompt_total += R.t_prompt; S.live.n_gen_total += R.n_gen; S.live.t_gen_total += R.t_gen; S.live.n_past = S.eng.n_past();
    S.live.n_input_total += R.n_input; S.live.n_cached_total += R.n_cached;
    S.live.n_pairs_total += R.n_pairs; S.live.n_accepted_total += R.n_accepted; S.live.n_drafted_total += R.n_drafted;
    S.live.last = live_stats::snapshot{R.n_input, R.n_cached, R.n_prompt, R.n_gen, R.n_pairs, R.n_accepted, R.n_drafted, R.t_prompt, R.t_gen, R.finish, now_unix(), true};
    S.live.have_last = true;
}

// One generation's working state, shared by its phases below.
struct gen_state {
    server & S;
    const server::prompt & P;
    sampler & smp;
    gen_result & R;
    std::string & e;
    gen_sink & out;
    const std::vector<std::string> & stops;
    const int reasoning_budget;
    std::vector<int32_t> hist;           // the prompt, then every token generated
    const float * lg = nullptr;          // the logits of the last single-position eval
    int  budget = 0, n = 0;              // tokens allowed, tokens generated
    bool in_think = false;
    // The template puts "\n\n" between </think> and the answer. Harnesses
    // compare strings, so the answer must not start with it.
    bool content_started = false;
    std::string acc;                     // everything emitted, for stop matching
    clk::time_point td;
    // With the draft head loaded (--mtp), a sampled token goes in as a pair
    // with the head's draft for the one after it; the trunk's logits at the
    // first position sample the real next token, and when it is the draft,
    // the second position's logits are already the one after. `tok_in` marks
    // a token that came in that way: in the history, evaluated, sampled
    // from, so nothing to do at the evaluation point but move on.
    int32_t tok = 0, tok_next = -1;
    bool tok_in = false;
    std::vector<int32_t> evald;   // accepted drafts still to emit (already evaluated), after `tok`

    gen_state(server & s_, const server::prompt & p_, sampler & smp_, gen_result & r_, std::string & e_,
              gen_sink & o_, const std::vector<std::string> & st_, int rb_)
        : S(s_), P(p_), smp(smp_), R(r_), e(e_), out(o_), stops(st_), reasoning_budget(rb_), hist(p_.tok) {}
};

// Feed the prompt's tail in batches of n_batch, cut once at `split_at` to save the shared prefix.
static bool prefill(gen_state & G, int32_t split_at) {
    server & S = G.S;
    int32_t fed = (int32_t) S.consumed.size();
    const int32_t fed0 = fed;
    qwfn::assert_that(fed0 >= 0 && fed0 <= (int32_t) G.hist.size(), "prefill: the reused prefix within the prompt");
    const auto tp = clk::now();
    while (fed < (int32_t) G.hist.size()) {
        int32_t take = std::min<int32_t>(S.n_batch, (int32_t) G.hist.size() - fed);
        if (split_at > fed) take = std::min(take, split_at - fed);
        { std::lock_guard<std::mutex> lk(S.live.mu); S.live.prompt_base = fed - fed0; S.live.prompt_done = fed - fed0; }
        G.lg = S.eng.eval(G.hist.data(), fed + take, take, G.e);
        if (!G.lg) return false;
        fed += take;
        // Publish the finished batch now: a shared-prefix snapshot below can take seconds, and the
        // engine's in-batch progress is already cleared.
        { std::lock_guard<std::mutex> lk(S.live.mu); S.live.prompt_base = fed - fed0; S.live.prompt_done = fed - fed0; }
        if (fed == split_at) {
            server::saved_seq sv; sv.tok.assign(G.hist.begin(), G.hist.begin() + split_at);
            if (S.eng.snapshot_save(sv.snap)) {
                fprintf(stderr, "[qwfn-server] saved the shared prefix of %d tokens (%.0f MB)\n", split_at, sv.snap.bytes() / 1e6);
                S.save_seq(std::move(sv));
            }
        }
        { std::lock_guard<std::mutex> lk(S.live.mu); S.live.prompt_base = fed - fed0; S.live.prompt_done = fed - fed0; S.live.n_past = S.eng.n_past(); }
        G.out.tick();
    }
    qwfn::assert_that(fed == (int32_t) G.hist.size(), "prefill: the whole prompt fed");
    G.R.t_prompt = since(tp);
    { std::lock_guard<std::mutex> lk(S.live.mu); S.live.t_prompt = G.R.t_prompt; S.live.prefilling = false; S.live.prompt_done = G.R.n_prompt; S.live.n_past = S.eng.n_past(); }
    return true;
}
// The token budget, and the first token, sampled from the prompt's logits.
static bool decode_begin(gen_state & G, int max_tok, bool thinking) {
    server & S = G.S;
    const int32_t room = (int32_t) S.n_ctx - (int32_t) G.hist.size() - 2;
    G.budget = std::max(0, max_tok > 0 ? std::min(max_tok, room) : room);
    qwfn::assert_that(G.budget >= 0, "decode: a token budget");
    G.in_think = thinking;
    G.content_started = false;
    G.td = clk::now();
    G.n = 0;
    G.tok = G.smp.pick(G.lg, S.eng.n_vocab());
    if (!S.eng.mtp_step(&G.tok, 1, G.e)) return false;
    G.tok_in = false; G.tok_next = -1;
    qwfn::assert_that(G.evald.empty(), "decode: no accepted drafts yet");
    return true;
}

// Draft length: the head's acceptance per draft position, tracked as running
// means; a position is drafted while the chance that everything before it
// and it are accepted beats what the extra position costs the step (~0.4 of
// a single-token step, measured). Capped by --mtp-drafts.
static int drafts_wanted(const server & S) {
    // What an extra draft position costs, relative to a single-token step. 0.7 was
    // measured on QwFN's reference box (4080 SUPER); on the 5060 Ti + 265F with the
    // hybrid CPU path a position costs ~0.4 (replays: 93 / 120 / 156 ms per step at
    // 1 / 2 / 3 drafts), and 0.4 with --mtp-drafts 2 measured best end to end
    // (21.6 tok/s vs 19.5 at 0.7). QWFN_DRAFT_COST overrides.
    static const float draft_cost = (float) qwfn::env_float("QWFN_DRAFT_COST", 0.4, 0, 100);
    // The k that maximises expected tokens per unit of step cost: tokens(k) =
    // 1 + a1 + a1 a2 + ... , cost(k) = 1 + k * draft_cost.
    const int cap = (int) std::min<uint32_t>(S.mtp_drafts, (uint32_t) engine::MTP_MAX_DRAFTS);
    qwfn::assert_that(cap >= 0 && cap <= engine::MTP_MAX_DRAFTS, "drafts: a cap within the head's limit");
    int best = 1; float best_rate = 0.0f, tokens = 1.0f, p = 1.0f;
    for (int k = 1; k <= cap; k++) {
        p *= S.acc_at[k - 1]; tokens += p;
        const float rate = tokens / (1.0f + (float) k * draft_cost);
        if (rate > best_rate) { best_rate = rate; best = k; }
    }
    qwfn::assert_that(best >= 1 && best <= std::max(cap, 1), "drafts: a draft count within the cap");
    return best;
}
// Layer 0's reads for the tokens the next eval will take, issued as soon as they
// are known (engine::spec_layer0): the bonus token before the head runs, the pair
// once drafted, a sampled token before its eval.
static void spec_l0(gen_state & G, const int32_t * toks, int n_toks, int T) {
    qwfn::assert_that(toks != nullptr && n_toks >= 0, "spec_l0: tokens");
    const size_t n0 = G.hist.size();
    for (int k = 0; k < n_toks; k++) G.hist.push_back(toks[k]);
    std::string se; G.S.eng.spec_layer0(G.hist.data(), (int32_t) G.hist.size(), T, se);
    for (int k = 0; k < n_toks; k++) G.hist.pop_back();
    qwfn::assert_that(G.hist.size() == n0, "spec_l0: the history restored");
}
// The drafts of this step. Greedy: the head's argmax chain. At temperature:
// each draft sampled from the head's distribution under the request's
// sampler, the next one chained from that sample, so the acceptance test
// sees the distribution the draft was drawn from.
static bool make_drafts(gen_state & G, bool sampled, std::vector<int32_t> & drafts, std::vector<dist_t> & qs) {
    server & S = G.S;
    const int want = drafts_wanted(S);
    qwfn::assert_that(drafts.empty() && qs.empty(), "drafts: a fresh step");
    if (!sampled) {
        if (!S.eng.mtp_draft_more(want, G.e)) return false;
        for (int k = 0; k < S.eng.mtp_draft_count(); k++) drafts.push_back(S.eng.mtp_draft_k(k));
    } else {
        for (int k = 0; k < want; k++) {
            const float * hl = S.eng.mtp_logits_k(k);
            if (!hl) break;
            auto qd = G.smp.dist(hl, S.eng.n_vocab());
            if (qd.empty()) break;
            const int32_t d = G.smp.sample(qd);
            drafts.push_back(d); qs.push_back(std::move(qd));
            if (k + 1 < want && !S.eng.mtp_draft_next(G.e, d)) return false;
        }
    }
    // No draft past an end-of-generation token, the budget or the context.
    for (size_t k = 0; k < drafts.size(); k++) if (S.vb.is_eog(drafts[k])) { drafts.resize(k); qs.resize(std::min(qs.size(), k)); break; }
    while (!drafts.empty() && (G.n + (int) drafts.size() >= G.budget || (int32_t) G.hist.size() + 1 + (int32_t) drafts.size() > (int32_t) S.n_ctx)) { drafts.pop_back(); if (qs.size() > drafts.size()) qs.pop_back(); }
    qwfn::assert_that(qs.size() <= drafts.size(), "drafts: a head distribution per sampled draft at most");
    return true;
}
// On a rejection at temperature: a draw from the residual max(0, p - q), normalised,
// over p's candidates (p itself when nothing is left).
static int32_t sample_residual(sampler & smp, const dist_t & pd, const dist_t & q) {
    qwfn::assert_that(!pd.empty(), "residual: a trunk distribution");
    std::vector<std::pair<int32_t, float>> res; double sum = 0;
    for (const auto & [t, p] : pd) { const float r = p - sampler::prob_of(q, t); if (r > 0) { res.emplace_back(t, r); sum += r; } }
    if (res.empty() || sum <= 0) return smp.sample(pd);
    qwfn::assert_that(res.size() <= pd.size(), "residual: within the trunk's candidates");
    for (auto & [t, r] : res) r = (float) (r / sum);
    return smp.sample(res);
}
// Verify position by position: accept draft j against the trunk's logits at
// position j; the first rejection ends the step with a token drawn there (in `y`).
// Returns the number accepted.
static int verify_drafts(gen_state & G, bool sampled, const std::vector<int32_t> & drafts, const std::vector<dist_t> & qs, int32_t & y) {
    server & S = G.S;
    sampler & smp = G.smp;
    const int K = (int) drafts.size();
    qwfn::assert_that(K > 0 && K <= engine::MTP_MAX_DRAFTS, "verify: drafts within the head's limit");
    int j = 0;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (j = 0; j < K; j++) {
        const float * lj = S.eng.logits_pos(j);
        bool accept;
        if (!sampled || j >= (int) qs.size()) {
            y = smp.pick(lj, S.eng.n_vocab());
            accept = y == drafts[j];
        } else {
            const auto pd = smp.dist(lj, S.eng.n_vocab());
            const float p_d = sampler::prob_of(pd, drafts[j]), q_d = sampler::prob_of(qs[j], drafts[j]);
            accept = q_d <= 0.0f || p_d >= q_d || U(smp.rng) < (double) p_d / (double) q_d;
            if (!accept) y = sample_residual(smp, pd, qs[j]);
        }
        S.acc_at[j] += 0.05f * ((accept ? 1.0f : 0.0f) - S.acc_at[j]);
        if (!accept) break;
        G.R.n_accepted++;
        smp.gen.push_back(drafts[j]);
    }
    qwfn::assert_that(j >= 0 && j <= K, "verify: accepted within the drafts");
    return j;
}
// One step with drafts: evaluate the token and its drafts together, verify them,
// roll back the rejected tail, and queue the accepted ones for emission.
static bool verify_step(gen_state & G, bool sampled, const std::vector<int32_t> & drafts, const std::vector<dist_t> & qs) {
    server & S = G.S;
    const int K = (int) drafts.size();
    qwfn::assert_that(K > 0, "verify_step: drafts");
    {
        std::vector<int32_t> step(drafts);
        spec_l0(G, step.data(), K, K + 1);
    }
    for (int32_t d : drafts) G.hist.push_back(d);
    if (!S.eng.eval_decode(G.hist.data(), (int32_t) G.hist.size(), K + 1, G.e)) return false;
    G.R.n_pairs++; G.R.n_drafted += K;
    int32_t y = -1;
    const int j = verify_drafts(G, sampled, drafts, qs, y);
    std::vector<int32_t> step_toks(drafts.begin(), drafts.begin() + j);
    if (j == K) {
        y = G.smp.pick(S.eng.logits_pos(K), S.eng.n_vocab());
    } else {
        if (!S.eng.rollback_n(K - j, G.e)) return false;
        G.hist.resize(G.hist.size() - (size_t) (K - j));
    }
    step_toks.push_back(y);
    spec_l0(G, &y, 1, 1);
    if (!S.eng.mtp_step(step_toks.data(), (int) step_toks.size(), G.e)) return false;
    if (j > 0) { G.tok = drafts[0]; G.evald.assign(drafts.begin() + 1, drafts.begin() + j); G.tok_next = y; G.tok_in = true; }
    else       { G.tok = y; }
    qwfn::assert_that(G.hist.size() >= G.P.tok.size(), "verify_step: the history keeps the prompt");
    return true;
}
// Evaluate the sampled token and sample the next one: with the head's drafts when
// it has any, else one position.
static bool advance(gen_state & G) {
    server & S = G.S;
    qwfn::assert_that(!G.tok_in, "advance: the token is not evaluated yet");
    // The draft: the head's argmax when sampling is greedy; at temperature,
    // a sample from the head's own distribution under the request's
    // sampler (speculative sampling: accept with probability
    // min(1, p(d)/q(d)), on rejection draw from the residual p - q, so
    // every emitted token is distributed exactly as the trunk's p).
    // Sampling the draft rather than taking its argmax is what keeps the
    // acceptance near the greedy rate when the trunk itself is sampled.
    const bool sampled = G.smp.cfg.temp > 0.0f && S.eng.mtp_logits() != nullptr;
    std::vector<int32_t> drafts; std::vector<dist_t> qs;
    if (S.eng.mtp_draft_id() >= 0 && !make_drafts(G, sampled, drafts, qs)) return false;
    if (!drafts.empty()) return verify_step(G, sampled, drafts, qs);
    G.lg = S.eng.eval(G.hist.data(), (int32_t) G.hist.size(), 1, G.e);
    if (!G.lg) return false;
    G.tok = G.smp.pick(G.lg, S.eng.n_vocab());
    if (S.eng.mtp_on()) spec_l0(G, &G.tok, 1, 1);   // a lead only when the head runs next
    if (!S.eng.mtp_step(&G.tok, 1, G.e)) return false;
    qwfn::assert_that(G.tok >= 0, "advance: a sampled token");
    return true;
}

// What one decode step does next: carry on with the step, go to the next token,
// stop generating, or fail the request.
enum class dstep { go, next, stop, fail };

// Thinking budget (Qwen's mechanism): tell the model time is up,
// close the think block, and let it answer from what it has.
// Harness timeouts (15 min in one) are shorter than an xhigh
// think on a hard prompt at ~12 tok/s.
static dstep think_budget_cut(gen_state & G, const std::string & piece) {
    server & S = G.S;
    gen_result & R = G.R;
    qwfn::assert_that(G.in_think, "budget cut: inside the think block");
    (G.in_think ? R.reasoning : R.content) += piece;
    if (!G.out.delta(piece, true)) { R.finish = "stop"; return dstep::stop; }
    const std::string cut = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n";
    const auto ct = S.vb.encode(cut, false, true);
    for (int32_t t : ct) G.hist.push_back(t);
    R.reasoning += "\n\n[thinking budget reached]";
    if ((int32_t) G.hist.size() + 1 > (int32_t) S.n_ctx) { R.finish = "length"; return dstep::stop; }
    // The token just sampled has not been evaluated yet (unless it
    // came in as an accepted draft): it goes in with the injected
    // phrase, or the engine ends one position behind the history.
    G.lg = S.eng.eval(G.hist.data(), (int32_t) G.hist.size(), (int32_t) ct.size() + (G.tok_in ? 0 : 1), G.e);
    if (!G.lg) return dstep::fail;
    G.in_think = false;
    R.reasoning_budget_hit = true;
    G.n++;
    G.tok = G.smp.pick(G.lg, S.eng.n_vocab()); G.tok_in = false;
    if (!S.eng.mtp_step(&G.tok, 1, G.e)) return dstep::fail;
    qwfn::assert_that(!G.in_think && R.reasoning_budget_hit, "budget cut: the think block closed");
    return dstep::next;
}
// A stop sequence at the end of the content: cut it off and say which one.
static bool stop_hit(gen_result & R, const std::vector<std::string> & stops) {
    qwfn::assert_that(R.stop_seq.empty(), "stop: no stop sequence hit yet");
    for (const auto & s : stops)
        if (!s.empty() && R.content.size() >= s.size() &&
            R.content.compare(R.content.size() - s.size(), s.size(), s) == 0) {
            R.content.erase(R.content.size() - s.size());
            R.stop_seq = s;
            qwfn::assert_that(!R.stop_seq.empty(), "stop: the sequence that matched");
            return true;
        }
    return false;
}
// A piece of reasoning or content: to the result, to the client, and against the stops.
static dstep emit_piece(gen_state & G, const std::string & piece) {
    gen_result & R = G.R;
    std::string emit = piece;
    if (!G.in_think && !G.content_started) {
        const size_t nb = emit.find_first_not_of(" \t\r\n");
        if (nb == std::string::npos) emit.clear();
        else { emit = emit.substr(nb); G.content_started = true; }
    }
    qwfn::assert_that(emit.size() <= piece.size(), "emit: at most the piece");
    (G.in_think ? R.reasoning : R.content) += emit;
    G.acc += emit;
    if (!emit.empty() && !G.out.delta(emit, G.in_think)) { R.finish = "stop"; return dstep::stop; }
    if (stop_hit(R, G.stops)) { R.finish = "stop"; return dstep::stop; }
    qwfn::assert_that(R.stop_seq.empty(), "emit: no stop sequence hit");
    return dstep::go;
}
// One generated token: record it, emit it, then evaluate it and sample the next.
static dstep decode_step(gen_state & G) {
    server & S = G.S;
    qwfn::assert_that(G.n >= 0 && G.n < G.budget, "decode: within the budget");
    { std::lock_guard<std::mutex> lk(S.live.mu); S.live.n_gen = G.n + 1; S.live.t_gen = since(G.td); S.live.n_past = S.eng.n_past(); }
    if (!G.tok_in) G.hist.push_back(G.tok);
    if (S.vb.is_eog(G.tok)) return dstep::stop;
    const std::string piece = S.vb.piece(G.tok, false);
    if (!G.tok_in) G.smp.gen.push_back(G.tok);

    dstep r = dstep::go;
    if (G.in_think && piece.find("</think>") != std::string::npos) {
        G.in_think = false;
    } else if (G.in_think && G.reasoning_budget > 0 && G.n + 1 >= G.reasoning_budget) {
        r = think_budget_cut(G, piece);
    } else {
        r = emit_piece(G, piece);
    }
    if (r != dstep::go) return r;

    if ((int32_t) G.hist.size() + 1 > (int32_t) S.n_ctx) { G.R.finish = "length"; return dstep::stop; }
    G.out.tick();
    if (G.tok_in) {   // already evaluated with its predecessor: the next accepted draft, then the token after them
        if (!G.evald.empty()) { G.tok = G.evald.front(); G.evald.erase(G.evald.begin()); return dstep::next; }
        G.tok = G.tok_next; G.tok_in = false; return dstep::next;
    }
    qwfn::assert_that(G.hist.size() >= G.P.tok.size(), "decode: the history keeps the prompt");
    return advance(G) ? dstep::next : dstep::fail;
}
// Close the turn so the next request can continue from here. The sampled
// end-of-turn token was appended but never evaluated, so the engine's
// n_past -- not hist.size() -- is what it has actually consumed.
static void commit_turn(server & S, const server::prompt & P, const std::vector<int32_t> & hist, const gen_result & R, bool thinking) {
    qwfn::assert_that(S.eng.n_past() >= 0 && (size_t) S.eng.n_past() <= hist.size(), "commit: evaluated within the history");
    qwfn::assert_that(P.tok.size() <= hist.size(), "commit: the history keeps the prompt");
    S.consumed.assign(hist.begin(), hist.begin() + S.eng.n_past());
    S.last_gen.assign(hist.begin() + P.tok.size(), hist.end());
    // Generation stops before <|im_end|>\n closes the turn; add it so a
    // replayed history lines up with what the engine will next be fed.
    for (int32_t t : S.vb.encode("\n", false, true)) S.last_gen.push_back(t);
    S.last_prompt    = P.tok;
    S.last_content   = R.content;
    S.last_reasoning = R.reasoning;
    S.last_thinking  = thinking;
}

// out.delta(text, is_reasoning) is called as tokens land; false stops.
// out.tick() is called after every prefill batch and every generated token,
// whether or not anything was emitted: a streaming client uses it to keep
// bytes moving through a silent stretch.
static bool generate(server & S, const server::prompt & P, sampler & smp, int max_tok,
                     bool thinking, const std::vector<std::string> & stops,
                     gen_sink & out, gen_result & R, std::string & e, int reasoning_budget = 0) {
    qwfn::assert_that(P.splices.size() <= P.tok.size(), "generate: at most one image per prompt position");
    // Nothing to evaluate means no logits to sample from: reject before touching the engine.
    if (P.tok.empty()) { e = "empty prompt"; return false; }
    // Prefix continuation: only valid when the new prompt strictly extends
    // what the engine already holds.
    if (prefix_reuse(S, P) == 0) restore_or_reset(S, P);
    const int32_t split_at = shared_prefix_split(S, P);
    if ((int32_t) P.tok.size() >= (int32_t) S.n_ctx) {
        e = "context_length_exceeded: prompt of " + std::to_string(P.tok.size())
          + " tokens exceeds the " + std::to_string(S.n_ctx) + " token context";
        return false;
    }
    // From here on this call mutates the engine's sequence state (see dirty_guard).
    bool state_dirty = false;
    dirty_guard dg{S, state_dirty};
    apply_splices(S, P, state_dirty);
    gen_state G(S, P, smp, R, e, out, stops, reasoning_budget);
    if ((int32_t) P.tok.size() > (int32_t) S.consumed.size()) state_dirty = true;   // a tail will be fed
    R.n_input  = (int32_t) G.hist.size();
    R.n_cached = (int32_t) S.consumed.size();
    R.n_prompt = (int32_t) G.hist.size() - R.n_cached;
    live_begin(S, R);
    busy_guard guard{S.live};
    S.gen_thread = pthread_self(); S.gen_thread_set = true;
    smp.gen.clear();
    if (!prefill(G, split_at)) return false;
    if (!decode_begin(G, max_tok, thinking)) return false;
    for (; G.n < G.budget; G.n++) {
        const dstep r = decode_step(G);
        if (r == dstep::fail) return false;
        if (r == dstep::stop) { G.n++; break; }
    }
    if (G.n >= G.budget && G.budget > 0) R.finish = "length";
    R.t_gen = since(G.td);
    R.n_gen = G.n;
    live_end(S, R);
    dg.ok = true;   // finished cleanly: the state describes exactly S.consumed
    commit_turn(S, P, G.hist, R, thinking);
    qwfn::assert_that(S.consumed.size() <= G.hist.size(), "generate: consumed within the history");
    return true;
}

// ---- one chat request, whatever format it came in ---------------------------
// The OpenAI shape is the internal one: /v1/chat/completions takes it as it
// is, /v1/messages (the Anthropic Messages API) is translated into it first.
// From there everything is shared -- the request's fields, the log line, the
// prompt, the generation, the tool-call parsing and the bookkeeping that
// lets the next turn continue the engine's prefix. Only the wire format
// differs, and each route supplies its own.
struct chat_request {
    json msgs, tools;
    std::string effort, tools_block, forced, id, stamp;   // stamp: unique per request; the tool-call ids derive from it
    bool thinking = true, stream = false, timings_per_token = false;
    sampler smp;
    int max_tok = 0, reasoning_budget = 0;
    std::vector<std::string> stops;
    size_t n_images = 0;
};
// What a generation produced: the reply, its tool calls parsed out, and the
// prose before the first call (the whole reply when there is none).
struct chat_result { gen_result R; json calls; std::string text; };

// Qwen's soft switch at the end of a message: 0 = /no_think, 1 = /think, -1 = none.
// The tag is stripped from `txt` when found.
static int strip_soft_switch(std::string & txt) {
    const size_t e = txt.find_last_not_of(" \t\r\n");
    if (e == std::string::npos) return -1;
    std::string t = txt.substr(0, e + 1);
    qwfn::assert_that(!t.empty(), "soft switch: text before the trailing space");
    for (int which = 0; which < 2; which++) {
        const std::string tag = which ? "/think" : "/no_think";
        if (t.size() >= tag.size() && t.compare(t.size() - tag.size(), tag.size(), tag) == 0 &&
            (t.size() == tag.size() || isspace((unsigned char) t[t.size() - tag.size() - 1]))) {
            txt = t.substr(0, t.size() - tag.size());
            while (!txt.empty() && isspace((unsigned char) txt.back())) txt.pop_back();
            qwfn::assert_that(txt.size() < t.size(), "soft switch: the tag removed");
            return which;
        }
    }
    return -1;
}
// Qwen's soft switches, for harnesses that show no thinking toggle: a
// trailing "/think" or "/no_think" in the last user message sets the
// mode for this request and is stripped before the model sees it.
static void apply_soft_switch(server & S, chat_request & Q) {
    qwfn::assert_that(Q.msgs.is_array(), "soft switch: a message list");
    qwfn::assert_that(Q.msgs.size() <= (size_t) INT_MAX, "soft switch: an int-indexable list");
    std::string def_now; { std::lock_guard<std::mutex> lk(S.props_mu); def_now = S.def_effort; }
    for (int m = (int) Q.msgs.size() - 1; m >= 0; m--) {
        if (Q.msgs[m].value("role", "") != "user") continue;
        int r = -1;
        if (Q.msgs[m]["content"].is_string()) {
            std::string c = Q.msgs[m]["content"].get<std::string>();
            if ((r = strip_soft_switch(c)) >= 0) Q.msgs[m]["content"] = c;
        } else if (Q.msgs[m]["content"].is_array()) {
            for (auto & part : Q.msgs[m]["content"])
                if (part.value("type", "") == "text") {
                    std::string c = part.value("text", "");
                    if ((r = strip_soft_switch(c)) >= 0) { part["text"] = c; break; }
                }
        }
        if (r >= 0) Q.effort = r == 0 ? "off" : (def_now != "off" ? def_now : "xhigh");
        break;
    }
}
// Tools: rendered into the system turn unless tool_choice is "none";
// "required" or a named function forces the call's opening.
static void request_tools(const json & body, chat_request & Q) {
    qwfn::assert_that(parsed(body), "request tools: a parsed JSON value");
    Q.tools = body.contains("tools") && body["tools"].is_array() ? body["tools"] : json::array();
    if (body.contains("tool_choice")) {
        const json & tc = body["tool_choice"];
        if (tc.is_string()) {
            if (tc == "none") Q.tools = json::array();
            // "required" opens the block and stops there: a prefix ending in
            // "<function=" ends on a token boundary the model never produces
            // and it continued with a call id ("call_5067") as the name.
            else if (tc == "required" && !Q.tools.empty()) Q.forced = "<tool_call>\n";
        } else if (tc.is_object() && tc.contains("function") && !Q.tools.empty()) {
            Q.forced = "<tool_call>\n<function=" + tc["function"].value("name", "") + ">\n";
        }
    }
    Q.tools_block = render_tools_block(Q.tools);
    qwfn::assert_that(Q.forced.empty() || !Q.tools.empty(), "request tools: a forced call has tools");
}
// Sampling, limits and stops: the server's preset for the mode, then the request's fields.
// False, with `err`, on a field of the wrong type or out of range.
static bool request_sampling(server & S, const json & body, chat_request & Q, std::string & err) {
    qwfn::assert_that(parsed(body), "request sampling: a parsed JSON value");
    qwfn::assert_that(Q.thinking == (Q.effort != "off"), "request sampling: the mode decided");
    { std::lock_guard<std::mutex> lk(S.props_mu); Q.smp.cfg = Q.thinking ? S.preset_think : S.preset_nothink; Q.max_tok = S.def_max_tokens; Q.reasoning_budget = S.def_reasoning_budget; }
    if (!Q.smp.cfg.from_json(body, err)) return false;   // any sampling field in the request wins
    if (body.contains("seed") && body["seed"].is_number_integer())
        Q.smp.rng.seed((unsigned) body["seed"].get<long long>());
    else Q.smp.rng.seed((unsigned) std::chrono::steady_clock::now().time_since_epoch().count());
    // -1 and 0: until the context is full.
    if (body.contains("max_completion_tokens")) { if (!num_field(body, "max_completion_tokens", -1, INT_MAX, Q.max_tok, err)) return false; }
    else if (!num_field(body, "max_tokens", -1, INT_MAX, Q.max_tok, err)) return false;
    Q.timings_per_token = body.value("timings_per_token", false);
    return true;
}
static bool parse_chat_request(server & S, const json & body, const char * id_prefix, chat_request & Q, std::string & err) {
    qwfn::assert_that(id_prefix != nullptr, "chat request: an id prefix");
    if (!body.contains("messages") || !body["messages"].is_array()) { err = "messages is required"; return false; }

    // Thinking: either OpenAI-ish reasoning_effort, or the Qwen template's
    // own chat_template_kwargs.enable_thinking.
    { std::lock_guard<std::mutex> lk(S.props_mu); Q.effort = body.value("reasoning_effort", S.def_effort); }
    if (body.contains("chat_template_kwargs")) {
        const auto & k = body["chat_template_kwargs"];
        if (k.contains("enable_thinking") && !k["enable_thinking"].get<bool>()) Q.effort = "off";
    }
    if (!effort_valid(Q.effort)) { err = "reasoning_effort must be xhigh|medium|low|off"; return false; }

    Q.msgs = body["messages"];
    apply_soft_switch(S, Q);
    Q.thinking = Q.effort != "off";

    if (!request_sampling(S, body, Q, err)) return false;
    request_tools(body, Q);
    if (!num_field(body, "reasoning_budget", -1, INT_MAX, Q.reasoning_budget, err)) return false;
    if (body.contains("stop")) {
        if (body["stop"].is_string()) Q.stops.push_back(body["stop"].get<std::string>());
        else for (const auto & s : body["stop"]) Q.stops.push_back(s.get<std::string>());
    }
    Q.stream = body.value("stream", false);
    Q.stamp = std::to_string(now_unix()) + "-" + std::to_string(++S.req_seq);
    Q.id = std::string(id_prefix) + Q.stamp;
    for (const auto & m : Q.msgs)
        if (m.contains("content") && m["content"].is_array())
            for (const auto & part : m["content"]) { const std::string ty = part.value("type", ""); if (ty == "image_url" || ty == "input_image") Q.n_images++; }
    qwfn::assert_that(!Q.id.empty() && Q.msgs.is_array(), "chat request: an id and a message list");
    return true;
}
// One line per request, so a harness's exact ask is visible in the log.
static void log_request(server & S, const httplib::Request & req, const chat_request & Q, const json & body, const char * kind) {
    qwfn::assert_that(kind != nullptr, "log_request: a kind");
    qwfn::assert_that(!Q.id.empty(), "log_request: a request id");
    fprintf(stderr, "[qwfn-server] request from %s%s: %zu messages, %zu images, stream=%d, max_tokens=%d, effort=%s, budget=%d, temp=%.2f, timings_per_token=%d, tools=%zu%s, keys:",
            req.remote_addr.c_str(), kind, Q.msgs.size(), Q.n_images, (int) Q.stream, Q.max_tok, Q.effort.c_str(), Q.reasoning_budget, Q.smp.cfg.temp, (int) Q.timings_per_token,
            Q.tools.size(), Q.forced.empty() ? "" : " (forced)");
    for (auto it = body.begin(); it != body.end(); ++it) fprintf(stderr, " %s", it.key().c_str());
    fprintf(stderr, "\n");
    std::string dump; { std::lock_guard<std::mutex> lk(S.props_mu); dump = S.dump_dir; }
    if (!dump.empty()) {
        const std::string path = dump + "/" + Q.id + ".json";
        if (FILE * f = fopen(path.c_str(), "w")) { fwrite(req.body.data(), 1, req.body.size(), f); fclose(f); }
        else fprintf(stderr, "[qwfn-server] cannot write %s\n", path.c_str());
    }
}
static bool too_long(const std::string & e) { return e.rfind("context_length_exceeded", 0) == 0; }
static json openai_usage(const gen_result & R) {
    return json{{"prompt_tokens", R.n_input}, {"prompt_tokens_details", {{"cached_tokens", R.n_cached}}},
                {"completion_tokens", R.n_gen}, {"total_tokens", R.n_input + R.n_gen}};
}
static void finish_request(server & S, const chat_request & Q, chat_result & C) {
    qwfn::assert_that(Q.msgs.is_array(), "finish_request: a message list");
    C.text = parse_tool_calls(Q.forced + C.R.content, Q.tools, C.calls);
    S.last_tool_key = tool_calls_key(C.calls); S.last_content = C.calls.empty() ? C.R.content : C.text;
    if (!C.calls.empty()) C.R.finish = "tool_calls";
    S.last_msgs = Q.msgs;
    qwfn::assert_that(C.calls.is_array(), "finish_request: a list of calls");
}
static void log_done(const chat_request & Q, const gen_result & R) {
    qwfn::assert_that(!Q.id.empty(), "log_done: a request id");
    qwfn::assert_that(R.n_accepted <= R.n_drafted, "log_done: no more drafts accepted than proposed");
    fprintf(stderr, "[qwfn-server] %s: prompt %d tok (%d cached) %.1f tok/s | generated %d tok (%zu reasoning chars%s) in %.1f s, %.1f tok/s, finish %s%s\n",
            Q.id.c_str(), R.n_prompt, R.n_cached, R.t_prompt > 0 ? R.n_prompt / R.t_prompt : 0.0, R.n_gen, R.reasoning.size(),
            R.reasoning_budget_hit ? ", budget hit" : "", R.t_gen, R.t_gen > 0 ? R.n_gen / R.t_gen : 0.0, R.finish.c_str(),
            R.n_pairs ? (" | drafts: " + std::to_string(R.n_accepted) + " of " + std::to_string(R.n_drafted) + " accepted over " + std::to_string(R.n_pairs) + " steps").c_str() : "");
}
// A client that times out and retries would otherwise queue its retry behind
// the answer it abandoned, which then times out as well: stop when it leaves.
static bool run_batch(server & S, chat_request & Q, const server::prompt & P, chat_result & C, std::string & err,
                      const httplib::Request & req) {
    qwfn::assert_that(!Q.stream, "run_batch: a request without streaming");
    gen_sink out; out.req = &req;
    if (!generate(S, P, Q.smp, Q.max_tok, Q.thinking, Q.stops, out, C.R, err, Q.reasoning_budget)) {
        S.last_msgs = json();   // cache is no longer trustworthy
        fprintf(stderr, "[qwfn-server] %s: generation failed after %d tokens: %s\n", Q.id.c_str(), C.R.n_gen, err.c_str());
        return false;
    }
    if (out.gone) {
        S.last_msgs = json();   // the engine holds a partial answer no client will send back
        fprintf(stderr, "[qwfn-server] %s: client disconnected, generation cancelled after %d tokens\n", Q.id.c_str(), C.R.n_gen);
        err = "client disconnected";
        return false;
    }
    finish_request(S, Q, C);
    log_done(Q, C.R);
    qwfn::assert_that(C.R.n_gen >= 0, "run_batch: a generated count");
    return true;
}

// A streamed generation, format-agnostic. The route supplies the writers;
// the driver coalesces reasoning, splits tool blocks out of the content
// while they are written (see tool_streamer), keeps bytes moving through a
// silent stretch, and does the bookkeeping when the reply is done.
// The events go to the route's writer, exactly one of `oai` and `ant` (a struct, not callbacks: rule 9).
struct openai_sse;
struct anthropic_sse;
struct stream_events {
    openai_sse *    oai = nullptr;
    anthropic_sse * ant = nullptr;
    bool write_failed = false;                      // the route sets it when a write fails...
    clk::time_point last_write = clk::now();        // ...and stamps this on every write
    bool reasoning(const std::string & t);
    bool content(const std::string & t);
    bool tool_delta(const json & d);                // an OpenAI tool_calls delta: with an id it opens a call, else it extends the arguments
    bool tool_end();                                // the open call's block is complete
    bool keepalive();                               // nothing has gone out for 15 s
    bool one_writer() const { return (oai != nullptr) != (ant != nullptr); }
};
bool tool_streamer::emit(const json & d) { return ev.tool_delta(d); }
struct stream_outcome { chat_result C; bool ok = false; std::string aborted, err; };

// The per-stream state of run_stream: the held UTF-8 tail, the reasoning
// coalescer and the content / tool-call splitter.
struct stream_driver {
    const chat_request & Q;
    stream_events & ev;
    std::string held;   // incomplete UTF-8 tail of the previous delta
    // Reasoning deltas are coalesced (every 100 ms or 16 tokens): a
    // 15-minute think is ~13,000 tokens, and a UI re-rendering its
    // reasoning block per chunk is what stalls the socket.
    std::string rbuf; int rcount = 0; clk::time_point rlast = clk::now();
    // Tool calls are never streamed as text: content before the first
    // <tool_call> streams normally (holding back a possible partial tag);
    // a block streams as tool-call deltas while it is written and the text
    // after it streams as content.
    std::string tacc; size_t temitted = 0; bool tool_mode = false;
    int n_calls = 0;
    std::unique_ptr<tool_streamer> ts;
    bool with_tools = false;
    const std::string mark = TC_MARK;

    stream_driver(const chat_request & q, stream_events & e)
        : Q(q), ev(e), tacc(q.forced), tool_mode(!q.forced.empty()), with_tools(q.tools.is_array() && !q.tools.empty()) {
        qwfn::assert_that(tool_mode == !tacc.empty(), "stream: tool mode exactly when a call is forced");
        qwfn::assert_that(!tool_mode || with_tools, "stream: a forced call has tools");
        if (tool_mode) open_streamer(0);
    }
    // A silent stretch -- a long prefill, a tool call being written, a
    // held-back parameter -- still has to put bytes on the wire: Unsloth
    // Studio's proxy reads the stream with a 300 s per-read timeout and
    // reports "Timeout waiting for custom response" when nothing arrives,
    // with the generation still running here; Claude Code drops a stream
    // silent for 300 s the same way.
    void tick() {
        qwfn::assert_that(ev.one_writer(), "stream: a writer for the keepalive");
        qwfn::assert_that(temitted <= tacc.size(), "stream: emitted within the content");
        if (ev.write_failed || since(ev.last_write) < 15.0) return;
        if (!ev.keepalive()) ev.write_failed = true;
        ev.last_write = clk::now();
    }
    std::string complete(std::string piece) {   // returns what may be emitted now
        piece = held + piece; held.clear();
        const size_t t = utf8_incomplete_tail(piece);
        qwfn::assert_that(t <= piece.size(), "stream: a held tail within the piece");
        if (t) { held = piece.substr(piece.size() - t); piece.erase(piece.size() - t); }
        qwfn::assert_that(held.size() <= 3, "stream: at most 3 bytes held");
        return piece;
    }
    bool flush_reasoning() {
        if (rbuf.empty()) return true;
        const bool ok = ev.reasoning(rbuf);
        rbuf.clear(); rcount = 0; rlast = clk::now();
        qwfn::assert_that(rbuf.empty() && rcount == 0, "stream: reasoning flushed");
        qwfn::assert_that(ev.one_writer(), "stream: a writer for the reasoning");
        return ok;
    }
    void open_streamer(size_t block_start) {
        qwfn::assert_that(!ts, "stream: one tool call open at a time");
        qwfn::assert_that(block_start <= tacc.size(), "stream: the block within the content");
        ts.reset(new tool_streamer(Q.tools, block_start, n_calls, "call_" + std::to_string(n_calls) + "_" + Q.stamp, ev));
        n_calls++;
    }
    // Outside a tool block: emit text up to a real block start (then return -1:
    // go on in tool mode), or everything except a tail that could still become
    // the marker (1). 0: a write failed.
    int drain_text() {
        qwfn::assert_that(!tool_mode, "stream: outside a tool block");
        qwfn::assert_that(temitted <= tacc.size(), "stream: emitted within the content");
        // A real block start, or a bare tag that is still undecided
        // (not enough characters yet to tell it from the marker)?
        size_t p = tacc.find(TC_OPEN, temitted); bool undecided = false;
        while (p != std::string::npos) {
            if (tacc.size() - p < mark.size()) { undecided = true; break; }
            if (tacc.compare(p, mark.size(), mark) == 0) break;
            p = tacc.find(TC_OPEN, p + 1);          // prose: keep looking
        }
        if (p != std::string::npos && !undecided) {
            std::string t = tacc.substr(temitted, p - temitted);
            while (!t.empty() && (t.back() == '\n' || t.back() == ' ')) t.pop_back();
            if (!t.empty() && !ev.content(t)) return 0;
            temitted = p; tool_mode = true; open_streamer(p);
            return -1;
        }
        // Emit everything except a tail that could still become the marker.
        size_t hold = undecided ? tacc.size() - p : 0;
        if (!undecided)
            for (size_t k = std::min(mark.size() - 1, tacc.size() - temitted); k > 0; k--)
                if (mark.compare(0, k, tacc, tacc.size() - k, k) == 0) { hold = k; break; }
        const std::string t = tacc.substr(temitted, tacc.size() - temitted - hold);
        if (!t.empty() && !ev.content(t)) return 0;
        temitted = tacc.size() - hold;
        return 1;
    }
    bool drain_content() {
        qwfn::assert_that(temitted <= tacc.size(), "stream: emitted within the content");
        if (!with_tools) return ev.content(tacc.substr(temitted)) && (temitted = tacc.size(), true);
        // A pass that does not return consumes a whole tool block: the content bounds the passes.
        for (size_t pass = 0, cap = tacc.size() + 1; pass < cap; pass++) {
            if (!tool_mode) { const int r = drain_text(); if (r >= 0) return r == 1; }
            qwfn::assert_that(tool_mode && ts != nullptr, "stream: a tool call open in tool mode");
            if (!ts->feed(tacc)) return false;
            const size_t close = tacc.find(TC_CLOSE, temitted);
            if (close == std::string::npos) return true;
            if (ts->opened && ts->st != tool_streamer::CLOSED && !ts->close_object()) return false;
            if (ts->opened && !ev.tool_end()) return false;
            ts.reset();
            temitted = close + 12;
            tool_mode = false;   // text after a block (the template forbids it, models do it) streams as content
        }
        qwfn::assert_that(false, "stream: every pass consumes a tool block");
        return false;
    }
    bool on_piece(const std::string & piece_in, bool is_reasoning) {
        const std::string piece = complete(piece_in);
        if (piece.empty()) return true;
        qwfn::assert_that(rcount >= 0 && rcount < 16, "stream: a reasoning batch below its size");
        if (is_reasoning) {
            rbuf += piece; rcount++;
            if (rcount < 16 && since(rlast) < 0.1) return true;
            return flush_reasoning();
        }
        if (!flush_reasoning()) return false;
        tacc += piece;
        qwfn::assert_that(!tacc.empty(), "stream: content to drain");
        return drain_content();
    }
    // The reply is done: flush what is left.
    void finish() {
        flush_reasoning();
        if (tool_mode && ts) {
            // The reply ended inside a block. Closed with </function>:
            // a complete call, the batch parser agrees. Cut off mid-way:
            // leave the fragment as it is -- a client that cannot parse
            // the arguments describes the call instead of running it,
            // which is the right outcome for a call the model never
            // finished. Never a name: plain text.
            ts->feed(tacc);
            if (!ts->opened && temitted < tacc.size()) ev.content(tacc.substr(temitted));
            temitted = tacc.size();
        }
        if (temitted < tacc.size()) { ev.content(tacc.substr(temitted)); temitted = tacc.size(); }
        qwfn::assert_that(temitted == tacc.size(), "stream: all content emitted");
        qwfn::assert_that(rbuf.empty(), "stream: all reasoning emitted");
    }
};
bool gen_sink::delta(const std::string & piece, bool is_reasoning) {
    qwfn::assert_that(!(stream && req), "gen sink: a stream or a client check, not both");
    qwfn::assert_that(req != nullptr || !gone, "gen sink: only a client check records a client leaving");
    if (stream) return stream->on_piece(piece, is_reasoning);
    if (req) return !(gone = req->is_connection_closed());
    return true;
}
void gen_sink::tick() {
    qwfn::assert_that(!(stream && req), "gen sink: a stream or a client check, not both");
    qwfn::assert_that(req != nullptr || !gone, "gen sink: only a client check records a client leaving");
    if (stream) stream->tick();
}
static stream_outcome run_stream(server & S, chat_request & Q, const server::prompt & P, stream_events & ev) {
    qwfn::assert_that(Q.stream, "run_stream: a streaming request");
    stream_outcome out;
    gen_result & R = out.C.R;
    stream_driver D(Q, ev);
    gen_sink sink; sink.stream = &D;
    try {
        out.ok = generate(S, P, Q.smp, Q.max_tok, Q.thinking, Q.stops, sink, R, out.err, Q.reasoning_budget);
        if (out.ok) D.finish();
    } catch (const std::exception & ex) {
        out.aborted = ex.what();
    } catch (...) {
        out.aborted = "unknown exception";
    }
    if (!out.aborted.empty()) {
        // The engine's history is unknown from here: drop the prefix cache
        // so the next request starts clean instead of running on a
        // desynchronised state.
        fprintf(stderr, "[qwfn-server] %s: stream aborted by an exception after %d generated tokens: %s\n", Q.id.c_str(), R.n_gen, out.aborted.c_str());
        S.last_msgs = json(); S.consumed.clear(); S.eng.reset(); S.eng.clear_embeddings();
        { std::lock_guard<std::mutex> lg(S.live.mu); S.live.busy = false; }
        return out;
    }
    if (out.ok) finish_request(S, Q, out.C); else S.last_msgs = json();
    if (ev.write_failed)
        fprintf(stderr, "[qwfn-server] %s: client stopped reading after %d prompt + %d generated tokens (%.0f s); stream dropped\n",
                Q.id.c_str(), R.n_prompt, R.n_gen, R.t_prompt + R.t_gen);
    else if (!out.ok)
        fprintf(stderr, "[qwfn-server] %s: generation failed after %d tokens: %s\n", Q.id.c_str(), R.n_gen, out.err.c_str());
    else
        log_done(Q, R);
    qwfn::assert_that(out.aborted.empty(), "run_stream: not aborted");
    return out;
}

// ---- HTTP ---------------------------------------------------------------
static void fail(httplib::Response & res, int code, const std::string & msg,
                 const std::string & type = "invalid_request_error") {
    qwfn::assert_that(code >= 400 && code < 600, "error reply: an error status");
    qwfn::assert_that(!type.empty(), "error reply: an error type");
    res.status = code;
    res.set_content(json{{"error", {{"message", msg}, {"type", type}}}}.dump(2, ' ', false, json::error_handler_t::replace),
                    "application/json");
}

// ---- the OpenAI chat API ------------------------------------------------------
static void openai_batch(server & S, const httplib::Request & req, httplib::Response & res,
                         chat_request & Q, const server::prompt & P, std::string & e) {
    qwfn::assert_that(!Q.stream, "openai batch: a request without streaming");
    if (req.is_connection_closed()) { fprintf(stderr, "[qwfn-server] %s: client left while queued, skipped\n", Q.id.c_str()); return; }
    chat_result C;
    if (!run_batch(S, Q, P, C, e, req)) {
        fail(res, too_long(e) ? 400 : 500, e, too_long(e) ? "invalid_request_error" : "server_error");
        return;
    }
    json msg{{"role", "assistant"}, {"content", C.calls.empty() ? json(C.R.content) : (C.text.empty() ? json(nullptr) : json(C.text))}};
    if (!C.calls.empty()) msg["tool_calls"] = C.calls;
    if (!C.R.reasoning.empty()) msg["reasoning_content"] = C.R.reasoning;
    qwfn::assert_that(msg.contains("content"), "openai batch: a message");
    res.set_content(json{
        {"id", Q.id}, {"object", "chat.completion"}, {"created", now_unix()},
        {"model", S.model_id},
        {"choices", json::array({ json{
            {"index", 0}, {"message", msg}, {"finish_reason", C.R.finish}} })},
        {"usage", openai_usage(C.R)},
        {"timings", S.live.timings()}
    }.dump(2, ' ', false, json::error_handler_t::replace), "application/json");
}
// The OpenAI stream's writer: chat.completion.chunk events.
struct openai_sse {
    server & S;
    const chat_request & Q;
    httplib::DataSink & sink;
    stream_events & ev;

    bool send(const json & j) {
        const std::string s = "data: " + json_dump(j) + "\n\n";
        const bool ok = sink.write(s.data(), s.size());
        ev.last_write = clk::now();
        if (!ok) ev.write_failed = true;
        qwfn::assert_that(ok || ev.write_failed, "openai sse: a failed write is recorded");
        qwfn::assert_that(!s.empty(), "openai sse: an event");
        return ok;
    }
    json chunk(const json & d, const json & finish) const {
        return json{{"id", Q.id}, {"object", "chat.completion.chunk"}, {"created", now_unix()}, {"model", S.model_id},
                    {"choices", json::array({ json{{"index", 0}, {"delta", d}, {"finish_reason", finish}} })}};
    }
    bool emit(const json & d) {
        qwfn::assert_that(d.is_object(), "openai sse: a delta object");
        json c = chunk(d, nullptr);
        if (Q.timings_per_token) c["timings"] = S.live.timings();   // live tok/s per chunk
        qwfn::assert_that(c.contains("choices"), "openai sse: a chunk");
        return send(c);
    }
    // An SSE comment line is ignored by every client (Studio relays
    // non-data lines untouched).
    bool keepalive() { static const std::string ka = ": keepalive\n\n"; return sink.write(ka.data(), ka.size()); }
};
// The streamed reply as chat.completion.chunk events, then [DONE].
static void openai_stream(server & S, chat_request & Q, const server::prompt & P, httplib::DataSink & sink) {
    qwfn::assert_that(Q.stream, "openai stream: a streaming request");
    stream_events ev;
    openai_sse W{S, Q, sink, ev};
    ev.oai = &W;
    W.send(W.chunk(json{{"role", "assistant"}}, nullptr));

    stream_outcome out = run_stream(S, Q, P, ev);
    if (!out.aborted.empty()) {
        W.send(json{{"error", {{"message", "stream aborted: " + out.aborted}, {"type", "server_error"}}}});
    } else if (!out.ok) {
        W.send(json{{"error", {{"message", out.err}, {"type", "server_error"}}}});
    } else {
        json last = W.chunk(json::object(), out.C.R.finish);
        last["usage"] = openai_usage(out.C.R);
        last["timings"] = S.live.timings();
        W.send(last);
    }
    const std::string done = "data: [DONE]\n\n";
    sink.write(done.data(), done.size());
    sink.done();
    qwfn::assert_that(!done.empty(), "openai stream: the terminator written");
}
static void handle_chat(server & S, const httplib::Request & req, httplib::Response & res) {
    json body;
    try { body = json::parse(req.body); }
    catch (const std::exception & ex) { fail(res, 400, std::string("bad JSON: ") + ex.what()); return; }
    qwfn::assert_that(parsed(body), "chat: a parsed body");
    auto Q = std::make_shared<chat_request>();
    std::string e;
    if (!parse_chat_request(S, body, "chatcmpl-", *Q, e)) { fail(res, 400, e); return; }
    log_request(S, req, *Q, body, "");

    // One generation at a time. For a streamed response the lock must
    // outlive this handler: the chunked provider runs after it returns, and
    // a second request arriving mid-stream (a harness generating a title,
    // the next turn) must wait, not enter the engine. So the lock is shared
    // with the provider and released when the stream is done.
    auto lk = std::make_shared<std::unique_lock<std::mutex>>(S.mu);
    qwfn::assert_that(lk->owns_lock(), "chat: the engine lock held");
    auto P = std::make_shared<server::prompt>();
    if (!build_prompt(S, Q->msgs, Q->effort, Q->thinking, Q->tools_block, Q->forced, *P, e)) { fail(res, 400, e); return; }

    if (!Q->stream) { openai_batch(S, req, res, *Q, *P, e); return; }

    // Streaming: the provider owns the engine lock until it is done.
    res.set_header("Cache-Control", "no-cache");
    server * sp = &S;
    res.set_chunked_content_provider("text/event-stream",
        [sp, Q, P, lk](size_t, httplib::DataSink & sink) mutable {
            openai_stream(*sp, *Q, *P, sink);
            lk->unlock();
            // true: the body is complete (done() wrote the trailer). false
            // would be "cancelled" to httplib, which then closes a keep-alive
            // connection the client is about to reuse.
            return true;
        });
}

// ---- the Anthropic Messages API: Claude Code --------------------------------
// ANTHROPIC_BASE_URL=http://127.0.0.1:8080 (with any ANTHROPIC_AUTH_TOKEN and
// an ANTHROPIC_MODEL; the model name is echoed, not checked). Thinking comes
// back as thinking blocks, tool calls as tool_use blocks streamed as
// input_json_delta while the model writes them, usage as input / cached /
// output tokens. See anthropic_to_openai for what is mapped and dropped.
static void fail_anthropic(httplib::Response & res, int code, const std::string & msg,
                           const std::string & type = "invalid_request_error") {
    qwfn::assert_that(code >= 400 && code < 600, "error reply: an error status");
    qwfn::assert_that(!type.empty(), "error reply: an error type");
    res.status = code;
    res.set_content(json{{"type", "error"}, {"error", {{"type", type}, {"message", msg}}}}.dump(2, ' ', false, json::error_handler_t::replace),
                    "application/json");
}
// Claude Code compacts its conversation when a rejection carries the API's
// own wording for a prompt over the context.
static std::string too_long_msg(const server & S, size_t n_tok) {
    return "prompt is too long: " + std::to_string(n_tok) + " tokens > " + std::to_string(S.n_ctx) + " maximum";
}
static json anthropic_usage(const gen_result & R) {
    return json{{"input_tokens", R.n_prompt}, {"cache_creation_input_tokens", 0}, {"cache_read_input_tokens", R.n_cached}, {"output_tokens", R.n_gen}};
}
static json anthropic_message(const chat_request & Q, const std::string & model, const chat_result & C) {
    qwfn::assert_that(C.calls.is_array(), "anthropic message: a list of calls");
    json content = json::array();
    if (!C.R.reasoning.empty())
        content.push_back(json{{"type", "thinking"}, {"thinking", C.R.reasoning}, {"signature", thinking_signature(C.R.reasoning)}});
    const std::string & text = C.calls.empty() ? C.R.content : C.text;
    if (!text.empty()) content.push_back(json{{"type", "text"}, {"text", text}});
    for (const auto & tc : C.calls)
        content.push_back(json{{"type", "tool_use"}, {"id", tool_use_id(tc["id"].get<std::string>())},
                               {"name", tc["function"]["name"]}, {"input", tool_args_object(tc["function"])}});
    qwfn::assert_that(content.size() <= C.calls.size() + 2, "anthropic message: thinking, text and the calls");
    return json{{"id", Q.id}, {"type", "message"}, {"role", "assistant"}, {"model", model}, {"content", content},
                {"stop_reason", anthropic_stop_reason(C.R.finish, C.R.stop_seq)},
                {"stop_sequence", C.R.stop_seq.empty() ? json(nullptr) : json(C.R.stop_seq)},
                {"usage", anthropic_usage(C.R)}};
}
// The translated request, parsed as the chat path parses its own.
static bool parse_anthropic(server & S, const httplib::Request & req, httplib::Response & res, const char * id_prefix, json & body, chat_request & Q) {
    qwfn::assert_that(id_prefix != nullptr, "anthropic: an id prefix");
    try { body = json::parse(req.body); }
    catch (const std::exception & ex) { fail_anthropic(res, 400, std::string("bad JSON: ") + ex.what()); return false; }
    std::string req_effort; int req_budget;
    { std::lock_guard<std::mutex> lk(S.props_mu); req_effort = S.def_effort; req_budget = S.def_reasoning_budget; }
    qwfn::assert_that(effort_valid(req_effort), "anthropic: a valid default effort");
    json oai; std::string e;
    if (!anthropic_to_openai(body, req_effort, req_budget, oai, e)) { fail_anthropic(res, 400, e); return false; }
    if (!parse_chat_request(S, oai, id_prefix, Q, e)) { fail_anthropic(res, 400, e); return false; }
    return true;
}

// The Anthropic stream's writer: events, and content blocks one open at a
// time, indexed in the order opened.
struct anthropic_sse {
    httplib::DataSink & sink;
    stream_events & ev;
    int next_index = 0, open = -1;
    std::string open_type = std::string(), thinking_acc = std::string();

    bool send(const char * event, const json & j) {
        qwfn::assert_that(event != nullptr, "sse: an event name");
        const std::string s = std::string("event: ") + event + "\ndata: " + json_dump(j) + "\n\n";
        const bool ok = sink.write(s.data(), s.size());
        ev.last_write = clk::now();
        if (!ok) ev.write_failed = true;
        qwfn::assert_that(ok || ev.write_failed, "sse: a failed write is recorded");
        return ok;
    }
    bool delta(const json & d) {
        return send("content_block_delta", json{{"type", "content_block_delta"}, {"index", open}, {"delta", d}});
    }
    bool close_block() {
        qwfn::assert_that(open < next_index, "sse: an opened block");
        if (open < 0) return true;
        if (open_type == "thinking" && !delta(json{{"type", "signature_delta"}, {"signature", thinking_signature(thinking_acc)}})) return false;
        const bool ok = send("content_block_stop", json{{"type", "content_block_stop"}, {"index", open}});
        open = -1; open_type.clear();
        qwfn::assert_that(open_type.empty(), "sse: no block open");
        return ok;
    }
    bool open_block(const std::string & ty, const json & block) {
        qwfn::assert_that(!ty.empty(), "sse: a block type");
        if (!close_block()) return false;
        open = next_index++; open_type = ty;
        qwfn::assert_that(open >= 0 && open == next_index - 1, "sse: the newest block open");
        return send("content_block_start", json{{"type", "content_block_start"}, {"index", open}, {"content_block", block}});
    }
    // The stream events as content blocks.
    bool reasoning(const std::string & t) {
        qwfn::assert_that(open < next_index, "sse: an opened block");
        if (open_type != "thinking" && !open_block("thinking", json{{"type", "thinking"}, {"thinking", ""}})) return false;
        thinking_acc += t;
        qwfn::assert_that(open_type == "thinking", "sse: a thinking block open");
        return delta(json{{"type", "thinking_delta"}, {"thinking", t}});
    }
    bool content(const std::string & t) {
        qwfn::assert_that(open < next_index, "sse: an opened block");
        if (t.empty()) return true;
        if (open_type != "text" && !open_block("text", json{{"type", "text"}, {"text", ""}})) return false;
        qwfn::assert_that(open_type == "text", "sse: a text block open");
        return delta(json{{"type", "text_delta"}, {"text", t}});
    }
    bool tool_delta(const json & d) {
        qwfn::assert_that(d.is_object(), "sse: a tool delta object");
        qwfn::assert_that(d.contains("function"), "sse: a tool delta carries its function");
        if (d.contains("id"))   // the opening delta: the name is known, the arguments follow
            return open_block("tool_use", json{{"type", "tool_use"}, {"id", tool_use_id(d["id"].get<std::string>())},
                                               {"name", d["function"]["name"]}, {"input", json::object()}});
        const std::string frag = d["function"].value("arguments", "");
        return frag.empty() || delta(json{{"type", "input_json_delta"}, {"partial_json", frag}});
    }
    bool keepalive() { return send("ping", json{{"type", "ping"}}); }
};
// The stream events, routed to the request's writer.
bool stream_events::reasoning(const std::string & t) {
    qwfn::assert_that(one_writer(), "stream events: exactly one writer");
    qwfn::assert_that(!t.empty(), "stream events: reasoning to write");
    return oai ? oai->emit(json{{"reasoning_content", t}}) : ant->reasoning(t);
}
bool stream_events::content(const std::string & t) {
    qwfn::assert_that(one_writer(), "stream events: exactly one writer");
    qwfn::assert_that(t.size() < SIZE_MAX / 2, "stream events: a sane content length");
    return oai ? oai->emit(json{{"content", t}}) : ant->content(t);
}
bool stream_events::tool_delta(const json & d) {
    qwfn::assert_that(one_writer(), "stream events: exactly one writer");
    qwfn::assert_that(d.is_object(), "stream events: a tool delta object");
    return oai ? oai->emit(json{{"tool_calls", json::array({d})}}) : ant->tool_delta(d);
}
bool stream_events::tool_end() {
    qwfn::assert_that(one_writer(), "stream events: exactly one writer");
    qwfn::assert_that(last_write <= clk::now(), "stream events: a write stamp from the past");
    return oai ? true : ant->close_block();
}
bool stream_events::keepalive() {
    qwfn::assert_that(one_writer(), "stream events: exactly one writer");
    qwfn::assert_that(last_write <= clk::now(), "stream events: a write stamp from the past");
    return oai ? oai->keepalive() : ant->keepalive();
}
// message_start goes out before the prefill, so its usage is the
// prompt as it will be counted: what the engine already holds and
// the rest; message_delta repeats it measured.
static void anthropic_message_start(server & S, const chat_request & Q, const server::prompt & P, const std::string & model, anthropic_sse & A) {
    int32_t reuse = prefix_reuse(S, P);
    if (reuse == 0) {   // what generate() will restore, so message_start and message_delta agree
        const int b = S.best_saved(P.tok, !P.splices.empty());
        if (b >= 0 && snapshots_on()) reuse = (int32_t) S.saved[b].tok.size();
    }
    qwfn::assert_that(reuse >= 0 && (size_t) reuse <= P.tok.size(), "message_start: the reused prefix within the prompt");
    qwfn::assert_that(A.open == -1 && A.next_index == 0, "message_start: before any block");
    A.send("message_start", json{{"type", "message_start"}, {"message", {
        {"id", Q.id}, {"type", "message"}, {"role", "assistant"}, {"model", model}, {"content", json::array()},
        {"stop_reason", nullptr}, {"stop_sequence", nullptr},
        {"usage", {{"input_tokens", (int) P.tok.size() - reuse}, {"cache_creation_input_tokens", 0},
                   {"cache_read_input_tokens", reuse}, {"output_tokens", 0}}}}}});
}
static void anthropic_stream(server & S, chat_request & Q, const server::prompt & P, const std::string & model, httplib::DataSink & sink) {
    qwfn::assert_that(Q.stream, "anthropic stream: a streaming request");
    stream_events ev;
    anthropic_sse A{sink, ev};
    ev.ant = &A;
    anthropic_message_start(S, Q, P, model, A);
    stream_outcome out = run_stream(S, Q, P, ev);
    if (!out.aborted.empty() || !out.ok) {
        const bool client = out.aborted.empty() && too_long(out.err);
        A.send("error", json{{"type", "error"}, {"error", {
            {"type", client ? "invalid_request_error" : "api_error"},
            {"message", !out.aborted.empty() ? "stream aborted: " + out.aborted : client ? too_long_msg(S, P.tok.size()) : out.err}}}});
        sink.done();
        return;
    }
    A.close_block();
    A.send("message_delta", json{{"type", "message_delta"},
        {"delta", {{"stop_reason", anthropic_stop_reason(out.C.R.finish, out.C.R.stop_seq)},
                   {"stop_sequence", out.C.R.stop_seq.empty() ? json(nullptr) : json(out.C.R.stop_seq)}}},
        {"usage", anthropic_usage(out.C.R)}});
    A.send("message_stop", json{{"type", "message_stop"}});
    sink.done();
    // A failed write (the client left) can leave a block open; that is not a bug.
    qwfn::assert_that(A.open == -1 || ev.write_failed, "anthropic stream: every block closed unless the client left");
}
static void handle_messages(server & S, const httplib::Request & req, httplib::Response & res) {
    json body;
    auto Q = std::make_shared<chat_request>();
    if (!parse_anthropic(S, req, res, "msg_", body, *Q)) return;
    const std::string model = body.value("model", S.model_id);
    log_request(S, req, *Q, body, " (anthropic)");

    auto lk = std::make_shared<std::unique_lock<std::mutex>>(S.mu);
    qwfn::assert_that(lk->owns_lock(), "messages: the engine lock held");
    auto P = std::make_shared<server::prompt>();
    std::string e;
    if (!build_prompt(S, Q->msgs, Q->effort, Q->thinking, Q->tools_block, Q->forced, *P, e)) { fail_anthropic(res, 400, e); return; }

    if (!Q->stream) {
        if (req.is_connection_closed()) { fprintf(stderr, "[qwfn-server] %s: client left while queued, skipped\n", Q->id.c_str()); return; }
        chat_result C;
        if (!run_batch(S, *Q, *P, C, e, req)) {
            if (too_long(e)) fail_anthropic(res, 400, too_long_msg(S, P->tok.size()));
            else             fail_anthropic(res, 500, e, "api_error");
            return;
        }
        res.set_content(anthropic_message(*Q, model, C).dump(2, ' ', false, json::error_handler_t::replace), "application/json");
        return;
    }

    res.set_header("Cache-Control", "no-cache");
    server * sp = &S;
    qwfn::assert_that(sp != nullptr, "messages: a server for the provider");
    res.set_chunked_content_provider("text/event-stream",
        [sp, Q, P, lk, model](size_t, httplib::DataSink & sink) mutable {
            anthropic_stream(*sp, *Q, *P, model, sink);
            lk->unlock();
            return true;
        });
}
// The exact count needs the tokenizer's view of the whole prompt (framing,
// the tools block, the images encoded) and that is built under the engine
// lock; while a generation holds it, an estimate goes back instead of a wait.
static void handle_count_tokens(server & S, const httplib::Request & req, httplib::Response & res) {
    json body; chat_request Q;
    if (!parse_anthropic(S, req, res, "count_", body, Q)) return;
    qwfn::assert_that(Q.msgs.is_array(), "count_tokens: a message list");
    std::unique_lock<std::mutex> lk(S.mu, std::try_to_lock);
    if (!lk.owns_lock()) {
        res.set_content(json{{"input_tokens", (long long) (req.body.size() / 4)}}.dump(), "application/json");
        return;
    }
    server::prompt P; std::string e;
    if (!build_prompt(S, Q.msgs, Q.effort, Q.thinking, Q.tools_block, Q.forced, P, e)) { fail_anthropic(res, 400, e); return; }
    qwfn::assert_that(!P.tok.empty(), "count_tokens: a prompt");
    res.set_content(json{{"input_tokens", (long long) P.tok.size()}}.dump(), "application/json");
}

// Raw completion: no chat framing at all, for perplexity-style harnesses.
static void handle_completions(server & S, const httplib::Request & req, httplib::Response & res) {
    json body;
    try { body = json::parse(req.body); }
    catch (const std::exception & ex) { fail(res, 400, std::string("bad JSON: ") + ex.what()); return; }
    qwfn::assert_that(parsed(body), "completions: a parsed body");
    const std::string p = body.value("prompt", "");
    sampler smp;
    { std::lock_guard<std::mutex> lk(S.props_mu); smp.cfg = S.preset_nothink; }
    int max_tok = 128;
    std::string e;
    if (!smp.cfg.from_json(body, e) || !num_field(body, "max_tokens", -1, INT_MAX, max_tok, e)) { fail(res, 400, e); return; }
    std::vector<std::string> stops;
    if (body.contains("stop")) {
        if (body["stop"].is_string()) stops.push_back(body["stop"].get<std::string>());
        else for (const auto & s : body["stop"]) stops.push_back(s.get<std::string>());
    }

    std::lock_guard<std::mutex> lk(S.mu);
    S.last_msgs = json();            // raw completions break the chat chain
    server::prompt P;
    P.tok = S.vb.encode(p, false, true);
    if (P.tok.empty()) { fail(res, 400, "prompt is empty"); return; }
    gen_result R;
    gen_sink out;   // no client check, no ticks: a plain completion runs to its end
    if (!generate(S, P, smp, max_tok, /*thinking=*/false, stops, out, R, e)) {
        fail(res, 500, e, "server_error"); return;
    }
    qwfn::assert_that(R.n_input >= R.n_cached, "completions: the reused prefix within the prompt");
    res.set_content(json{
        {"id", "cmpl-" + std::to_string(now_unix())}, {"object", "text_completion"},
        {"created", now_unix()}, {"model", S.model_id},
        {"choices", json::array({ json{
            {"index", 0}, {"text", R.reasoning + R.content},
            {"finish_reason", R.finish}} })},
        {"usage", {{"prompt_tokens", R.n_input}, {"prompt_tokens_details", {{"cached_tokens", R.n_cached}}},
                   {"completion_tokens", R.n_gen}, {"total_tokens", R.n_input + R.n_gen}}}
    }.dump(2, ' ', false, json::error_handler_t::replace), "application/json");
}

// ---- settings a harness can read and change live -------------------------
static json props_json(server & S) {
    const engine_config & cfg = S.cfg;
    qwfn::assert_that(S.n_ctx == cfg.n_ctx && S.n_batch == cfg.n_batch, "props: the context as started");
    std::lock_guard<std::mutex> lk(S.props_mu);
    qwfn::assert_that(effort_valid(S.def_effort), "props: a valid default effort");
    return json{
        {"model", S.model_id}, {"n_ctx", S.n_ctx}, {"n_batch", S.n_batch},
        {"default_generation_settings", {
            {"reasoning_effort", S.def_effort}, {"max_tokens", S.def_max_tokens},
            {"reasoning_budget", S.def_reasoning_budget},
            {"thinking", S.preset_think.to_json()}, {"non_thinking", S.preset_nothink.to_json()}}},
        {"dump_requests", S.dump_dir},
        {"skip_miss", cfg.skip_miss}, {"spec_block", cfg.spec_block}, {"mtp", S.eng.mtp_loaded()}, {"model_file", S.model_file},
        {"vision", S.vis.loaded()}, {"vision_weights", S.vis.loaded() ? "cpu" : "off"},
        {"state_host", cfg.kv_host && cfg.idx_host ? "kv,idx" : cfg.kv_host ? "kv" : cfg.idx_host ? "idx" : "none"},
        {"n_threads", S.eng.n_threads()}, {"kv_type", cfg.type_k == GGML_TYPE_Q4_0 ? "q4_0" : cfg.type_k == GGML_TYPE_Q8_0 ? "q8_0" : "f16"},
        {"total_slots", 1}};
}
// POST /props: {"reasoning_effort":"off", "max_tokens":1024,
//               "thinking":{"temperature":..}, "non_thinking":{...}}
static void post_props(server & S, const httplib::Request & req, httplib::Response & res) {
    json body;
    try { body = json::parse(req.body); } catch (const std::exception & ex) { fail(res, 400, std::string("bad JSON: ") + ex.what()); return; }
    qwfn::assert_that(parsed(body), "props: a parsed body");
    {
        std::lock_guard<std::mutex> lk(S.props_mu);
        // Validated as a whole before anything changes: a bad field leaves the settings as they were.
        std::string effort = S.def_effort;
        if (body.contains("reasoning_effort")) {
            if (!body["reasoning_effort"].is_string() || !effort_valid(body["reasoning_effort"].get<std::string>())) {
                fail(res, 400, "reasoning_effort must be xhigh|medium|low|off"); return;
            }
            effort = body["reasoning_effort"].get<std::string>();
        }
        int max_tokens = S.def_max_tokens, budget = S.def_reasoning_budget;
        sampling think = S.preset_think, nothink = S.preset_nothink;
        std::string e;
        const bool ok = num_field(body, "max_tokens", -1, INT_MAX, max_tokens, e) && num_field(body, "reasoning_budget", -1, INT_MAX, budget, e)
            && (!body.contains("thinking") || think.from_json(body["thinking"], e))
            && (!body.contains("non_thinking") || nothink.from_json(body["non_thinking"], e))
            && think.from_json(body, e) && nothink.from_json(body, e);   // flat sampling fields apply to both presets
        if (!ok) { fail(res, 400, e); return; }
        S.def_effort = effort;
        S.def_max_tokens = max_tokens; S.def_reasoning_budget = budget;
        S.preset_think = think; S.preset_nothink = nothink;
        if (body.contains("dump_requests")) S.dump_dir = body["dump_requests"].is_string() ? body["dump_requests"].get<std::string>() : "";
        qwfn::assert_that(effort_valid(S.def_effort), "props: a valid default effort");
    }
    // CPU threads for the RAM-served experts, applied between requests (the console's
    // auto-tune sweeps it on the running server). Refused while a generation holds the engine.
    if (body.contains("threads")) {
        int n = 0; std::string te;
        if (!num_field(body, "threads", 1, 512, n, te)) { fail(res, 400, "threads must be 1..512"); return; }
        std::unique_lock<std::mutex> lk(S.mu, std::try_to_lock);
        if (!lk.owns_lock()) { fail(res, 409, "a request is running; set threads between requests"); return; }
        S.eng.set_n_threads(n);
        fprintf(stderr, "[qwfn-server] threads set to %d\n", n);
    }
    res.set_content(props_json(S).dump(2, ' ', false, json::error_handler_t::replace), "application/json");
}
// ---- the live counter ------------------------------------------------------
static json stats_json(server & S) {
    json t = S.live.timings();
    const auto & c = S.eng.cache_stats();   // racy reads of plain counters: a monitor, not a ledger
    long long np, ng, nr, npair, nacc, ndraft; double tp, tg; bool busy; int n_past;
    { std::lock_guard<std::mutex> lk(S.live.mu); np = S.live.n_prompt_total; ng = S.live.n_gen_total; nr = S.live.n_requests;
      tp = S.live.t_prompt_total; tg = S.live.t_gen_total; busy = S.live.busy; n_past = S.live.n_past;
      npair = S.live.n_pairs_total; nacc = S.live.n_accepted_total; ndraft = S.live.n_drafted_total; }
    long long ninp, ncach; { std::lock_guard<std::mutex> lk(S.live.mu); ninp = S.live.n_input_total; ncach = S.live.n_cached_total; }
    qwfn::assert_that(nacc <= ndraft && npair >= 0, "stats: drafts accepted within those proposed");
    qwfn::assert_that(np >= 0 && ng >= 0 && nr >= 0, "stats: totals");
    return json{
        {"busy", busy},
        // prompt: the current request's prompt while busy, the last one's when idle.
        // input = cached (reused from the engine's prefix) + n (prefilled); done counts
        // the prefilled tokens so far, fractional inside a batch.
        {"prompt", {{"n", t["prompt_n"]}, {"ms", t["prompt_ms"]}, {"tokens_per_second", t["prompt_per_second"]},
                    {"input", t["prompt_input_n"]}, {"cached", t["prompt_cached_n"]}, {"done", t["prompt_done_n"]}, {"prefilling", t["prefilling"]}}},
        {"generation", {{"n", t["predicted_n"]}, {"ms", t["predicted_ms"]}, {"tokens_per_second", t["predicted_per_second"]}}},
        {"last", S.live.last_json()},
        {"context", {{"n_past", n_past}, {"n_ctx", S.n_ctx}}},
        {"threads", S.eng.n_threads()},
        {"totals", {{"requests", nr}, {"prompt_tokens", np}, {"prompt_tokens_per_second", tp > 0 ? np / tp : 0.0},
                    {"input_tokens", ninp}, {"cached_tokens", ncach},
                    {"generated_tokens", ng}, {"generated_tokens_per_second", tg > 0 ? ng / tg : 0.0},
                    {"prompt_seconds", tp}, {"generation_seconds", tg}}},
        {"expert_cache", {{"hit_rate", c.hit_rate()}, {"vram_served", c.gpu_rate()},
                          {"bytes_from_disk", c.bytes_from_disk},
                          // the raw counters, so a harness can difference two samples
                          {"lookups", c.lookups}, {"hits", c.hits}, {"gpu_hits", c.gpu_hits},
                          {"promotions", c.promotions}, {"pf_issued", c.pf_issued}, {"pf_used", c.pf_used},
                          {"read_errors", c.read_errors}, {"read_short", c.read_short}}},
        {"speculative", {{"pairs", npair}, {"accepted", nacc}, {"drafted", ndraft}, {"acceptance", ndraft ? (double) nacc / ndraft : 0.0},
                         {"tokens_per_step", npair ? (double) (npair + nacc) / npair : 1.0}}},
        {"timings", t}};
}
static void get_metrics(server & S, httplib::Response & res) {
    json st = stats_json(S);
    qwfn::assert_that(st.contains("totals") && st.contains("expert_cache"), "metrics: the stats sections");
    char buf[2048];
    const int w = snprintf(buf, sizeof buf,
        "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n# TYPE llamacpp:prompt_tokens_total counter\nllamacpp:prompt_tokens_total %lld\n"
        "# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.\n# TYPE llamacpp:tokens_predicted_total counter\nllamacpp:tokens_predicted_total %lld\n"
        "# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s.\n# TYPE llamacpp:prompt_tokens_seconds gauge\nllamacpp:prompt_tokens_seconds %.2f\n"
        "# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s (current or last request).\n# TYPE llamacpp:predicted_tokens_seconds gauge\nllamacpp:predicted_tokens_seconds %.2f\n"
        "# HELP llamacpp:n_busy_slots_per_decode Busy slots.\n# TYPE llamacpp:n_busy_slots_per_decode gauge\nllamacpp:n_busy_slots_per_decode %d\n"
        "# HELP qwfn:expert_cache_hit_rate Share of expert lookups served from VRAM or RAM.\n# TYPE qwfn:expert_cache_hit_rate gauge\nqwfn:expert_cache_hit_rate %.4f\n"
        "# HELP qwfn:expert_vram_rate Share of expert lookups served from VRAM.\n# TYPE qwfn:expert_vram_rate gauge\nqwfn:expert_vram_rate %.4f\n"
        "# HELP qwfn:expert_bytes_from_disk_total Bytes of experts read from the NVMe.\n# TYPE qwfn:expert_bytes_from_disk_total counter\nqwfn:expert_bytes_from_disk_total %llu\n",
        st["totals"]["prompt_tokens"].get<long long>(), st["totals"]["generated_tokens"].get<long long>(),
        st["prompt"]["tokens_per_second"].get<double>(), st["generation"]["tokens_per_second"].get<double>(),
        st["busy"].get<bool>() ? 1 : 0,
        st["expert_cache"]["hit_rate"].get<double>(), st["expert_cache"]["vram_served"].get<double>(),
        (unsigned long long) st["expert_cache"]["bytes_from_disk"].get<double>());
    qwfn::assert_that(w > 0, "metrics: formatted");
    res.set_content(buf, "text/plain; version=0.0.4");
}
static void register_info_routes(httplib::Server & svr, server & S) {
    qwfn::assert_that(svr.is_valid(), "routes: a usable server");
    qwfn::assert_that(!S.model_id.empty(), "routes: a model id");
    server * sp = &S;
    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    });
    svr.Get("/v1/models", [sp](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{
            {"object", "list"},
            {"data", json::array({ json{
                {"id", sp->model_id}, {"object", "model"},
                {"created", now_unix()}, {"owned_by", "qwfnfer"},
                // The Claude tier this model stands in for, for clients that
                // discover models by it (Claude Desktop); everyone else ignores it.
                {"display_name", sp->model_id}, {"anthropic_family_tier", "sonnet"}} })}
        }.dump(2, ' ', false, json::error_handler_t::replace), "application/json");
    });
    svr.Get("/props", [sp](const httplib::Request &, httplib::Response & res) {
        res.set_content(props_json(*sp).dump(2, ' ', false, json::error_handler_t::replace), "application/json");
    });
    svr.Post("/props", [sp](const httplib::Request & req, httplib::Response & res) { post_props(*sp, req, res); });
    svr.Get("/stats", [sp](const httplib::Request &, httplib::Response & res) {
        res.set_content(stats_json(*sp).dump(2, ' ', false, json::error_handler_t::replace), "application/json");
    });
    svr.Get("/slots", [sp](const httplib::Request &, httplib::Response & res) {
        json st = stats_json(*sp);
        res.set_content(json::array({ json{
            {"id", 0}, {"id_task", -1}, {"is_processing", st["busy"]},
            {"n_ctx", sp->n_ctx}, {"n_past", st["context"]["n_past"]},
            {"model", sp->model_id}, {"params", props_json(*sp)["default_generation_settings"]},
            {"next_token", {{"n_decoded", st["generation"]["n"]}}}} }).dump(2, ' ', false, json::error_handler_t::replace), "application/json");
    });
    svr.Get("/metrics", [sp](const httplib::Request &, httplib::Response & res) { get_metrics(*sp, res); });
}
static void register_chat_routes(httplib::Server & svr, server & S) {
    qwfn::assert_that(svr.is_valid(), "routes: a usable server");
    server * sp = &S;
    qwfn::assert_that(sp != nullptr, "routes: a server");
    svr.Post("/v1/chat/completions", [sp](const httplib::Request & req, httplib::Response & res) { handle_chat(*sp, req, res); });
    svr.Post("/v1/messages", [sp](const httplib::Request & req, httplib::Response & res) { handle_messages(*sp, req, res); });
    svr.Post("/v1/messages/count_tokens", [sp](const httplib::Request & req, httplib::Response & res) { handle_count_tokens(*sp, req, res); });
    // Claude Code warms its connection with HEAD /api/hello (httplib answers a
    // HEAD from the GET handler).
    svr.Get("/api/hello", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"ok", true}}.dump(), "application/json");
    });
    svr.Post("/v1/completions", [sp](const httplib::Request & req, httplib::Response & res) { handle_completions(*sp, req, res); });
}

// Stall watchdog: a request that makes no progress is logged with what the expert cache is
// waiting on, and the stuck thread prints its own stack. Progress is prefill layers AND generated
// tokens: prompt_done_locked() includes the engine's per-layer prefill progress, so a long prefill that
// is moving is not a stall (it used to be reported every 30 s as "no new token ... at generated
// token 0"). Thresholds: 30 s while generating, 120 s while prefilling, because one layer of the
// first prefill after a load can spend ~60 s in kernel JIT.
static void stall_watchdog(server & S) {
    double last_p = -1, stalled = 0;
    // rule 2 deviation: the watchdog thread runs for the life of the server process (detached), see docs/CODING_RULES.md
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        bool busy, prefilling; int n; double done, input;
        { std::lock_guard<std::mutex> lk(S.live.mu); busy = S.live.busy; prefilling = S.live.prefilling;
          n = S.live.n_gen; done = S.live.prompt_done_locked(); input = (double) S.live.n_prompt; }
        if (!busy) { last_p = -1; stalled = 0; continue; }
        const double p = done + (double) n;
        if (p != last_p) { last_p = p; stalled = 0; continue; }
        stalled += 5;
        qwfn::assert_that(stalled >= 5, "watchdog: a stall is counted in 5 s steps");
        const double limit = prefilling ? 120 : 30;
        if (stalled >= limit && ((int) stalled % 30) == 0) {
            const int ws = S.eng.cache_wait_state();
            const char * what = ws == 1 ? "demand reads" : ws == 2 ? "speculative reads" : "nothing (compute or lock)";
            qwfn::assert_that(what != nullptr, "watchdog: a wait state");
            if (prefilling)
                fprintf(stderr, "[qwfn-server] STALL: prefill has not advanced for %.0f s (at %.0f of %.0f prompt tokens); expert cache waiting on %s (%zu reads)\n",
                        stalled, done, input, what, S.eng.cache_wait_count());
            else
                fprintf(stderr, "[qwfn-server] STALL: no new token for %.0f s at generated token %d; expert cache waiting on %s (%zu reads)\n",
                        stalled, n, what, S.eng.cache_wait_count());
            if (S.gen_thread_set) pthread_kill(S.gen_thread, SIGUSR2);   // the stuck thread prints its own stack
        }
    }
}
static void setup_http(httplib::Server & svr, server & S) {
    qwfn::assert_that(svr.is_valid(), "http: a usable server");
    svr.set_payload_max_length(256ull << 20);   // base64 images are bulky
    // SO_REUSEADDR, as llama-server sets, instead of httplib's default SO_REUSEPORT. A
    // supervisor may reuse one child port across servers: connections the previous
    // server closed stay in TIME_WAIT on it for up to 60 s, and Linux lets a new bind through
    // only when both sockets carry the same option. With REUSEPORT only, switching between
    // this server and llama-server or vLLM failed with "address in use" in both directions
    // until TIME_WAIT expired. REUSEPORT would also let a stale instance share the port.
    svr.set_socket_options([](socket_t sock) { httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1); });
    // httplib's socket timeouts default to 5 s per write and per read. A client
    // UI that stops draining the stream for 5 s (rendering a long reasoning
    // trace) would get the connection cut without a trailer and without a log
    // line here -- "peer closed connection without sending complete message
    // body" on its side. A local server can afford to wait.
    // prime the unwinder: glibc's first backtrace() dlopens libgcc_s, taking the loader and malloc locks.
    // Done first inside the stall probe's handler, it deadlocked a generation thread interrupted inside
    // malloc (a long first-request JIT compile) for good. Loaded here, while nothing is interrupted.
    { void * f[2]; (void) backtrace(f, 2); }
    signal(SIGSEGV, on_fatal); signal(SIGABRT, on_fatal); signal(SIGBUS, on_fatal); signal(SIGFPE, on_fatal);
    signal(SIGUSR2, on_stall_probe);
    // A local web page (the console, a harness) may read /stats and /props
    // from another origin: allow it.
    svr.set_default_headers({{"Access-Control-Allow-Origin", "*"}, {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
                             {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"}});
    svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) { res.status = 204; });
    svr.set_write_timeout(3600, 0);
    server * sp = &S;
    qwfn::assert_that(sp != nullptr, "http: a server for the watchdog");
    std::thread([sp]() { stall_watchdog(*sp); }).detach();
    svr.set_read_timeout(600, 0);
}

// ---- startup ------------------------------------------------------------------
static const char * USAGE =
          "usage: qwfn-server <shard.gguf> [options]\n"
          "\n"
          "      --host HOST     bind address (default 127.0.0.1)\n"
          "      --port N        port (default 8080)\n"
          "      --mmproj PATH   vision projector gguf; enables image input. It runs on the CPU (weights in RAM, no VRAM\n"
          "                      at all): a 1400x1000 screenshot encodes in ~15 s, a 320x240 image in 0.3 s\n"
          "      --vision-threads N  threads for the image encode (default: --threads, the physical cores)\n"
          "      --state-host V  attention state in pinned host memory: none (default) | idx | kv,idx. The VRAM it held goes to\n"
          "                      the expert tier; costs ~0.35 ms/token (idx) or ~2 ms/token (kv,idx) of PCIe reads\n"
          "      --alias NAME    model id reported by /v1/models\n"
          "      --think LEVEL   default reasoning effort: xhigh|medium|low|off\n"
          "      --think-budget N  max reasoning tokens per answer (0 = unlimited); also POST /props {\"reasoning_budget\":N} or per request\n"
          "      --ctx N         context (default 32768)   --batch N (default 32768)\n"
          "      --ram GB        --vram GB   --threads N   --cpu   --kv f16|q8_0\n"
          "      --reserve MB    VRAM kept free after the expert tier is sized (default 768; raise it on a desktop GPU)\n"
          "      --ram-frac F    MemAvailable share the RAM tier may take (default 0.75)\n"
          "      --spec-ahead N  predict 1 or 2 layers ahead (default 2)\n"
          "      --no-prefill-overlap   single prefill staging buffer, saves ~1.8 GB RAM\n"
          "\n"
          "Endpoints: GET /health, GET /v1/models,\n"
          "           POST /v1/chat/completions (stream supported; timings_per_token:true adds live tok/s to every chunk),\n"
          "           POST /v1/messages, POST /v1/messages/count_tokens (the Anthropic Messages API: Claude Code with\n"
          "               ANTHROPIC_BASE_URL=http://127.0.0.1:PORT; thinking and tool_use blocks, streamed),\n"
          "           GET /props, POST /props (reasoning_effort, max_tokens, thinking/non_thinking sampling presets),\n"
          "           GET /stats (live tok/s, context, expert cache), GET /slots, GET /metrics (Prometheus),\n"
          "           POST /v1/completions\n";

struct options {
    std::string host = "127.0.0.1", mmproj_path, alias, def_effort = "xhigh";
    int vision_threads = 0;
    int def_reasoning_budget = 0;
    int port = 8080;
    engine_config cfg;
};
// The server's own options; true when `a` was one of them (i then points at its last argument).
static bool parse_server_opt(const std::string & a, int argc, char ** argv, int & i, options & o) {
    qwfn::assert_that(argv != nullptr && i >= 2 && i < argc, "args: an argument index");
    engine_config & cfg = o.cfg;
    auto next = [&]() { return argv[++i]; };
    if (a == "--host"   && i + 1 < argc) { o.host = next(); return true; }
    if (a == "--port"   && i + 1 < argc) { o.port = (int) qwfn::arg_int("--port", next(), 1, 65535); return true; }
    if (a == "--mmproj" && i + 1 < argc) { o.mmproj_path = next(); return true; }
    if (a == "--vision-threads" && i + 1 < argc) { o.vision_threads = (int) qwfn::arg_int("--vision-threads", next(), 0, 1024); return true; }
    if (a == "--alias"  && i + 1 < argc) { o.alias = next(); return true; }
    if (a == "--think"  && i + 1 < argc) { o.def_effort = next(); return true; }
    if (a == "--think-budget" && i + 1 < argc) { o.def_reasoning_budget = (int) qwfn::arg_int("--think-budget", next(), 0, 1 << 24); return true; }
    if (a == "--ctx"    && i + 1 < argc) { cfg.n_ctx = (uint32_t) qwfn::arg_int("--ctx", next(), 1, 2147483647); return true; }
    if (a == "--batch"  && i + 1 < argc) { cfg.n_batch = (uint32_t) qwfn::arg_int("--batch", next(), 1, 1 << 20); return true; }
    if (a == "--ram"    && i + 1 < argc) { cfg.ram_bytes = (size_t)(qwfn::arg_float("--ram", next(), 0, 1e6) * 1e9); return true; }
    if (a == "--vram"   && i + 1 < argc) { cfg.vram_bytes = (size_t)(qwfn::arg_float("--vram", next(), 0, 1e6) * 1e9); return true; }
    if (a == "--threads"&& i + 1 < argc) { cfg.n_threads = (int) qwfn::arg_int("--threads", next(), 1, 1024); return true; }
    if (a == "--ram-frac" && i + 1 < argc) { cfg.ram_frac = qwfn::arg_float("--ram-frac", next(), 0, 1); return true; }
    if (a == "--reserve" && i + 1 < argc) { cfg.vram_reserve = (size_t) qwfn::arg_float("--reserve", next(), 0, 1e7) * (1ull << 20); return true; }
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
    qwfn::assert_that(i < argc, "args: no value taken for an unknown option");
    return false;
}
// The engine's tuning options, as parse_server_opt.
static bool parse_engine_opt(const std::string & a, int argc, char ** argv, int & i, options & o) {
    qwfn::assert_that(argv != nullptr && i >= 2 && i < argc, "args: an argument index");
    engine_config & cfg = o.cfg;
    auto next = [&]() { return argv[++i]; };
    if (a == "--ubatch-kv" && i + 1 < argc) { cfg.ubatch_kv_product = (uint64_t)(qwfn::arg_float("--ubatch-kv", next(), 0, 1e7) * 1e6); return true; }
    if (a == "--indexer-top-k" && i + 1 < argc) { cfg.indexer_top_k = (uint32_t) qwfn::arg_int("--indexer-top-k", next(), 0, 1 << 20); return true; }
    if (a == "--spec-ahead" && i + 1 < argc) { cfg.speculate_ahead = (uint32_t) qwfn::arg_int("--spec-ahead", next(), 0, 16); return true; }
    if (a == "--no-prefill-overlap") { cfg.prefill_overlap = false; return true; }
    if (a == "--cpu")   { cfg.use_gpu = false; return true; }
    if (a == "--no-qsa"){ cfg.use_qsa = false; return true; }
    if (a == "--skip-miss") { cfg.skip_miss = true; return true; }
    if (a == "--spec-depth" && i + 1 < argc) { cfg.speculate_depth = (uint32_t) qwfn::arg_int("--spec-depth", argv[++i], 0, QWFN_SPEC_MAX);
        if (cfg.speculate_depth == 0) { cfg.speculate = false; }
        return true; }
    if (a == "--spec-ahead" && i + 1 < argc) { cfg.speculate_ahead = (uint32_t) qwfn::arg_int("--spec-ahead", argv[++i], 0, 16); return true; }
    if (a == "--spec-depth2" && i + 1 < argc) { cfg.speculate_depth2 = (uint32_t) qwfn::arg_int("--spec-depth2", argv[++i], 0, QWFN_SPEC_MAX); return true; }
    if (a == "--spec-margin" && i + 1 < argc) { cfg.spec_margin = (float) qwfn::arg_float("--spec-margin", next(), 0, 1000); return true; }
    if (a == "--spec-gate-inflight" && i + 1 < argc) { cfg.spec_gate_inflight = (uint32_t) qwfn::arg_int("--spec-gate-inflight", next(), 0, 1 << 20); return true; }
    if (a == "--spec-block") { cfg.spec_block = true; return true; }
    // Prompts up to this many new tokens go through the cache-batched decode
    // path (one batch, experts through the tiers) instead of the streamed
    // prefill, whose cost is a full expert sweep (~12 s on Q4) whatever T is.
    if (a == "--prefill-decode-max" && i + 1 < argc) { cfg.prefill_decode_max = (uint32_t) qwfn::arg_int("--prefill-decode-max", argv[++i], 0, 1 << 20); return true; }
    if (a == "--gate-drop" && i + 1 < argc) { cfg.gate_drop = (float) qwfn::arg_float("--gate-drop", argv[++i], 0, 1); return true; }
    if (a == "--mtp-drafts" && i + 1 < argc) { cfg.mtp_drafts = (uint32_t) qwfn::arg_int("--mtp-drafts", argv[++i], 1, engine::MTP_MAX_DRAFTS); return true; }
    if (a == "--spec-block-layers" && i + 1 < argc) { cfg.spec_block = true; cfg.spec_block_layers = next(); return true; }
    if (a == "--mtp" && i + 1 < argc) { cfg.mtp_path = next(); cfg.rollback_snapshots = true; return true; }   // the nextn draft head: pairs verified by the trunk, exact
    qwfn::assert_that(i < argc, "args: no value taken for an unknown option");
    return false;
}
static bool parse_args(int argc, char ** argv, options & o) {
    qwfn::assert_that(argc >= 2 && argv != nullptr, "args: a model path");
    engine_config & cfg = o.cfg;
    // vram defaults high on purpose: the tier self-tunes down to whatever the
    // device can spare, and without it every routed expert computes on the CPU
    // at 3.2x the cost. --vram 0 still disables it.
    // --batch 32768: a long prompt streams every layer's experts once per batch, so a 21K-token
    // prompt at 16384 read all 43 GB twice (49.3 s) and at 32768 once (37.3 s), same peak RAM.
    cfg.n_ctx = 32768; cfg.n_batch = 32768; cfg.ram_bytes = 8e9; cfg.vram_bytes = 12e9;   // see qwfn-chat
    cfg.type_k = cfg.type_v = GGML_TYPE_Q8_0;   // see qwfn-chat

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (parse_server_opt(a, argc, argv, i, o) || parse_engine_opt(a, argc, argv, i, o)) continue;
        fprintf(stderr, "unknown option: %s\n", a.c_str());
        return false;
    }
    if (!effort_valid(o.def_effort)) { fprintf(stderr, "--think must be xhigh|medium|low|off\n"); return false; }
    qwfn::assert_that(effort_valid(o.def_effort), "args: a valid default effort");
    return true;
}
// Tokenizer, model index, engine and (with --mmproj) the vision tower. False: exit 1.
static bool init_server(server & S, const options & o, const char * model_path) {
    qwfn::assert_that(model_path != nullptr, "init: a model path");
    S.cfg = o.cfg;
    const engine_config & cfg = S.cfg;
    S.n_ctx = cfg.n_ctx; S.n_batch = cfg.n_batch; S.mtp_drafts = cfg.mtp_drafts; S.def_effort = o.def_effort; S.def_reasoning_budget = o.def_reasoning_budget;
    S.model_file = model_path;
    std::string init_err;

    fprintf(stderr, "loading tokenizer...\n");
    if (!S.vb.load(model_path, init_err)) { fprintf(stderr, "error: %s\n", init_err.c_str()); return false; }
    if (!S.mi.load(model_path, init_err)) { fprintf(stderr, "error: %s\n", init_err.c_str()); return false; }
    // The vision projector runs on the CPU backend (its weights in RAM, the
    // graph on the cores), so it takes no VRAM: nothing is reserved for it and
    // the expert tier is not lent while an image is encoded. Loaded after the
    // engine, which loads the ggml backends.
    if (!S.eng.init(&S.mi, nullptr, cfg,
                    qwfn::backend_dir(), init_err)) {
        fprintf(stderr, "engine init: %s\n", init_err.c_str()); return false;
    }
    fprintf(stderr, "%s\n", S.eng.memory_summary().c_str());
    S.live.eng = &S.eng;
    S.eng.set_mtp_logits(true);   // the draft is sampled from the head's distribution at temperature
    if (!o.mmproj_path.empty()) {
        S.vis_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!S.vis_backend) { fprintf(stderr, "vision: no CPU backend\n"); return false; }
        if (!S.vis.load(o.mmproj_path, S.vis_backend, ggml_backend_get_default_buffer_type(S.vis_backend), init_err)) {
            fprintf(stderr, "vision: %s\n", init_err.c_str()); return false;
        }
        // Physical cores, like the engine's own workers: 8 threads encoded a
        // 1400x1000 screenshot in 14.2 s where 16 took 15.5 (SMT starves the GEMMs).
        S.vis.set_n_threads(o.vision_threads > 0 ? o.vision_threads : cfg.n_threads);
        const auto ip = S.vb.encode("<|image_pad|>", false, true);
        if (ip.size() != 1) { fprintf(stderr, "vision: <|image_pad|> is not one token\n"); return false; }
        S.tok_image_pad = ip[0];
    }
    if (!o.alias.empty()) S.model_id = o.alias;
    qwfn::assert_that(!S.vis.loaded() || S.tok_image_pad >= 0, "init: an image pad token with the vision tower");
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fputs(USAGE, stderr); return 1; }
    qwfn::assert_that(argv[1] != nullptr, "main: a model path");
    options o;
    if (!parse_args(argc, argv, o)) return 1;
    server S;
    if (!init_server(S, o, argv[1])) return 1;
    qwfn::assert_that(S.n_ctx == o.cfg.n_ctx, "main: the server configured");

    httplib::Server svr;
    setup_http(svr, S);
    register_info_routes(svr, S);
    register_chat_routes(svr, S);

    fprintf(stderr, "qwfn-server listening on http://%s:%d  (model id: %s%s)\n",
            o.host.c_str(), o.port, S.model_id.c_str(),
            S.vis.loaded() ? ", vision enabled" : "");
    if (!svr.listen(o.host.c_str(), o.port)) {
        fprintf(stderr, "failed to bind %s:%d\n", o.host.c_str(), o.port);
        return 1;
    }
    return 0;
}
