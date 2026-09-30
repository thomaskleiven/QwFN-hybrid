#include "qwfn_vocab.h"
#include "qwfn_check.h"

#include "llama.h"

#include <cstdio>

namespace qwfn {

vocab::~vocab() {
    assert_that(!v_ || model_, "the vocab belongs to its model");
    if (model_) llama_model_free(model_);
    model_ = nullptr;
    v_     = nullptr;
    assert_that(!model_ && !v_, "vocab released");
}

bool vocab::load(const std::string & path, std::string & err) {
    assert_that(model_ == nullptr && v_ == nullptr, "vocab loaded once");
    static bool backend_ready = false;
    if (!backend_ready) { llama_backend_init(); backend_ready = true; }

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;              // parse the tokenizer, skip all 91 GB
    mp.n_gpu_layers = 0;

    model_ = llama_model_load_from_file(path.c_str(), mp);
    if (!model_) { err = "failed to load vocab from " + path; return false; }

    v_ = llama_model_get_vocab(model_);
    if (!v_) { err = "model has no vocab"; return false; }
    assert_that(model_ != nullptr && v_ != nullptr, "a loaded vocab has its model");
    return true;
}

std::vector<int32_t> vocab::encode(const std::string & text,
                                   bool add_special, bool parse_special) const {
    if (!v_) return {};
    assert_that(model_ != nullptr, "the vocab belongs to its model");
    // Negative return = -(required size); one retry at that size always suffices.
    std::vector<int32_t> out(text.size() + 16);
    int32_t n = llama_tokenize(v_, text.data(), (int32_t) text.size(),
                               out.data(), (int32_t) out.size(),
                               add_special, parse_special);
    if (n < 0) {
        out.resize(-n);
        n = llama_tokenize(v_, text.data(), (int32_t) text.size(),
                           out.data(), (int32_t) out.size(),
                           add_special, parse_special);
        if (n < 0) return {};
    }
    assert_that(n >= 0, "only a successful tokenization is returned");
    out.resize(n);
    return out;
}

std::string vocab::piece(int32_t tok, bool special) const {
    if (!v_) return {};
    assert_that(model_ != nullptr, "the vocab belongs to its model");
    char buf[256];
    int32_t n = llama_token_to_piece(v_, tok, buf, (int32_t) sizeof(buf), 0, special);
    if (n < 0) {
        std::string big((size_t) -n, '\0');
        n = llama_token_to_piece(v_, tok, big.data(), -n, 0, special);
        if (n < 0) return {};
        big.resize(n);
        return big;
    }
    assert_that(n >= 0 && (size_t) n <= sizeof(buf), "the piece fits the buffer it was written to");
    return std::string(buf, (size_t) n);
}

std::string vocab::decode(const std::vector<int32_t> & toks, bool unparse_special) const {
    if (!v_ || toks.empty()) return {};
    assert_that(model_ != nullptr, "the vocab belongs to its model");
    std::string out(toks.size() * 8 + 64, '\0');
    int32_t n = llama_detokenize(v_, toks.data(), (int32_t) toks.size(),
                                 out.data(), (int32_t) out.size(),
                                 false, unparse_special);
    if (n < 0) {
        out.assign((size_t) -n, '\0');
        n = llama_detokenize(v_, toks.data(), (int32_t) toks.size(),
                             out.data(), (int32_t) out.size(),
                             false, unparse_special);
        if (n < 0) return {};
    }
    assert_that(n >= 0, "only a successful tokenization is returned");
    out.resize(n);
    return out;
}

bool    vocab::is_eog(int32_t tok) const { return v_ && llama_vocab_is_eog(v_, tok); }
int32_t vocab::bos() const {
    assert_that(!v_ || model_, "the vocab belongs to its model");
    const int32_t t = v_ ? llama_vocab_bos(v_) : -1;
    assert_that(t >= -1, "a token id or none");
    return t;
}

int32_t vocab::eos() const {
    assert_that(!v_ || model_, "the vocab belongs to its model");
    const int32_t t = v_ ? llama_vocab_eos(v_) : -1;
    assert_that(t >= -1, "a token id or none");
    return t;
}

int32_t vocab::n_tokens() const {
    assert_that(!v_ || model_, "the vocab belongs to its model");
    const int32_t n = v_ ? llama_vocab_n_tokens(v_) : 0;
    assert_that(n >= 0, "a token count");
    return n;
}

}  // namespace qwfn
