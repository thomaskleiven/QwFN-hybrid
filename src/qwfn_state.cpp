#include "qwfn_state.h"
#include "qwfn_check.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "ggml-alloc.h"

namespace qwfn {

state::~state() {
    assert_that(!buf_ || ctx_, "the state buffer belongs to its context");
    assert_that(!buf_h_ || ctx_h_, "the host state buffer belongs to its context");
    if (buf_)   ggml_backend_buffer_free(buf_);
    if (ctx_)   ggml_free(ctx_);
    if (buf_h_) ggml_backend_buffer_free(buf_h_);
    if (ctx_h_) ggml_free(ctx_h_);
}

bool state::init(const hparams * hp, const state_config & cfg,
                 ggml_backend_buffer_type_t buft, std::string & err) {
    assert_that(hp != nullptr, "state needs hyper-parameters");
    assert_that(ctx_ == nullptr && buf_ == nullptr, "state initialised once");
    hp_  = hp;
    cfg_ = cfg;

    const uint32_t L = hp->n_layer;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (L * 5 + 16);
    ip.no_alloc = true;
    ctx_ = ggml_init(ip);
    if (!ctx_) { err = "ggml_init failed for state"; return false; }
    kv_host_  = cfg.kv_host;
    idx_host_ = cfg.idx_host;
    if (kv_host_ || idx_host_) { ctx_h_ = ggml_init(ip); if (!ctx_h_) { err = "ggml_init failed for host state"; return false; } }
    ggml_context * ck = kv_host_  ? ctx_h_ : ctx_;   // where the KV cache lives
    ggml_context * ci = idx_host_ ? ctx_h_ : ctx_;   // where the indexer cache lives
    declare_tensors(ck, ci);
    if (!alloc_buffers(buft, err)) return false;

    // Recurrent state and conv history must start at zero; the caches need not.
    reset();
    return true;
}

void state::declare_tensors(ggml_context * ck, ggml_context * ci) {
    assert_that(hp_ != nullptr && ctx_ != nullptr, "state context created before its tensors");
    assert_that(ck != nullptr && ci != nullptr, "cache contexts exist");
    const hparams * hp = hp_;
    const state_config & cfg = cfg_;
    const uint32_t L = hp->n_layer;
    k_.assign(L, nullptr); v_.assign(L, nullptr); idx_.assign(L, nullptr);
    rs_.assign(L, nullptr); conv_.assign(L, nullptr);

    const int64_t n_ctx      = cfg.n_ctx;
    const int64_t kv_dim     = (int64_t) hp->n_embd_head_k * hp->n_head_kv;   // 512
    const int64_t v_dim      = (int64_t) hp->n_embd_head_v * hp->n_head_kv;   // 512
    const int64_t idx_dim    = hp->idx_key_len;                               // 128
    const int64_t head_k     = hp->ssm_d_state;                               // 128
    const int64_t head_v     = hp->ssm_d_state;                               // 128
    const int64_t n_v_heads  = hp->ssm_dt_rank;                               // 48
    const int64_t n_k_heads  = hp->ssm_n_group;                               // 16
    const int64_t conv_dim   = head_k * n_k_heads * 2 + head_v * n_v_heads;   // 10240
    const int64_t hc_dim     = (int64_t) hp->hc_count * hp->n_embd;           // 10240

    for (uint32_t il = 0; il < L; il++) {
        if (hp->is_attn_layer(il)) {
            k_[il]   = ggml_new_tensor_1d(ck, cfg.type_k,   kv_dim * n_ctx);
            v_[il]   = ggml_new_tensor_1d(ck, cfg.type_v,   v_dim  * n_ctx);
            idx_[il] = ggml_new_tensor_1d(ci, cfg.type_idx, idx_dim * n_ctx);
            ggml_set_name(k_[il],   ("cache_k_l"   + std::to_string(il)).c_str());
            ggml_set_name(v_[il],   ("cache_v_l"   + std::to_string(il)).c_str());
            ggml_set_name(idx_[il], ("cache_idx_l" + std::to_string(il)).c_str());
        } else {
            // [head_v, head_v, n_v_heads] -- independent of context length
            rs_[il]   = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, head_v, head_v, n_v_heads);
            conv_[il] = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, hp->ssm_d_conv - 1, conv_dim);
            ggml_set_name(rs_[il],   ("cache_rs_l"   + std::to_string(il)).c_str());
            ggml_set_name(conv_[il], ("cache_conv_l" + std::to_string(il)).c_str());
        }
    }

    // The PLE convolution is dilated by the n-gram size, so its history is longer.
    if (hp->ple_n_head() > 0) {
        const int64_t hist = (int64_t) (hp->ple_conv_kernel - 1) * hp->ple_ngram_size;
        ple_conv_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, hist, hc_dim);
        ggml_set_name(ple_conv_, "cache_ple_conv");
    }
}

bool state::alloc_buffers(ggml_backend_buffer_type_t buft, std::string & err) {
    assert_that(ctx_ != nullptr && buf_ == nullptr, "state tensors declared, not yet allocated");
    assert_that(buf_h_ == nullptr, "host state not yet allocated");
    buf_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft);
    if (!buf_) { err = "failed to allocate state buffer"; return false; }
    bytes_ = ggml_backend_buffer_get_size(buf_);
    if (ctx_h_) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        ggml_backend_buffer_type_t hbuft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (!hbuft) { err = "no pinned host buffer type on this backend for --state-host"; return false; }
        buf_h_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h_, hbuft);
        if (!buf_h_) { err = "failed to allocate the host state buffer"; return false; }
        bytes_h_ = ggml_backend_buffer_get_size(buf_h_);
        fprintf(stderr, "[qwfn] state: %s%s%s in pinned host memory (%s), %.2f GB\n",
                kv_host_ ? "KV cache" : "", kv_host_ && idx_host_ ? " + " : "", idx_host_ ? "indexer cache" : "",
                ggml_backend_buft_name(hbuft), bytes_h_ / 1e9);
    }
    return true;
}

void state::reset() {
    assert_that(rs_.size() == conv_.size(), "recurrent tables sized together");
    assert_that(!ple_conv_ || buf_, "the PLE history is allocated before it is cleared");
    for (size_t il = 0; il < rs_.size(); il++) {
        if (rs_[il])   ggml_backend_tensor_memset(rs_[il],   0, 0, ggml_nbytes(rs_[il]));
        if (conv_[il]) ggml_backend_tensor_memset(conv_[il], 0, 0, ggml_nbytes(conv_[il]));
    }
    if (ple_conv_) ggml_backend_tensor_memset(ple_conv_, 0, 0, ggml_nbytes(ple_conv_));
}

ggml_tensor * state::k_cache(uint32_t il) const {
    assert_that(caches_sized_together(), "per-layer tables sized together");
    assert_that(hp_ != nullptr || k_.empty(), "tables exist only after init");
    return il < k_.size() ? k_[il] : nullptr;
}
ggml_tensor * state::v_cache(uint32_t il) const {
    assert_that(caches_sized_together(), "per-layer tables sized together");
    assert_that(hp_ != nullptr || v_.empty(), "tables exist only after init");
    return il < v_.size() ? v_[il] : nullptr;
}
ggml_tensor * state::idx_cache(uint32_t il) const {
    assert_that(caches_sized_together(), "per-layer tables sized together");
    assert_that(hp_ != nullptr || idx_.empty(), "tables exist only after init");
    return il < idx_.size() ? idx_[il] : nullptr;
}
ggml_tensor * state::rs_state(uint32_t il) const {
    assert_that(caches_sized_together(), "per-layer tables sized together");
    assert_that(hp_ != nullptr || rs_.empty(), "tables exist only after init");
    return il < rs_.size() ? rs_[il] : nullptr;
}
ggml_tensor * state::rs_conv(uint32_t il) const {
    assert_that(caches_sized_together(), "per-layer tables sized together");
    assert_that(hp_ != nullptr || conv_.empty(), "tables exist only after init");
    return il < conv_.size() ? conv_[il] : nullptr;
}
ggml_tensor * state::ple_conv()             const { return ple_conv_; }

bool state::caches_sized_together() const {
    return k_.size() == v_.size() && v_.size() == idx_.size() && idx_.size() == rs_.size() && rs_.size() == conv_.size();
}

std::string state::summary() const {
    assert_that(caches_sized_together(), "per-layer tables sized together");
    size_t kv = 0, ix = 0, rs = 0, cv = 0;
    for (size_t i = 0; i < k_.size(); i++) {
        if (k_[i])    kv += ggml_nbytes(k_[i]) + ggml_nbytes(v_[i]);
        if (idx_[i])  ix += ggml_nbytes(idx_[i]);
        if (rs_[i])   rs += ggml_nbytes(rs_[i]);
        if (conv_[i]) cv += ggml_nbytes(conv_[i]);
    }
    if (ple_conv_) cv += ggml_nbytes(ple_conv_);
    std::ostringstream o;
    o << "state @ " << cfg_.n_ctx << " ctx: "
      << "KV " << kv / 1e9 << " GB (" << ggml_type_name(cfg_.type_k) << "), "
      << "indexer " << ix / 1e9 << " GB, "
      << "deltanet " << rs / 1e6 << " MB (constant), "
      << "conv " << cv / 1e6 << " MB  =>  total " << bytes_ / 1e9 << " GB on the device";
    if (bytes_h_) o << " + " << bytes_h_ / 1e9 << " GB in pinned host memory";
    assert_that(!o.fail(), "summary stream written");
    return o.str();
}

} // namespace qwfn
