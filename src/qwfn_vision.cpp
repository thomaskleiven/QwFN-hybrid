#include "qwfn_vision.h"
#include "qwfn_check.h"

#include "gguf.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#include "stb_image.h"

#include <algorithm>
#include <cmath>

namespace qwfn {

// ---- image loading ----------------------------------------------------------

bool image_u8::load(const std::string & path, std::string & err) {
    int w = 0, h = 0, c = 0;
    unsigned char * data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) { err = "cannot decode image " + path + ": " + stbi_failure_reason(); return false; }
    assert_that(w > 0 && h > 0, "stb_image decodes a non-empty image");
    assert_that(c >= 1 && c <= 4, "stb_image reports 1..4 source channels");
    nx = w; ny = h;
    rgb.assign(data, data + (size_t) w * h * 3);
    stbi_image_free(data);
    return true;
}

bool image_u8::load_memory(const uint8_t * data, size_t n, std::string & err) {
    assert_that(data != nullptr || n == 0, "an image buffer, or an empty one");
    int w = 0, h = 0, c = 0;
    unsigned char * px = stbi_load_from_memory(data, (int) n, &w, &h, &c, 3);
    if (!px) { err = std::string("cannot decode image: ") + stbi_failure_reason(); return false; }
    assert_that(w > 0 && h > 0, "stb_image decodes a non-empty image");
    nx = w; ny = h;
    rgb.assign(px, px + (size_t) w * h * 3);
    stbi_image_free(px);
    return true;
}

// ---- preprocessing ----------------------------------------------------------

// "smart resize" from the transformers reference: preserve aspect, align to a
// multiple of patch*merge, then pull inside the token budget.
static void smart_resize(int w, int h, int align, int min_px, int max_px,
                         int & w_out, int & h_out) {
    assert_that(w > 0 && h > 0, "smart_resize of a non-empty image");
    assert_that(min_px <= max_px, "smart_resize budget is ordered");
    auto round_by = [&](float x) { return (int) std::lround(x / align) * align; };
    auto ceil_by  = [&](float x) { return (int) std::ceil (x / align) * align; };
    auto floor_by = [&](float x) { return (int) std::floor(x / align) * align; };

    w_out = std::max(align, round_by((float) w));
    h_out = std::max(align, round_by((float) h));

    if ((int64_t) w_out * h_out > max_px) {
        const float beta = std::sqrt((float) h * w / max_px);
        w_out = std::max(align, floor_by(w / beta));
        h_out = std::max(align, floor_by(h / beta));
    } else if ((int64_t) w_out * h_out < min_px) {
        const float beta = std::sqrt((float) min_px / ((float) h * w));
        w_out = ceil_by(w * beta);
        h_out = ceil_by(h * beta);
    }
}

// rule 5 deviation: called ~60 times per output pixel; resize_bicubic asserts its tap range once.
static float cubic(float x) {                       // Catmull-Rom / a = -0.5
    x = std::fabs(x);
    if (x <= 1.0f) return ((1.5f * x - 2.5f) * x) * x + 1.0f;
    if (x <  2.0f) return (((-0.5f * x + 2.5f) * x - 4.0f) * x) + 2.0f;
    return 0.0f;
}

// Bicubic to match the reference preprocessor (RESIZE_ALGO_BICUBIC).
static void resize_bicubic(const image_u8 & src, int dw, int dh, std::vector<uint8_t> & dst) {
    assert_that(src.nx > 0 && src.ny > 0 && src.rgb.size() == (size_t) src.nx * src.ny * 3, "a decoded RGB source");
    assert_that(dw > 0 && dh > 0, "a non-empty target size");
    dst.assign((size_t) dw * dh * 3, 0);
    const float sx = (float) src.nx / dw, sy = (float) src.ny / dh;
    // Every tap offset is frac - m with frac in [0, 1) and m in [-1, 2]: |offset| <= 2.
    assert_that(std::isfinite(sx) && std::isfinite(sy) && sx > 0 && sy > 0, "finite, positive scale factors");
    for (int y = 0; y < dh; y++) {
        const float fy = (y + 0.5f) * sy - 0.5f;
        const int   iy = (int) std::floor(fy);
        for (int x = 0; x < dw; x++) {
            const float fx = (x + 0.5f) * sx - 0.5f;
            const int   ix = (int) std::floor(fx);
            for (int c = 0; c < 3; c++) {
                float acc = 0, wsum = 0;
                for (int m = -1; m <= 2; m++) {
                    const int py = std::clamp(iy + m, 0, src.ny - 1);
                    const float wy = cubic(fy - (iy + m));
                    for (int n = -1; n <= 2; n++) {
                        const int px = std::clamp(ix + n, 0, src.nx - 1);
                        const float w = wy * cubic(fx - (ix + n));
                        acc  += w * src.rgb[((size_t) py * src.nx + px) * 3 + c];
                        wsum += w;
                    }
                }
                const float v = wsum > 0 ? acc / wsum : 0.0f;
                dst[((size_t) y * dw + x) * 3 + c] = (uint8_t) std::clamp(v, 0.0f, 255.0f);
            }
        }
    }
}

void vision_encoder::plan(int nx, int ny, int & grid_w, int & grid_h) const {
    assert_that(ctx_ != nullptr, "plan after load");
    assert_that(nx > 0 && ny > 0, "plan of a non-empty image");
    // Budget from the reference: 8..4096 tokens, one per (patch*merge)^2 pixels.
    const int patch_area = (int) (hp_.patch * hp_.patch * hp_.merge * hp_.merge);
    int w = 0, h = 0;
    smart_resize(nx, ny, (int) hp_.align(), 8 * patch_area, 4096 * patch_area, w, h);
    grid_w = w / (int) hp_.align();
    grid_h = h / (int) hp_.align();
}

// ---- loading ----------------------------------------------------------------

vision_encoder::~vision_encoder() {
    assert_that(dbuf_ == nullptr, "no encode is staged at destruction");
    assert_that(galloc_ == nullptr || ctx_ != nullptr, "an allocator exists only after a load");
    if (galloc_) ggml_gallocr_free(galloc_);
    if (buf_)    ggml_backend_buffer_free(buf_);
    if (ctx_)    ggml_free(ctx_);
}

ggml_tensor * vision_encoder::get(const std::string & name) const {
    return ggml_get_tensor(ctx_, name.c_str());
}

void vision_encoder::set_n_threads(int n) {
    if (!backend_ || n <= 0) return;
    assert_that(backend_ != nullptr && n > 0, "a backend and a positive thread count");
    assert_that(ggml_backend_get_device(backend_) != nullptr, "the backend has a device");
    // The CPU module is loaded dynamically, so its setter comes from the registry.
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_));
    auto fn = reg ? (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads") : nullptr;
    if (fn) fn(backend_, n);
}

void vision_encoder::read_hparams(const gguf_context * gc) {
    assert_that(gc != nullptr, "read_hparams has an open gguf");
    assert_that(ctx_ == nullptr, "hparams are read before the tensors are created");
    auto u32 = [&](const char * k, uint32_t & dst) {
        const int64_t i = gguf_find_key(gc, k);
        if (i >= 0) dst = (uint32_t) gguf_get_val_u32(gc, i);
    };
    auto f32 = [&](const char * k, float & dst) {
        const int64_t i = gguf_find_key(gc, k);
        if (i >= 0) dst = gguf_get_val_f32(gc, i);
    };
    u32("clip.vision.embedding_length",       hp_.n_embd);
    u32("clip.vision.feed_forward_length",    hp_.n_ff);
    u32("clip.vision.attention.head_count",   hp_.n_head);
    u32("clip.vision.block_count",            hp_.n_layer);
    u32("clip.vision.patch_size",             hp_.patch);
    u32("clip.vision.image_size",             hp_.image_size);
    u32("clip.vision.projection_dim",         hp_.proj_dim);
    hp_.merge = 2;
    u32("clip.vision.spatial_merge_size",     hp_.merge);
    f32("clip.vision.attention.layer_norm_epsilon", hp_.eps);
    {
        const int64_t i = gguf_find_key(gc, "clip.vision.image_mean");
        if (i >= 0 && gguf_get_arr_n(gc, i) == 3) {
            const float * a = (const float *) gguf_get_arr_data(gc, i);
            for (int k = 0; k < 3; k++) hp_.mean[k] = a[k];
        }
        const int64_t j = gguf_find_key(gc, "clip.vision.image_std");
        if (j >= 0 && gguf_get_arr_n(gc, j) == 3) {
            const float * a = (const float *) gguf_get_arr_data(gc, j);
            for (int k = 0; k < 3; k++) hp_.std_[k] = a[k];
        }
    }
}

// Rebuild the tensors in our own context so we own their lifetime.
void vision_encoder::create_tensors(const gguf_context * gc, ggml_context * meta, int n_tensors) {
    assert_that(gc != nullptr && meta != nullptr, "create_tensors has the gguf and its metadata");
    assert_that(ctx_ == nullptr && n_tensors >= 0, "create_tensors runs once, over the file's tensors");
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (n_tensors + 8);
    ip.no_alloc = true;
    ctx_ = ggml_init(ip); assert_that(ctx_ != nullptr, "ggml_init: ctx_");

    // On the CPU the linear layers are converted to BF16 at load: the tiled GEMM
    // has an AVX-512 BF16 path, measured 17.2 -> 15.4 s on a 1400x1000 screenshot
    // with embeddings at rounding level (first values move in the 4th decimal).
    // Q8_0 was no faster (16.0 s) and cannot take ffn_down (4304 is not a
    // multiple of 32).
    const ggml_type wtype = cpu_ ? GGML_TYPE_BF16 : GGML_TYPE_COUNT;
    auto is_linear = [](const std::string & n) {
        return n.size() > 7 && n.compare(n.size() - 7, 7, ".weight") == 0 &&
               n != "v.position_embd.weight" && n.rfind("v.patch_embd", 0) != 0;
    };
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gc, i);
        ggml_tensor * src = ggml_get_tensor(meta, name);
        ggml_type ty = src->type;
        if (wtype != GGML_TYPE_COUNT && src->type == GGML_TYPE_F16 && ggml_n_dims(src) == 2 && is_linear(name) &&
            src->ne[0] % ggml_blck_size(wtype) == 0) ty = wtype;
        ggml_tensor * dst = ggml_new_tensor(ctx_, ty, ggml_n_dims(src), src->ne);
        ggml_set_name(dst, name);
    }
}

// Stream every tensor's data from the file into the weight buffer, converting where
// create_tensors changed the type. On failure `err` is set and the file is closed.
bool vision_encoder::read_weights(const std::string & path, const gguf_context * gc, int n_tensors,
                                  std::string & err) {
    assert_that(gc != nullptr && ctx_ != nullptr, "read_weights has the gguf and the tensor context");
    assert_that(buf_ != nullptr, "read_weights has an allocated weight buffer");
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot reopen mmproj"; return false; }
    const size_t data_off = gguf_get_data_offset(gc);
    std::vector<uint8_t> tmp;
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gc, i);
        ggml_tensor * dst = get(name);
        const size_t nb   = gguf_get_tensor_size(gc, i);
        tmp.resize(nb);
        if (fseek(f, (long) (data_off + gguf_get_tensor_offset(gc, i)), SEEK_SET) != 0 ||
            fread(tmp.data(), 1, nb, f) != nb) {
            err = std::string("short read for mmproj tensor ") + name;
            fclose(f); return false;
        }
        if (dst->type != gguf_get_tensor_type(gc, i)) {
            const int64_t n = ggml_nelements(dst);
            std::vector<float> as_f32(n);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) tmp.data(), as_f32.data(), n);
            std::vector<uint8_t> conv(ggml_nbytes(dst));
            if (dst->type == GGML_TYPE_BF16) ggml_fp32_to_bf16_row(as_f32.data(), (ggml_bf16_t *) conv.data(), n);
            else ggml_quantize_chunk(dst->type, as_f32.data(), conv.data(), 0, dst->ne[1], dst->ne[0], nullptr);
            ggml_backend_tensor_set(dst, conv.data(), 0, conv.size());
        } else ggml_backend_tensor_set(dst, tmp.data(), 0, nb);
    }
    fclose(f);
    return true;
}

// Bind the named weights.
bool vision_encoder::bind_weights(std::string & err) {
    assert_that(ctx_ != nullptr && buf_ != nullptr, "bind_weights after the weights are read");
    assert_that(hp_.n_layer > 0, "load checked the block count");
    pe0_        = get("v.patch_embd.weight");
    pe1_        = get("v.patch_embd.weight.1");
    patch_bias_ = get("v.patch_embd.bias");
    pos_embd_   = get("v.position_embd.weight");
    post_ln_w_  = get("v.post_ln.weight");
    post_ln_b_  = get("v.post_ln.bias");
    mm0_w_      = get("mm.0.weight");   mm0_b_ = get("mm.0.bias");
    mm2_w_      = get("mm.2.weight");   mm2_b_ = get("mm.2.bias");
    if (!pe0_ || !pos_embd_ || !mm0_w_ || !mm2_w_) {
        err = "mmproj is missing expected tensors"; return false;
    }

    layers_.resize(hp_.n_layer);
    for (uint32_t il = 0; il < hp_.n_layer; il++) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        layer & L = layers_[il];
        L.ln1_w = get(p + "ln1.weight");     L.ln1_b = get(p + "ln1.bias");
        L.qkv_w = get(p + "attn_qkv.weight"); L.qkv_b = get(p + "attn_qkv.bias");
        L.out_w = get(p + "attn_out.weight"); L.out_b = get(p + "attn_out.bias");
        L.ln2_w = get(p + "ln2.weight");     L.ln2_b = get(p + "ln2.bias");
        L.up_w  = get(p + "ffn_up.weight");  L.up_b  = get(p + "ffn_up.bias");
        L.down_w= get(p + "ffn_down.weight");L.down_b= get(p + "ffn_down.bias");
        if (!L.qkv_w || !L.up_w) { err = "mmproj block " + std::to_string(il) + " incomplete"; return false; }
    }
    return true;
}

// Flash attention at this head size (72), on this backend? Ask it about the
// exact op the graph will carry; otherwise attention materialises its scores.
// The CPU backend says yes but its kernel is a per-key loop: 52 s against
// 17 s materialised on a 1400x1000 screenshot, so there attention goes
// through the GEMMs, chunked over the queries to bound the score matrix.
void vision_encoder::probe_flash_attn() {
    assert_that(!cpu_, "the flash-attention probe is for device backends");
    assert_that(backend_ != nullptr, "probe_flash_attn has a backend");
    ggml_init_params pp{}; pp.mem_size = ggml_tensor_overhead() * 8; pp.no_alloc = true;
    ggml_context * pc = ggml_init(pp); assert_that(pc != nullptr, "ggml_init: pc");
    const int64_t d = hp_.d_head(), n = 64, h = hp_.n_head;
    ggml_tensor * q = ggml_new_tensor_3d(pc, GGML_TYPE_F32, d, n, h);
    ggml_tensor * k = ggml_new_tensor_3d(pc, GGML_TYPE_F16, d, n, h);
    ggml_tensor * v = ggml_new_tensor_3d(pc, GGML_TYPE_F16, d, n, h);
    ggml_tensor * o = ggml_flash_attn_ext(pc, q, k, v, nullptr, 1.0f / std::sqrt((float) d), 0.0f, 0.0f);
    assert_that(ggml_prec_set_acc(o, GGML_PREC_F32), "flash attention accepts F32 accumulation");
    use_fa_ = ggml_backend_supports_op(backend_, o);
    ggml_free(pc);
}

bool vision_encoder::load(const std::string & path, ggml_backend_t backend,
                          ggml_backend_buffer_type_t buft, std::string & err) {
    assert_that(backend != nullptr && buft != nullptr, "load is given a backend and its buffer type");
    assert_that(ctx_ == nullptr && buf_ == nullptr, "a vision_encoder loads once");
    backend_ = backend;
    buft_    = buft;
    stage_   = false;
    cpu_     = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;

    // no_alloc: read metadata first, then place the tensors on the backend and
    // stream the data in. Loading into host memory first would cost 0.9 GB.
    ggml_context * meta = nullptr;
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx      = &meta;
    gguf_context * gc = gguf_init_from_file(path.c_str(), gp);
    if (!gc) { err = "failed to open mmproj " + path; return false; }

    read_hparams(gc);
    if (!hp_.n_embd || !hp_.n_layer || !hp_.n_head) {
        err = "mmproj is missing vision hyper-parameters"; gguf_free(gc); ggml_free(meta); return false;
    }

    const int n_tensors = (int) gguf_get_n_tensors(gc);
    create_tensors(gc, meta, n_tensors);

    // The weights' home is host memory, pinned when the backend offers it, and
    // they are staged per image; on a CPU backend the host buffer is the
    // compute buffer and nothing is staged.
    ggml_backend_buffer_type_t wbuft = buft_;
    {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft_);
        ggml_backend_buffer_type_t h = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (h && h != buft_) { wbuft = h; stage_ = true; }
    }
    buf_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, wbuft);
    if (!buf_) { err = "failed to allocate mmproj weights"; gguf_free(gc); ggml_free(meta); return false; }

    if (!read_weights(path, gc, n_tensors, err)) { gguf_free(gc); ggml_free(meta); return false; }
    gguf_free(gc);
    ggml_free(meta);

    if (!bind_weights(err)) return false;

    galloc_ = ggml_gallocr_new(buft_); assert_that(galloc_ != nullptr, "ggml_gallocr_new: galloc_");
    if (!cpu_) probe_flash_attn();
    fprintf(stderr, "[qwfn] vision: %u blocks, n_embd %u, patch %u, merge %u -> %u, attention: %s, weights %s\n",
            hp_.n_layer, hp_.n_embd, hp_.patch, hp_.merge, hp_.proj_dim,
            use_fa_ ? "flash" : cpu_ ? "materialised scores, chunked" : "materialised scores",
            cpu_ ? "in RAM (linears BF16), computed on the CPU"
                 : stage_ ? "in host memory, staged per image" : "on the device");
    return true;
}

// ---- staging: weights host -> device for one encode -------------------------

bool vision_encoder::stage_in(std::string & err) {
    assert_that(ctx_ != nullptr && buft_ != nullptr, "stage_in after load");
    if (!stage_ || dbuf_) return true;
    const size_t align = ggml_backend_buft_get_alignment(buft_);
    auto padded = [&](size_t n) { return (n + align - 1) / align * align; };
    size_t total = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t)) total += padded(ggml_nbytes(t));
    dbuf_ = ggml_backend_buft_alloc_buffer(buft_, total);
    if (!dbuf_) {
        err = "not enough free VRAM to stage the vision projector (" + std::to_string(total >> 20) + " MB)";
        return false;
    }
    ggml_backend_buffer_set_usage(dbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    char * base = (char *) ggml_backend_buffer_get_base(dbuf_);
    size_t off = 0;
    saved_.clear();
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t)) {
        const size_t nb = ggml_nbytes(t);
        void * host = t->data;
        saved_.emplace_back(t->data, t->buffer);
        t->buffer = dbuf_;
        t->data   = base + off;
        ggml_backend_tensor_set(t, host, 0, nb);   // a DMA from pinned memory
        off += padded(nb);
    }
    assert_that(off == total, "the staged tensors fill exactly the buffer sized for them");
    return true;
}

void vision_encoder::stage_out() {
    if (!dbuf_) return;
    assert_that(ctx_ != nullptr && !saved_.empty(), "stage_out restores a staged placement");
    size_t i = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t), i++) {
        t->data = saved_[i].first; t->buffer = saved_[i].second;
    }
    assert_that(i == saved_.size(), "every staged tensor is restored");
    ggml_backend_buffer_free(dbuf_); dbuf_ = nullptr;
    // The activation arena was carved from the same borrowed VRAM: return it too,
    // or the tier cannot take its dynamic buffer back after the prefill.
    if (galloc_) { ggml_gallocr_free(galloc_); galloc_ = ggml_gallocr_new(buft_); assert_that(galloc_ != nullptr, "ggml_gallocr_new: vision"); }
}

// ---- encode: inputs ---------------------------------------------------------

// planar [W,H,3], normalised
void vision_encoder::set_pixels(ggml_tensor * inp_raw, const std::vector<uint8_t> & px, int W, int H) const {
    assert_that(inp_raw != nullptr, "set_pixels has an input tensor");
    assert_that(px.size() == (size_t) W * H * 3, "the resized image is W x H RGB");
    std::vector<float> f((size_t) W * H * 3);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                f[((size_t) c * H + y) * W + x] =
                    (px[((size_t) y * W + x) * 3 + c] / 255.0f - hp_.mean[c]) / hp_.std_[c];
    ggml_backend_tensor_set(inp_raw, f.data(), 0, f.size() * sizeof(float));
}

// M-RoPE positions: 2x2 block order, [h,w,h,w] planes of n_pos each.
void vision_encoder::set_positions(ggml_tensor * positions, int pw, int ph) const {
    assert_that(positions != nullptr, "set_positions has a positions tensor");
    assert_that(ggml_nelements(positions) == (int64_t) pw * ph * 4, "positions hold four planes of pw*ph");
    const int n_pos = pw * ph;
    std::vector<int32_t> p((size_t) n_pos * 4);
    int ptr = 0;
    for (int y = 0; y < ph; y += (int) hp_.merge)
        for (int x = 0; x < pw; x += (int) hp_.merge)
            for (int dy = 0; dy < (int) hp_.merge; dy++)
                for (int dx = 0; dx < (int) hp_.merge; dx++) {
                    p[            ptr] = y + dy;
                    p[  n_pos +   ptr] = x + dx;
                    p[2*n_pos +   ptr] = y + dy;
                    p[3*n_pos +   ptr] = x + dx;
                    ptr++;
                }
    ggml_backend_tensor_set(positions, p.data(), 0, p.size() * sizeof(int32_t));
}

// Learned position embeddings, bilinearly resized on the host from the
// 48x48 grid the checkpoint stores, in raster order. The graph then applies
// the same 2x2 interleave it applies to the patches.
void vision_encoder::set_pos_embd(ggml_tensor * pos_in, int pw, int ph) const {
    assert_that(pos_in != nullptr && pos_embd_ != nullptr, "set_pos_embd has its source and target");
    assert_that(ggml_nelements(pos_in) == (int64_t) hp_.n_embd * pw * ph, "pos_in is [n_embd, pw, ph]");
    const int64_t n_embd = hp_.n_embd;
    const int n_pos = pw * ph;
    const int src_side = (int) std::lround(std::sqrt((double) pos_embd_->ne[1]));
    std::vector<float> src((size_t) pos_embd_->ne[0] * pos_embd_->ne[1]);
    ggml_backend_tensor_get(pos_embd_, src.data(), 0, src.size() * sizeof(float));
    std::vector<float> dst((size_t) n_embd * n_pos);
    // align_corners bilinear, matching GGML_SCALE_FLAG_ALIGN_CORNERS
    auto coord = [](int i, int n_dst, int n_src) {
        return n_dst > 1 ? (float) i * (n_src - 1) / (n_dst - 1) : 0.0f;
    };
    for (int y = 0; y < ph; y++) {
        const float fy = coord(y, ph, src_side);
        const int y0 = (int) fy, y1 = std::min(y0 + 1, src_side - 1);
        const float wy = fy - y0;
        for (int x = 0; x < pw; x++) {
            const float fx = coord(x, pw, src_side);
            const int x0 = (int) fx, x1 = std::min(x0 + 1, src_side - 1);
            const float wx = fx - x0;
            const size_t o = (size_t) (y * pw + x) * n_embd;
            const size_t a = (size_t) (y0 * src_side + x0) * n_embd;
            const size_t b = (size_t) (y0 * src_side + x1) * n_embd;
            const size_t c = (size_t) (y1 * src_side + x0) * n_embd;
            const size_t d = (size_t) (y1 * src_side + x1) * n_embd;
            for (int64_t k = 0; k < n_embd; k++)
                dst[o + k] = (1 - wy) * ((1 - wx) * src[a + k] + wx * src[b + k])
                           +      wy  * ((1 - wx) * src[c + k] + wx * src[d + k]);
        }
    }
    ggml_backend_tensor_set(pos_in, dst.data(), 0, dst.size() * sizeof(float));
}

// ---- encode: graph ----------------------------------------------------------

// Materialised attention is chunked over the queries so one chunk's scores
// [n_pos, chunk, n_head] stay under ~256 MB: 768 queries for a 1400x1000
// screenshot (5,456 patches), 256 for the largest image the token budget
// allows (16,384 patches, where the full matrix would be 17 GB).
int vision_encoder::attn_chunk(int n_pos) const {
    assert_that(n_pos > 0 && hp_.n_head > 0, "attn_chunk of a non-empty patch grid");
    int chunk = n_pos;
    if (!use_fa_) {
        const int64_t budget = 256ll << 20;
        chunk = (int) std::max<int64_t>(64, std::min<int64_t>(n_pos, budget / ((int64_t) n_pos * hp_.n_head * sizeof(float))));
        chunk = (chunk + 15) / 16 * 16;
    }
    assert_that(chunk > 0, "attention runs in chunks of at least one query");
    return chunk;
}

ggml_tensor * vision_encoder::norm(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) const {
    assert_that(c != nullptr, "norm has a graph context");
    assert_that(x != nullptr, "norm has an input");
    x = ggml_norm(c, x, hp_.eps);
    x = ggml_mul(c, x, w);
    return b ? ggml_add(c, x, b) : x;
}

// The 2x2 interleave the tower expects: patches within a merge block become
// adjacent, so the merger's reshape groups the right four.
ggml_tensor * vision_encoder::interleave(ggml_context * c, ggml_tensor * t, int pw, int ph) const {
    assert_that(c != nullptr && t != nullptr, "interleave has a graph context and an input");
    assert_that(pw > 0 && ph > 0, "interleave of a non-empty patch grid");
    const int64_t n_embd = hp_.n_embd;
    t = ggml_cont_4d(c, t, n_embd * 2, pw / 2, ph, 1);
    t = ggml_reshape_4d(c, t, n_embd * 2, pw / 2, 2, ph / 2);
    t = ggml_permute(c, t, 0, 2, 1, 3);
    return ggml_cont_3d(c, t, n_embd, (int64_t) pw * ph, 1);
}

// Full bidirectional attention, no mask. Flash attention, as llama.cpp's
// clip does it: the [n_pos, n_pos, n_head] score matrix is never
// materialised. It was -- 1.9 GB for a 1,364-token screenshot, 17 GB
// for the largest image the token budget allows -- and when that
// allocation failed the next image crashed inside the allocator.
ggml_tensor * vision_encoder::attention(ggml_context * c, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V,
                                        int n_pos, int chunk) const {
    assert_that(c != nullptr && Q != nullptr && K != nullptr && V != nullptr, "attention has q, k and v");
    assert_that(n_pos > 0 && chunk > 0, "attention over a non-empty grid in positive chunks");
    const int64_t n_embd = hp_.n_embd, n_head = hp_.n_head, d_head = hp_.d_head();
    const float kq_scale = 1.0f / std::sqrt((float) d_head);
    ggml_tensor * q = ggml_permute(c, Q, 0, 2, 1, 3);                 // [d,pos,head]
    if (use_fa_) {
        ggml_tensor * k = ggml_cast(c, ggml_permute(c, K, 0, 2, 1, 3), GGML_TYPE_F16);
        ggml_tensor * v = ggml_cast(c, ggml_permute(c, V, 0, 2, 1, 3), GGML_TYPE_F16);
        ggml_tensor * kqv = ggml_flash_attn_ext(c, q, k, v, nullptr, kq_scale, 0.0f, 0.0f);   // [d,head,pos]
        assert_that(ggml_prec_set_acc(kqv, GGML_PREC_F32), "flash attention accepts F32 accumulation");
        return ggml_reshape_2d(c, kqv, n_embd, n_pos);
    }
    // The CPU's tiled GEMM wants the reduction length a multiple of 16
    // and both operands contiguous: the head size 72 is padded to 80
    // with zeros (the dot products are unchanged), Q and K are laid out
    // [d,pos,head], and each query chunk is copied out contiguous.
    // Without the padding the scores fell to the per-row dot path.
    // (BF16 scores, padded to 96 for the BF16 GEMM, measured 15.6 -> 15.0 s: not taken.)
    const int64_t d_pad = (d_head + 15) / 16 * 16;
    ggml_tensor * Qp = d_pad != d_head ? ggml_pad(c, Q, (int) (d_pad - d_head), 0, 0, 0) : Q;
    ggml_tensor * Kp = d_pad != d_head ? ggml_pad(c, K, (int) (d_pad - d_head), 0, 0, 0) : K;
    ggml_tensor * qa = ggml_cont(c, ggml_permute(c, Qp, 0, 2, 1, 3));   // [d_pad,pos,head]
    ggml_tensor * k  = ggml_cont(c, ggml_permute(c, Kp, 0, 2, 1, 3));   // [d_pad,pos,head]
    ggml_tensor * v  = ggml_cont(c, ggml_permute(c, V, 1, 2, 0, 3));    // [pos,d,head]
    ggml_tensor * acc = nullptr;
    for (int s0 = 0; s0 < n_pos; s0 += chunk) {
        const int n = std::min(chunk, n_pos - s0);
        ggml_tensor * qi = ggml_cont(c, ggml_view_3d(c, qa, d_pad, n, n_head, qa->nb[1], qa->nb[2], (size_t) s0 * qa->nb[1]));
        ggml_tensor * kq = ggml_mul_mat(c, k, qi);                        // [pos,n,head]
        kq = ggml_soft_max_ext(c, kq, nullptr, kq_scale, 0.0f);
        ggml_tensor * oi = ggml_mul_mat(c, v, kq);                        // [d,n,head]
        oi = ggml_cont_2d(c, ggml_permute(c, oi, 0, 2, 1, 3), n_embd, n);
        acc = acc ? ggml_concat(c, acc, oi, 1) : oi;
    }
    return acc;                                                           // [n_embd,pos]
}

// One pre-norm transformer block: M-RoPE attention, then the GELU MLP.
ggml_tensor * vision_encoder::block(ggml_context * c, const layer & L, ggml_tensor * cur, ggml_tensor * positions,
                                    int * sections, int n_pos, int chunk) const {
    assert_that(c != nullptr && cur != nullptr && positions != nullptr, "block has a context, an input and positions");
    assert_that(L.qkv_w != nullptr && L.up_w != nullptr, "load bound this block's weights");
    const int64_t n_embd = hp_.n_embd, n_head = hp_.n_head, d_head = hp_.d_head();
    ggml_tensor * res = cur;

    ggml_tensor * x = norm(c, cur, L.ln1_w, L.ln1_b);
    x = ggml_add(c, ggml_mul_mat(c, L.qkv_w, x), L.qkv_b);

    ggml_tensor * Q = ggml_view_3d(c, x, d_head, n_head, n_pos,
                                   ggml_row_size(x->type, d_head), x->nb[1], 0);
    ggml_tensor * K = ggml_view_3d(c, x, d_head, n_head, n_pos,
                                   ggml_row_size(x->type, d_head), x->nb[1],
                                   ggml_row_size(x->type, n_embd));
    ggml_tensor * V = ggml_view_3d(c, x, d_head, n_head, n_pos,
                                   ggml_row_size(x->type, d_head), x->nb[1],
                                   ggml_row_size(x->type, 2 * n_embd));

    Q = ggml_rope_multi(c, ggml_cont(c, Q), positions, nullptr, (int) d_head / 2,
                        sections, GGML_ROPE_TYPE_VISION, 32768, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    K = ggml_rope_multi(c, ggml_cont(c, K), positions, nullptr, (int) d_head / 2,
                        sections, GGML_ROPE_TYPE_VISION, 32768, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    ggml_tensor * kqv = attention(c, Q, K, V, n_pos, chunk);

    x = ggml_add(c, ggml_mul_mat(c, L.out_w, kqv), L.out_b);
    cur = ggml_add(c, x, res);

    res = cur;
    x = norm(c, cur, L.ln2_w, L.ln2_b);
    x = ggml_add(c, ggml_mul_mat(c, L.up_w, x), L.up_b);
    x = ggml_gelu(c, x);
    x = ggml_add(c, ggml_mul_mat(c, L.down_w, x), L.down_b);
    return ggml_add(c, x, res);
}

// Patch embedding, the blocks, the post-norm and the merger: the projected
// embeddings [proj_dim, n_pos/4], marked as the graph output.
ggml_tensor * vision_encoder::build_tower(ggml_context * c, ggml_tensor * inp_raw, ggml_tensor * positions,
                                          ggml_tensor * pos_in, int pw, int ph, int chunk) const {
    assert_that(c != nullptr && inp_raw != nullptr && positions != nullptr && pos_in != nullptr, "build_tower has its inputs");
    assert_that(pe0_ != nullptr && layers_.size() == hp_.n_layer, "load bound the tower's weights");
    const int64_t n_embd = hp_.n_embd, d_head = hp_.d_head();
    const int n_pos = pw * ph;

    // patch embedding: two kernels summed (still image -> same frame twice)
    ggml_tensor * cur = ggml_add(c,
        ggml_conv_2d(c, pe0_, inp_raw, hp_.patch, hp_.patch, 0, 0, 1, 1),
        ggml_conv_2d(c, pe1_, inp_raw, hp_.patch, hp_.patch, 0, 0, 1, 1));
    cur = ggml_permute(c, cur, 1, 2, 0, 3);          // [w,h,c,b] -> [c,w,h,b]
    cur = interleave(c, cur, pw, ph);
    if (patch_bias_) cur = ggml_add(c, cur, patch_bias_);
    cur = ggml_add(c, cur, interleave(c, pos_in, pw, ph));

    int sections[4] = { (int) d_head / 4, (int) d_head / 4, (int) d_head / 4, (int) d_head / 4 };
    for (uint32_t il = 0; il < hp_.n_layer; il++) cur = block(c, layers_[il], cur, positions, sections, n_pos, chunk);

    cur = norm(c, cur, post_ln_w_, post_ln_b_);

    // merger: fold each 2x2 block into one vector, then the two-layer MLP
    cur = ggml_reshape_2d(c, cur, n_embd * 4, (int64_t) n_pos / 4);
    cur = ggml_add(c, ggml_mul_mat(c, mm0_w_, cur), mm0_b_);
    cur = ggml_gelu(c, cur);
    cur = ggml_add(c, ggml_mul_mat(c, mm2_w_, cur), mm2_b_);
    ggml_set_output(cur);
    return cur;
}

// Allocate, compute, read the embeddings back. On failure `err` is set; the caller frees.
bool vision_encoder::run_graph(ggml_cgraph * gf, ggml_tensor * cur, int n_pos, int n_out, int n_chunks, int chunk,
                               std::vector<float> & out, std::string & err) {
    assert_that(gf != nullptr && cur != nullptr, "run_graph has a graph and its output");
    assert_that(galloc_ != nullptr && backend_ != nullptr, "run_graph has an allocator and a backend");
    if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
        err = "vision graph allocation failed (" + std::to_string(n_pos) + " patches; not enough free VRAM for this image)";
        // A failed reserve leaves the allocator referencing buffers it has
        // freed, and the next alloc_graph dereferences them: start it over.
        ggml_gallocr_free(galloc_); galloc_ = ggml_gallocr_new(buft_); assert_that(galloc_ != nullptr, "ggml_gallocr_new: vision");
        return false;
    }
    fprintf(stderr, "[qwfn] vision graph: %d patches -> %d tokens, arena %.0f MB%s\n", n_pos, n_out,
            ggml_gallocr_get_buffer_size(galloc_, 0) / 1e6,
            use_fa_ ? " (flash attention)" : cpu_ ? (", attention in " + std::to_string(n_chunks) + " chunks of " + std::to_string(chunk)).c_str() : "");
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        err = "vision graph compute failed";
        return false;
    }

    out.resize((size_t) hp_.proj_dim * n_out);
    ggml_backend_tensor_get(cur, out.data(), 0, out.size() * sizeof(float));
    return true;
}

// ---- encode -----------------------------------------------------------------

bool vision_encoder::encode(const image_u8 & img, std::vector<float> & out,
                            int & n_out, int & grid_w, int & grid_h, std::string & err) {
    if (!ctx_) { err = "vision encoder not loaded"; return false; }
    assert_that(backend_ != nullptr && buft_ != nullptr, "a loaded encoder has a backend");
    assert_that(img.nx > 0 && img.ny > 0, "encode of a decoded image");
    // Weights onto the device for the duration of this encode (no-op when they live there).
    if (!stage_in(err)) return false;
    struct unstage { vision_encoder * v; ~unstage() { v->stage_out(); } } unstage_guard{this};

    const int patch_area = (int) (hp_.patch * hp_.patch * hp_.merge * hp_.merge);
    int W = 0, H = 0;
    smart_resize(img.nx, img.ny, (int) hp_.align(), 8 * patch_area, 4096 * patch_area, W, H);

    std::vector<uint8_t> px;
    resize_bicubic(img, W, H, px);

    const int pw = W / (int) hp_.patch,  ph = H / (int) hp_.patch;   // patch grid
    const int n_pos = pw * ph;
    grid_w = pw / (int) hp_.merge; grid_h = ph / (int) hp_.merge;
    n_out  = grid_w * grid_h;

    // ---- inputs ------------------------------------------------------------
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8;
    ip.no_alloc = true;
    ggml_context * ictx = ggml_init(ip); assert_that(ictx != nullptr, "ggml_init: ictx");
    ggml_tensor * inp_raw   = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, W, H, 3, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, n_pos * 4);
    ggml_tensor * pos_in    = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, (int64_t) hp_.n_embd, pw, ph, 1);
    ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ictx, buft_);
    if (!ibuf) { ggml_free(ictx); err = "failed to allocate vision inputs"; return false; }
    set_pixels(inp_raw, px, W, H);
    set_positions(positions, pw, ph);
    set_pos_embd(pos_in, pw, ph);

    // ---- graph -------------------------------------------------------------
    const int chunk = attn_chunk(n_pos);
    const int n_chunks = (n_pos + chunk - 1) / chunk;
    const size_t graph_size = 512 + (size_t) hp_.n_layer * (48 + (size_t) n_chunks * 14);
    ggml_init_params gp{};
    gp.mem_size = ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false);
    gp.no_alloc = true;
    ggml_context * c = ggml_init(gp); assert_that(c != nullptr, "ggml_init: c");
    ggml_cgraph * gf = ggml_new_graph_custom(c, graph_size, false);
    ggml_tensor * cur = build_tower(c, inp_raw, positions, pos_in, pw, ph, chunk);
    ggml_build_forward_expand(gf, cur);

    const bool ok = run_graph(gf, cur, n_pos, n_out, n_chunks, chunk, out, err);
    ggml_free(c);
    ggml_backend_buffer_free(ibuf);
    ggml_free(ictx);
    return ok;
}

}  // namespace qwfn
