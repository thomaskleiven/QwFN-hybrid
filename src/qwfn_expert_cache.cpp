#include "qwfn_expert_cache.h"
#include "qwfn_check.h"

#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <thread>

namespace qwfn {

namespace {
inline uint64_t xorshift(uint64_t & s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}
constexpr uint16_t SLOT_EMPTY = 0xFFFF;
// Every layer's VRAM slot arrays end with TIER_PAD zeroed bytes. CUDA's MMQ
// (the batched mul_mat_id kernel) over-reads the last row of a
// quantised matrix by up to MATRIX_ROW_PADDING elements; ffn_down is
// 640 wide (640 % 512 != 0), so the last slot of a layer's down array
// reads into whatever follows -- the next layer's gate array, another
// quant type, NaN scales. The pad is never written and the lent tail is
// re-zeroed when it comes back. (T=1 uses mmvq, which reads whole rows
// only; the bug showed on the batched path as NaN logits on the second
// turn of a chat, once some experts were VRAM-resident.)
constexpr size_t TIER_PAD = 4096;
bool any_failed(const uint64_t * tags, size_t n) {
    assert_that(tags != nullptr, "completion tag buffer");
    assert_that(n <= 256, "no more completions than the largest tag buffer holds");
    for (size_t k = 0; k < n; k++) if (tags[k] & IO_TAG_FAILED) return true;
    return false;
}
} // namespace

// 48 x ~100 MB of memcpy per ubatch is a second on one core; spread it.
void expert_cache::run_copies(const std::vector<warm_copy> & copies) {
    if (copies.empty()) return;
    const unsigned nt = std::min<unsigned>(8, std::max<unsigned>(1, std::thread::hardware_concurrency() / 2));
    assert_that(nt >= 1 && nt <= 8, "copy thread count within [1, 8]");
    assert_that(copies.front().dst != nullptr && copies.front().src != nullptr, "copy endpoints");
    if (copies.size() < 6 || nt <= 1) {
        for (const warm_copy & c : copies) memcpy(c.dst, c.src, c.n);
    } else {
        std::vector<std::thread> th;
        for (unsigned k = 0; k < nt; k++)
            th.emplace_back([&, k] { for (size_t i = k; i < copies.size(); i += nt) memcpy(copies[i].dst, copies[i].src, copies[i].n); });
        for (auto & t : th) t.join();
    }
}

// Device-tier sizing, carried between the steps of init_vram_tier().
struct expert_cache::vram_plan {
    std::vector<uint32_t> gslots;          // slots per layer
    std::vector<size_t>   nat;             // natural (unpadded) bytes per expert
    size_t   want = 0;                     // device bytes asked for, without the reservation
    size_t   lend = 0;                     // of those, the dynamic part
    uint32_t K = 0;                        // first layer whose arrays live in the dynamic buffer
    size_t   perm_bytes = 0, ext_bytes = 0;
};

// The misses of one fetch_begin() call: up to n_expert_used experts x 3 parts in flight for this layer.
struct expert_cache::fetch_ctx {
    static constexpr size_t MAX_REQ = 64 * EXPERT_NPARTS * 8;
    io_request reqs[MAX_REQ];
    bool       req_cold[MAX_REQ];
    uint32_t   miss_slot[64];
    uint32_t   miss_expert[64];
    size_t     n_req = 0, n_miss = 0;
    uint32_t   promoted = 0;
};

bool expert_cache::init(const model_index * hot, const model_index * cold,
                        const config & cfg, std::string & err) {
    assert_that(hot != nullptr, "init needs the hot model index");
    shutdown();
    assert_that(arena_ == nullptr && blk_.empty(), "shutdown released the previous tiers");
    hot_  = hot;
    cold_ = cfg.use_cold_tier ? cold : nullptr;
    cfg_  = cfg;

    if (!init_io(cfg, err)) return false;

    const uint32_t n_layer = hot->hp().n_layer;
    blk_.resize(n_layer);

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!layout_layer(il, err)) return false;
    }
    if (!size_ram_tier(err)) return false;
    if (!alloc_arena(err)) return false;
    init_ram_slots();
    // ---- T0: mirror the same block layout in device memory -----------------
    if (cfg.vram_bytes > 0 && cfg.vram_buft) init_vram_tier();
    deferred_.reserve(64);   // at most one per expert of a fetch

    arena_buf_ = ggml_backend_cpu_buffer_from_ptr(arena_, arena_bytes_);
    if (!arena_buf_) { err = "failed to wrap the expert arena in a ggml buffer"; return false; }

    fprintf(stderr, "[qwfn] expert RAM tier: %.2f GB arena%s, %zu of %u expert blocks (%.1f%%)\n",
            arena_bytes_ / 1e9, arena_pinned_ ? " (pinned)" : "", total_slots_, n_layer * hot->hp().n_expert,
            100.0 * (double) total_slots_ / (double) (n_layer * hot->hp().n_expert));
    return true;
}

bool expert_cache::init_io(const config & cfg, std::string & err) {
    assert_that(hot_ != nullptr, "hot index set before the I/O engines open");
    assert_that(!cold_ || cfg.use_cold_tier, "a cold index only with the cold tier enabled");
    if (!io_hot_.init(hot_->shard_paths(), cfg.queue_depth, /*direct_io=*/true, err, cfg.io_backend)) return false;
    {
        // The prefetch reads either file: hot shards first, the cold ones after.
        std::vector<std::string> pfp = hot_->shard_paths();
        n_hot_shards_ = (uint32_t) pfp.size();
        if (cold_) for (const auto & p : cold_->shard_paths()) pfp.push_back(p);
        if (!io_pf_.init(pfp, cfg.queue_depth, /*direct_io=*/true, err, cfg.io_backend)) return false;
    }
    if (cold_ && !io_cold_.init(cold_->shard_paths(), cfg.queue_depth, true, err, cfg.io_backend)) {
        fprintf(stderr, "[qwfn] cold tier disabled: %s\n", err.c_str());
        cold_ = nullptr;
        err.clear();
    }
    return true;
}

// Work out one layer's slot layout, and check the assumption the O_DIRECT
// path relies on: an expert slice is a whole number of 512-byte blocks, so
// the read-around padding is constant per (layer, part) rather than
// per-expert.
bool expert_cache::layout_layer(uint32_t il, std::string & err) {
    assert_that(hot_ != nullptr, "hot index set before the layout");
    assert_that(il < blk_.size(), "layer within the pool table");
    const model_index * hot = hot_;
    layer_pool & lp = blk_[il];
    uint32_t off = 0;
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        const byte_range r0 = hot->expert_range(il, 0, (expert_part) q);
        const byte_range r1 = hot->expert_range(il, 1, (expert_part) q);
        if (!r0.valid()) { err = "missing expert tensor on layer " + std::to_string(il); return false; }
        if (r1.valid() && ((r1.offset - r0.offset) % dio_align()) != 0) {
            err = "expert slice stride is not " + std::to_string(dio_align()) + "-byte aligned on layer " + std::to_string(il) +
                  "; the direct I/O layout cannot be used";
            return false;
        }
        const tensor_ref * t = hot->find("blk." + std::to_string(il) + "." +
            (q == EXPERT_GATE ? "ffn_gate_exps.weight" : q == EXPERT_UP ? "ffn_up_exps.weight" : "ffn_down_exps.weight"));
        lp.part_type[q]  = t ? t->type : GGML_TYPE_F32;
        lp.part_off[q]   = off;
        lp.part_pay[q]   = io_hot_.payload_offset(r0.offset);
        lp.part_bytes[q] = r0.nbytes;
        if (cold_) {
            const byte_range cr = cold_->expert_range(il, 0, (expert_part) q);
            const tensor_ref * ct = cold_->find("blk." + std::to_string(il) + "." +
                (q == EXPERT_GATE ? "ffn_gate_exps.weight" : q == EXPERT_UP ? "ffn_up_exps.weight" : "ffn_down_exps.weight"));
            lp.cold_type[q] = ct ? ct->type : lp.part_type[q];
            lp.cold_pay[q]  = io_cold_.payload_offset(cr.offset);
        }
        // A slot holds a block from either file: size it for the larger.
        size_t part_slot = dio_padded_size(r0.offset, r0.nbytes);
        if (cold_) part_slot = std::max(part_slot, (size_t) dio_padded_size(cold_->expert_range(il, 0, (expert_part) q).offset,
                                                                             cold_->expert_range(il, 0, (expert_part) q).nbytes));
        off += part_slot;
    }
    lp.block_bytes = off;
    return true;
}

bool expert_cache::size_ram_tier(std::string & err) {
    assert_that(hot_ != nullptr, "hot index set before sizing the RAM tier");
    assert_that(arena_ == nullptr, "the RAM tier is sized before its arena exists");
    const uint32_t n_layer = (uint32_t) blk_.size();
    // Never take more RAM than the kernel says it can spare. On a 30 GB box
    // sitting behind 30 GB of zram, over-committing here does not fail the
    // allocation -- it OOMs the machine.
    const size_t ram_budget = clamp_to_available(cfg_.ram_bytes, cfg_.ram_frac, cfg_.ram_headroom);
    if (ram_budget < (1ull << 30)) {
        err = "less than 1 GB can be spared for the expert RAM tier (MemAvailable = " +
              std::to_string(mem_available_bytes() >> 20) + " MiB); free memory or lower --ram";
        return false;
    }

    // Equal byte share per layer: traffic per layer is identical (n_expert_used
    // activations each), so bytes -- not slot count -- is the fair unit.
    const size_t per_layer_bytes = ram_budget / n_layer;
    arena_bytes_ = 0;
    for (uint32_t il = 0; il < n_layer; il++) {
        layer_pool & lp = blk_[il];
        lp.n_slots = (uint32_t) std::max<size_t>(1, per_layer_bytes / lp.block_bytes);
        lp.n_slots = std::min<uint32_t>(lp.n_slots, hot_->hp().n_expert);
        arena_bytes_ += (size_t) lp.n_slots * lp.block_bytes;
    }

    // Rounding slots up per layer can push the total past the budget; re-check.
    if (arena_bytes_ > ram_budget) {
        const double shrink = (double) ram_budget / (double) arena_bytes_;
        arena_bytes_ = 0;
        for (uint32_t il = 0; il < n_layer; il++) {
            layer_pool & lp = blk_[il];
            lp.n_slots = (uint32_t) std::max<size_t>(1, (size_t) (lp.n_slots * shrink));
            arena_bytes_ += (size_t) lp.n_slots * lp.block_bytes;
        }
    }
    return true;
}

// Pinned when the device offers a host buffer type. The CUDA host type
// silently falls back to an ordinary CPU buffer if pinning fails, and that
// one is neither pinned nor page-aligned, so check both before trusting it.
bool expert_cache::alloc_arena(std::string & err) {
    assert_that(arena_ == nullptr, "the arena is allocated once per init");
    assert_that(arena_hostbuf_ == nullptr, "no pinned arena left from a previous init");
    arena_pinned_ = false;
    if (cfg_.host_buft) {
        arena_hostbuf_ = ggml_backend_buft_alloc_buffer(cfg_.host_buft, arena_bytes_);
        uint8_t * p = arena_hostbuf_ ? (uint8_t *) ggml_backend_buffer_get_base(arena_hostbuf_) : nullptr;
        if (arena_hostbuf_ && p && ggml_backend_buffer_get_type(arena_hostbuf_) == cfg_.host_buft &&
            ((uintptr_t) p % 4096) == 0) {
            arena_ = p;
            arena_pinned_ = true;
        } else {
            if (arena_hostbuf_) ggml_backend_buffer_free(arena_hostbuf_);
            arena_hostbuf_ = nullptr;
            fprintf(stderr, "[qwfn] pinned expert arena unavailable; using pageable memory\n");
        }
    }
    if (!arena_) arena_ = (uint8_t *) dio_alloc(arena_bytes_);
    if (!arena_) { err = "failed to allocate " + std::to_string(arena_bytes_ >> 20) + " MiB expert arena"; return false; }
    return true;
}

void expert_cache::init_ram_slots() {
    assert_that(arena_ != nullptr, "arena allocated before its slots are laid out");
    assert_that(hot_ != nullptr, "hot index set before the slot tables");
    const uint32_t n_layer = (uint32_t) blk_.size();
    size_t cursor = 0;
    total_slots_ = 0;
    for (uint32_t il = 0; il < n_layer; il++) {
        layer_pool & lp = blk_[il];
        lp.base = arena_ + cursor;
        cursor += (size_t) lp.n_slots * lp.block_bytes;
        lp.slot_expert.assign(lp.n_slots, SLOT_EMPTY);
        lp.slot_valid.assign(lp.n_slots, 0);
        lp.slot_cold.assign(lp.n_slots, 0);
        lp.slot_speculative.assign(lp.n_slots, 0);
        lp.slot_pinned.assign(lp.n_slots, 0);
        lp.slot_pf_fetch.assign(lp.n_slots, 0);
        lp.slot_freq.assign(lp.n_slots, 0);
        lp.ef.assign(hot_->hp().n_expert, 0);
        lp.slot_used.assign(lp.n_slots, 0);
        lp.expert_slot.assign(hot_->hp().n_expert, -1);
        lp.seen.assign(hot_->hp().n_expert, 0);
        lp.hotw.assign(hot_->hp().n_expert, 0);
        total_slots_ += lp.n_slots;
    }
    assert_that(cursor == arena_bytes_, "the layers tile the arena exactly");
}

void expert_cache::init_vram_tier() {
    assert_that(cfg_.vram_bytes > 0 && cfg_.vram_buft != nullptr, "VRAM tier requested");
    assert_that(vram_buf_ == nullptr && vram_extra_ == nullptr, "VRAM tier allocated once per init");
    vram_plan p;
    vram_plan_sizes(p);
    if (!vram_fit(p)) {
        fprintf(stderr, "[qwfn] VRAM tier disabled: no device memory available\n");
        return;
    }
    vram_split(p);
    if (!vram_buf_) {
        fprintf(stderr, "[qwfn] VRAM tier disabled: no device memory available\n");
        return;
    }
    vram_assign(p);
}

void expert_cache::vram_plan_sizes(vram_plan & p) {
    assert_that(hot_ != nullptr, "hot index set before sizing the VRAM tier");
    assert_that(p.gslots.empty() && p.want == 0, "a fresh plan");
    const uint32_t n_layer = (uint32_t) blk_.size();
    p.gslots.assign(n_layer, 0);
    p.nat.assign(n_layer, 0);
    for (uint32_t il = 0; il < n_layer; il++) {
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            blk_[il].g_part_bytes[q] = hot_->expert_range(il, 0, (expert_part) q).nbytes;
            p.nat[il] += blk_[il].g_part_bytes[q];
        }
    }
    const size_t per_layer_g = cfg_.vram_bytes / n_layer;
    for (uint32_t il = 0; il < n_layer; il++) {
        p.gslots[il] = (uint32_t) std::min<size_t>(hot_->hp().n_expert,
                          std::max<size_t>(1, per_layer_g / p.nat[il]));
        p.want += (size_t) p.gslots[il] * p.nat[il] + TIER_PAD;
    }
    // Tail slack so a kernel that over-reads past the last row of the last
    // slot (CUDA's MMQ does, by MATRIX_ROW_PADDING) stays inside the buffer.
    p.want += 1ull << 20;
    // The dynamic part: the last layers' arrays, freed while a prefill
    // runs so its staging, batch buffers and arenas can have that memory.
    // The back-off keeps at least that much: a tier is an optimisation,
    // the prefill is not.
    p.lend = cfg_.lend_bytes;
    if (p.lend && p.want < p.lend + (1ull << 20)) p.want = p.lend + (1ull << 20);
}

// Back off rather than fail: the tier is an optimisation, and asking
// for more than the device has left should cost throughput, not the run.
// Step down in 4% increments, not quarters: coarse steps threw away up
// to a quarter of the device memory that was actually free, and 8% steps
// still turned a 0.4 GB draft head into a 0.73 GB loss of tier; every
// 2.18 MB block that fits is an expert that computes 3.2x faster.
// Probe once with the reservation included, then release it, so the
// back-off converges on a size that still leaves room for the graphs.
bool expert_cache::vram_fit(vram_plan & p) {
    assert_that(cfg_.vram_buft != nullptr, "device buffer type set");
    assert_that(p.gslots.size() == blk_.size() && p.nat.size() == blk_.size(), "plan covers every layer");
    const uint32_t n_layer = (uint32_t) blk_.size();
    if (ggml_backend_buffer_t probe = ggml_backend_buft_alloc_buffer(cfg_.vram_buft, cfg_.vram_reserve)) {
        ggml_backend_buffer_free(probe);
    }
    ggml_backend_buffer_t fit = nullptr;
    // Bounded: 0.96^200 shrinks any tier below the floor, and a size that stops shrinking
    // (every layer at its 1-slot minimum) ends the search instead of retrying it forever.
    size_t prev_want = 0;
    for (int tries = 0; tries < 200 && p.want != prev_want &&
                        p.want > std::max<size_t>(256ull << 20, p.lend + (1ull << 20)) &&
                        !(fit = ggml_backend_buft_alloc_buffer(cfg_.vram_buft, p.want + cfg_.vram_reserve)); tries++) {
        prev_want = p.want;
        p.want = 1ull << 20;
        for (uint32_t il = 0; il < n_layer; il++) {
            p.gslots[il] = (uint32_t) std::max<size_t>(1, (size_t) (p.gslots[il] * 0.96));
            p.want += (size_t) p.gslots[il] * p.nat[il] + TIER_PAD;
        }
        if (p.lend && p.want < p.lend + (1ull << 20)) p.want = p.lend + (1ull << 20);
    }
    const bool ok = fit != nullptr;
    if (fit) ggml_backend_buffer_free(fit);
    return ok;
}

// Layout: layers in order; the first layer whose arrays would end
// past (want - lend) starts the dynamic buffer.
void expert_cache::vram_split(vram_plan & p) {
    assert_that(vram_buf_ == nullptr && vram_extra_ == nullptr, "device buffers not yet allocated");
    assert_that(p.lend == 0 || p.want > p.lend, "the tier holds at least the lent part");
    const uint32_t n_layer = (uint32_t) blk_.size();
    std::vector<size_t> lbytes(n_layer);
    for (uint32_t il = 0; il < n_layer; il++) lbytes[il] = (size_t) p.gslots[il] * p.nat[il] + TIER_PAD;
    p.K = n_layer;
    if (p.lend) {
        size_t acc = 0;
        for (uint32_t il = 0; il < n_layer; il++) {
            if (acc + lbytes[il] > p.want - p.lend) { p.K = il; break; }
            acc += lbytes[il];
        }
    }
    p.perm_bytes = 1ull << 20; p.ext_bytes = 0;
    for (uint32_t il = 0; il < p.K; il++)       p.perm_bytes += lbytes[il];
    for (uint32_t il = p.K; il < n_layer; il++) p.ext_bytes  += lbytes[il];
    if (p.ext_bytes) p.ext_bytes += 1ull << 20;
    vram_buf_ = ggml_backend_buft_alloc_buffer(cfg_.vram_buft, p.perm_bytes);
    if (vram_buf_ && p.ext_bytes) {
        vram_extra_ = ggml_backend_buft_alloc_buffer(cfg_.vram_buft, p.ext_bytes);
        if (!vram_extra_) {   // no split possible: everything permanent, nothing dynamic
            ggml_backend_buffer_free(vram_buf_);
            vram_buf_ = ggml_backend_buft_alloc_buffer(cfg_.vram_buft, p.perm_bytes + p.ext_bytes);
            p.K = n_layer; p.ext_bytes = 0;
        }
    }
}

void expert_cache::vram_assign(const vram_plan & p) {
    assert_that(vram_buf_ != nullptr, "permanent device buffer allocated");
    assert_that(p.K <= blk_.size() && (p.ext_bytes == 0) == (vram_extra_ == nullptr), "split matches the buffers");
    const uint32_t n_layer = (uint32_t) blk_.size();
    extra_bytes_ = p.ext_bytes;
    // Every slot must decode to something finite from the start: the
    // mul_mat_id MoE points experts that live elsewhere at slot 0 with
    // weight 0, and 0 * NaN would poison the sum. Zero bytes decode to
    // zero for every quant type in this model.
    ggml_backend_buffer_clear(vram_buf_, 0);
    if (vram_extra_) ggml_backend_buffer_clear(vram_extra_, 0);
    size_t goff_p = 0, goff_e = 0;
    for (uint32_t il = 0; il < n_layer; il++) {
        layer_pool & lp = blk_[il];
        lp.g_slots   = p.gslots[il];
        lp.g_in_lent = il >= p.K;
        lp.g_buf     = lp.g_in_lent ? vram_extra_ : vram_buf_;
        size_t & goff = lp.g_in_lent ? goff_e : goff_p;
        lp.g_off = goff;
        uint8_t * gbase = (uint8_t *) ggml_backend_buffer_get_base(lp.g_buf);
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            lp.g_part[q] = gbase + goff;
            goff += (size_t) lp.g_slots * lp.g_part_bytes[q];
        }
        goff += TIER_PAD;
        lp.g_slot_expert.assign(lp.g_slots, SLOT_EMPTY);
        lp.g_valid.assign(lp.g_slots, 0);
        lp.g_cold.assign(lp.g_slots, 0);
        lp.g_freq.assign(lp.g_slots, 0);
        lp.g_used.assign(lp.g_slots, 0);
        lp.g_used_fetch.assign(lp.g_slots, 0);
        lp.g_expert_slot.assign(hot_->hp().n_expert, -1);
        total_gslots_ += lp.g_slots;
    }
    // One scratch tensor, repointed per copy: ggml_backend_tensor_set
    // dispatches on tensor->buffer, so this is how host bytes reach VRAM.
    ggml_init_params xp{}; xp.mem_size = ggml_tensor_overhead() * 4; xp.no_alloc = true;
    xfer_ctx_ = ggml_init(xp); assert_that(xfer_ctx_ != nullptr, "ggml_init: xfer_ctx_");
    xfer_ = ggml_new_tensor_1d(xfer_ctx_, GGML_TYPE_I8, 1);
    xfer_->buffer = vram_buf_;
    fprintf(stderr, "[qwfn] expert VRAM tier: %.2f GB, %zu blocks (%.1f%%)%s\n",
            (p.perm_bytes + p.ext_bytes) / 1e9, total_gslots_,
            100.0 * (double) total_gslots_ / (double) (n_layer * hot_->hp().n_expert),
            p.ext_bytes ? (", of which " + std::to_string(n_layer - p.K) + " layers (" +
                           std::to_string(p.ext_bytes >> 20) + " MiB) are handed to the prefill").c_str() : "");
}

void expert_cache::shutdown() {
    settle_promotions();   // copies still reading the arena must land before it goes
    if (xfer_ctx_) { ggml_free(xfer_ctx_); xfer_ctx_ = nullptr; xfer_ = nullptr; }
    if (vram_buf_) { ggml_backend_buffer_free(vram_buf_); vram_buf_ = nullptr; }
    if (vram_extra_) { ggml_backend_buffer_free(vram_extra_); vram_extra_ = nullptr; }
    extra_bytes_ = 0;
    total_gslots_ = 0;
    if (arena_buf_) { ggml_backend_buffer_free(arena_buf_); arena_buf_ = nullptr; }
    if (arena_hostbuf_) { ggml_backend_buffer_free(arena_hostbuf_); arena_hostbuf_ = nullptr; arena_ = nullptr; }
    else if (arena_) { dio_free(arena_); arena_ = nullptr; }
    arena_pinned_ = false;
    blk_.clear();
    io_hot_.shutdown();
    io_pf_.shutdown();
    io_cold_.shutdown();
    pf_pending_.clear();
    pf_reads_outstanding_ = 0;
    total_slots_ = 0;
    arena_bytes_ = 0;
    assert_that(pending_release_.empty(), "promotions settled before the tiers go");
    assert_that(arena_ == nullptr && vram_buf_ == nullptr && xfer_ == nullptr, "every tier released");
}

int32_t expert_cache::find_slot(layer_pool & lp, uint32_t expert_id) const {
    assert_that(expert_id < lp.expert_slot.size(), "expert id within the layer");
    const int32_t s = lp.expert_slot[expert_id];
    if (s < 0) return -1;
    assert_that((uint32_t) s < lp.n_slots, "expert maps to a slot of its layer");
    if (lp.slot_expert[s] != (uint16_t) expert_id || !lp.slot_valid[s]) return -1;
    return s;
}

int32_t expert_cache::choose_victim(layer_pool & lp) {
    // Prefer an empty slot, else the policy's coldest of evict_samples sampled
    // slots. Sampling keeps eviction O(1) while still tracking the heavy skew in
    // routing. Held and guarded slots are skipped and an unguarded one still
    // being read is waited for (slot_wait_busy): reusing one would point two
    // experts at one block. When those cover the layer, oldest_spec_victim.
    assert_that(lp.n_slots > 0, "every layer has at least one RAM slot");
    assert_that(lp.slot_pinned.size() == lp.n_slots && lp.slot_used.size() == lp.n_slots, "slot tables sized");
    static thread_local uint64_t rng = 0x243F6A8885A308D3ull;
    int32_t best = -1;
    uint64_t best_score = UINT64_MAX;
    for (uint32_t k = 0; k < cfg_.evict_samples; k++) {
        const uint32_t s = (uint32_t) (xorshift(rng) % lp.n_slots);
        if (lp.slot_pinned[s] || slot_wait_busy(lp, s, true)) continue;
        if (lp.slot_expert[s] == SLOT_EMPTY) return (int32_t) s;
        // lru    : evict the slot untouched longest.
        // lfu    : evict the expert with the fewest lifetime uses (survives eviction).
        // hybrid : frequency, tie-broken by recency, with recency dominating
        //          once counts are close -- a cheap stand-in for W-TinyLFU.
        uint64_t score;
        switch (cfg_.policy) {
            case config::evict_policy::lfu: score = lp.ef[lp.slot_expert[s]]; break;
            case config::evict_policy::hybrid:
                score = (uint64_t) lp.ef[lp.slot_expert[s]] * 4 + (lp.slot_used[s] >> 4); break;
            default: score = lp.slot_used[s]; break;
        }
        if (score < best_score) { best_score = score; best = (int32_t) s; }
    }
    if (best < 0) {
        for (uint32_t s = 0; s < lp.n_slots; s++)
            if (!lp.slot_pinned[s] && !slot_wait_busy(lp, s, true)) { best = (int32_t) s; break; }
    }
    if (best < 0) best = oldest_spec_victim(lp);
    return best;
}

// Handed to the caller during this same fetch (a demand read of it may be in
// flight), or read by an asynchronous promotion: evicting it now would repoint
// an expert at another expert's weights.
bool expert_cache::slot_held(const layer_pool & lp, uint32_t s) const {
    assert_that(s < lp.n_slots, "slot within the layer");
    assert_that(pending_release_.size() <= total_slots_, "pending releases bounded by the tier");
    for (int32_t l : live_) if (l == (int32_t) s) return true;
    for (const pending_rel & r : pending_release_)
        if (r.slot == (int32_t) s && &blk_[r.layer] == &lp) return true;
    return false;
}

// Whether slot s may not be evicted; BLOCKS while s's own speculative read is
// in flight. Every test is a function of the token sequence, never of I/O
// timing: the victim decides what the RAM tier holds, and a RAM-resident block
// can be promoted where a miss cannot. So a speculative claim is protected for
// pf_guard() fetches (with `guard`) landed or not, and an unguarded one still
// in flight is waited for rather than skipped.
bool expert_cache::slot_wait_busy(layer_pool & lp, uint32_t s, bool guard) {
    assert_that(s < lp.n_slots, "slot within the layer");
    assert_that(lp.slot_pf_fetch.size() == lp.n_slots, "speculative claim table sized");
    if (slot_held(lp, s)) return true;
    if (lp.slot_expert[s] == SLOT_EMPTY) return false;
    if (guard && pf_guarded(lp, s)) return true;
    // Every reap retires at least one read, so this ends with the outstanding count.
    while (!lp.slot_valid[s] && pf_reads_outstanding_ > 0 && pf_slot_pending(lp, s)) pf_reap(1);
    return lp.slot_expert[s] != SLOT_EMPTY && !lp.slot_valid[s];
}

bool expert_cache::pf_guarded(const layer_pool & lp, uint32_t s) const {
    assert_that(s < lp.slot_pf_fetch.size(), "slot within the layer");
    assert_that(lp.slot_pf_fetch[s] <= fetch_count_ + 1, "a claim is not from the future");
    return lp.slot_pf_fetch[s] != 0 && fetch_count_ + 1 - lp.slot_pf_fetch[s] < pf_guard();
}

// Whether a speculative read into slot s of lp has not landed yet.
bool expert_cache::pf_slot_pending(const layer_pool & lp, uint32_t s) const {
    assert_that(s < lp.n_slots, "slot within the layer");
    assert_that(pf_reads_outstanding_ > 0, "asked only while reads are outstanding");
    for (const pf_entry & e : pf_pending_)
        if (e.remaining > 0 && e.slot == (int32_t) s && &blk_[e.layer] == &lp) return true;
    return false;
}

// The last resort when held and guarded slots cover the whole layer (a small
// RAM tier): the oldest speculative claim, once its read has landed.
int32_t expert_cache::oldest_spec_victim(layer_pool & lp) {
    assert_that(lp.slot_pf_fetch.size() == lp.n_slots, "speculative claim table sized");
    assert_that(lp.slot_pinned.size() == lp.n_slots, "pin table sized");
    int32_t best = -1;
    for (uint32_t s = 0; s < lp.n_slots; s++) {
        if (lp.slot_pinned[s] || lp.slot_pf_fetch[s] == 0 || lp.slot_expert[s] == SLOT_EMPTY || slot_held(lp, s)) continue;
        if (best < 0 || lp.slot_pf_fetch[s] < lp.slot_pf_fetch[best]) best = (int32_t) s;
    }
    if (best >= 0 && slot_wait_busy(lp, (uint32_t) best, false)) best = -1;
    return best;
}

void expert_cache::fill_handle(const layer_pool & lp, uint32_t slot, expert_handle & h) const {
    assert_that(slot < lp.n_slots, "RAM slot within the layer");
    assert_that(lp.base != nullptr, "RAM tier laid out");
    uint8_t * base = lp.base + (size_t) slot * lp.block_bytes;
    const bool cold = lp.slot_cold[slot] != 0;
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        h.part[q] = base + lp.part_off[q] + (cold ? lp.cold_pay[q]  : lp.part_pay[q]);
        h.type[q] = cold ? lp.cold_type[q] : lp.part_type[q];
    }
    h.buffer    = arena_buf_;
    h.on_gpu    = false;
    h.from_cold = cold;
    h.slot      = (int32_t) slot;
}

void expert_cache::fill_gpu_handle(const layer_pool & lp, uint32_t gslot, expert_handle & h) const {
    assert_that(gslot < lp.g_slots, "VRAM slot within the layer");
    assert_that(lp.g_buf != nullptr && !lp.g_lent, "VRAM tier of the layer present");
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        h.part[q] = lp.g_part[q] + (size_t) gslot * lp.g_part_bytes[q];
        h.type[q] = lp.part_type[q];
    }
    h.buffer    = lp.g_buf;
    h.on_gpu    = true;
    h.from_cold = false;
    h.slot      = (int32_t) gslot;
    h.late      = false;
}

tier_view expert_cache::gpu_tier(uint32_t layer) const {
    tier_view v;
    if (layer >= blk_.size() || !vram_buf_) return v;
    const layer_pool & lp = blk_[layer];
    if (lp.g_slots == 0 || lp.g_lent) return v;
    assert_that(lp.g_buf != nullptr, "a present VRAM tier has its buffer");
    assert_that(lp.g_part[0] != nullptr, "a present VRAM tier has its arrays");
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        v.part[q]   = lp.g_part[q];
        v.stride[q] = lp.g_part_bytes[q];
        v.type[q]   = lp.part_type[q];
    }
    v.n_slots = lp.g_slots;
    v.buffer  = lp.g_buf;
    return v;
}

tier_view expert_cache::ram_tier(uint32_t layer) const {
    tier_view v;
    if (layer >= blk_.size() || !arena_) return v;
    const layer_pool & lp = blk_[layer];
    assert_that(lp.base != nullptr, "an allocated arena is laid out");
    assert_that(lp.n_slots > 0, "every layer has at least one RAM slot");
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        v.part[q]   = lp.base + lp.part_off[q] + lp.part_pay[q];
        v.stride[q] = lp.block_bytes;
        v.type[q]   = lp.part_type[q];
    }
    v.n_slots = lp.n_slots;
    v.buffer  = arena_buf_;
    return v;
}

tier_view expert_cache::ram_tier_cold(uint32_t layer) const {
    tier_view v;
    if (layer >= blk_.size() || !arena_ || !cold_) return v;
    const layer_pool & lp = blk_[layer];
    assert_that(lp.base != nullptr, "an allocated arena is laid out");
    assert_that(lp.n_slots > 0, "every layer has at least one RAM slot");
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        v.part[q]   = lp.base + lp.part_off[q] + lp.cold_pay[q];
        v.stride[q] = lp.block_bytes;
        v.type[q]   = lp.cold_type[q];
    }
    v.n_slots = lp.n_slots;
    v.buffer  = arena_buf_;
    return v;
}

expert_cache::census expert_cache::ram_census() const {
    census c;
    for (const layer_pool & lp : blk_) {
        assert_that(lp.slot_expert.size() == lp.n_slots, "slot table sized to the layer");
        c.slots += lp.n_slots;
        for (uint32_t s = 0; s < lp.n_slots; s++) {
            const auto e = lp.slot_expert[s];
            if (e == SLOT_EMPTY) { c.empty++; continue; }
            if (!lp.slot_valid[s]) c.inflight++;
            if (lp.slot_cold[s]) { c.cold++; if (e < lp.hotw.size() && lp.hotw[e]) c.cold_hotw++; } else c.hot++;
            if (lp.slot_speculative[s]) c.speculative++;
        }
        for (auto h : lp.hotw) c.hotw_marked += h ? 1 : 0;
    }
    assert_that(c.empty + c.hot + c.cold == c.slots, "every slot is empty, hot or cold");
    return c;
}

bool expert_cache::would_promote(layer_pool & lp, uint32_t expert_id) {
    if (!vram_buf_ || lp.g_slots == 0 || lp.g_lent) return false;
    assert_that(expert_id < lp.ef.size(), "expert id within the layer");
    assert_that(lp.g_slot_expert.size() == lp.g_slots && lp.g_used.size() == lp.g_slots, "VRAM slot tables sized");
    constexpr uint64_t STALE_FETCHES = 48 * 24;
    static thread_local uint64_t rng = 0x2545F4914F6CDD1Dull;
    uint32_t worst = UINT32_MAX; bool any = false;
    for (uint32_t k = 0; k < 16; k++) {
        const uint32_t s = (uint32_t) (xorshift(rng) % lp.g_slots);
        if (lp.g_slot_expert[s] == SLOT_EMPTY) return true;
        if (lp.g_used[s] >= fetch_epoch_) continue;
        if (fetch_count_ - lp.g_used_fetch[s] > STALE_FETCHES) return true;
        worst = std::min(worst, lp.ef[lp.g_slot_expert[s]]); any = true;
    }
    return any && lp.ef[expert_id] > worst;
}

// Victim: of 16 sampled slots, the lowest lifetime frequency, and only if
// the incoming expert is more frequent -- otherwise the tier thrashes.
// Measured against recency: recency evicts 96 blocks a
// token and the hot set cycles out to disk (hit 96.1% -> 94.4%, 40% more
// reads, 20 -> 16 tok/s). What was wrong with frequency was only that a
// long prefill's counts froze the tier on the prompt's experts; that is
// handled by capping the warm-up's bump and halving every counter when a
// prefill ends (lend_end), not by changing the rule.
// Hybrid, measured both ways on their own: pure recency churns the hot
// set out to disk at short context (96.1% -> 94.4% hit), pure frequency
// keeps a long prompt's experts in the tier after it (VRAM-served 59-65%
// against 80%). So: a resident unused for STALE_FETCHES fetches (24
// decoded tokens) is replaceable regardless of its count, the oldest of
// them first; otherwise the least frequent, and only by a more frequent
// newcomer. Returns the victim (-1 for none); `stale` and `worst` say how it was chosen.
int32_t expert_cache::gpu_victim(layer_pool & lp, int32_t & stale, uint32_t & worst) {
    assert_that(lp.g_slots > 0 && lp.g_slot_expert.size() == lp.g_slots, "VRAM slot tables sized");
    assert_that(stale == -1 && worst == UINT32_MAX, "caller starts from no victim");
    constexpr uint64_t STALE_FETCHES = 48 * 24;
    static thread_local uint64_t rng = 0x853C49E6748FEA9Bull;
    int32_t  victim = -1;
    uint64_t stale_oldest = UINT64_MAX;
    for (uint32_t k = 0; k < 16; k++) {
        const uint32_t s = (uint32_t) (xorshift(rng) % lp.g_slots);
        if (lp.g_slot_expert[s] == SLOT_EMPTY) { victim = (int32_t) s; stale = -1; break; }
        if (lp.g_used[s] >= fetch_epoch_) continue;              // this token's
        if (fetch_count_ - lp.g_used_fetch[s] > STALE_FETCHES && lp.g_used_fetch[s] < stale_oldest) {
            stale_oldest = lp.g_used_fetch[s]; stale = (int32_t) s;
        }
        const uint32_t f = lp.ef[lp.g_slot_expert[s]];
        if (f < worst) { worst = f; victim = (int32_t) s; }
    }
    if (stale >= 0) victim = stale;
    return victim;
}

// Asynchronous when the arena is pinned. The earlier attempt at this
// measured no gain because the arena was pageable: cudaMemcpyAsync from
// pageable memory is staged through the driver and blocks the caller
// regardless. From pinned memory it is a DMA the caller does not wait for.
// The lifetime hazard that attempt had -- the RAM slot being released and
// refilled by a disk read while the copy still read it -- is closed by
// deferring the release to settle_promotions(), after a stream sync.
void expert_cache::copy_to_gpu(layer_pool & lp, int32_t victim, const uint8_t * host_block) {
    assert_that(xfer_ != nullptr && lp.g_buf != nullptr, "transfer tensor and target buffer exist");
    assert_that(victim >= 0 && (uint32_t) victim < lp.g_slots && host_block != nullptr, "copy into a slot of the layer");
    last_promote_async_ = cfg_.vram_backend && arena_pinned_;
    xfer_->buffer = lp.g_buf;
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        xfer_->data  = lp.g_part[q] + (size_t) victim * lp.g_part_bytes[q];
        xfer_->ne[0] = lp.g_part_bytes[q];
        xfer_->nb[1] = xfer_->nb[2] = xfer_->nb[3] = lp.g_part_bytes[q];
        const uint8_t * src = host_block + lp.part_off[q] + lp.part_pay[q];
        if (last_promote_async_) ggml_backend_tensor_set_async(cfg_.vram_backend, xfer_, src, 0, lp.g_part_bytes[q]);
        else                     ggml_backend_tensor_set(xfer_, src, 0, lp.g_part_bytes[q]);
    }
}

bool expert_cache::promote(layer_pool & lp, uint32_t expert_id, const uint8_t * host_block) {
    if (!vram_buf_ || lp.g_slots == 0 || lp.g_lent) return false;
    assert_that(expert_id < lp.g_expert_slot.size() && expert_id < lp.ef.size(), "expert id within the layer");
    assert_that(lp.g_slot_expert.size() == lp.g_slots, "VRAM slot table sized");
    // A block from the cold checkpoint has other quant types and sizes; the
    // per-part device arrays hold one type per part, so it stays on the CPU.
    {
        const int32_t rs = lp.expert_slot[expert_id];
        if (rs >= 0 && lp.slot_cold[rs]) return false;
    }

    int32_t  stale = -1;
    uint32_t worst = UINT32_MAX;
    const int32_t victim = gpu_victim(lp, stale, worst);
    if (victim < 0) return false;
    const uint32_t incoming = lp.ef[expert_id];
    if (stale < 0 && lp.g_slot_expert[victim] != SLOT_EMPTY && worst >= incoming) return false;

    if (lp.g_slot_expert[victim] != SLOT_EMPTY) lp.g_expert_slot[lp.g_slot_expert[victim]] = -1;

    if (host_block) copy_to_gpu(lp, victim, host_block);   // else the caller copies it once its read lands

    lp.g_slot_expert[victim]  = (uint16_t) expert_id;
    lp.g_valid[victim]        = host_block ? 1 : 0;   // a reservation reads as absent until its copy
    lp.g_cold[victim]         = 0;
    lp.g_freq[victim]         = incoming;   // kept for stats only
    lp.g_used[victim]         = tick_;
    lp.g_used_fetch[victim]   = fetch_count_;
    lp.g_expert_slot[expert_id] = victim;
    lp.g_ver++;
    st_.promotions++;
    return true;
}

bool expert_cache::fetch(uint32_t layer, const uint32_t * expert_ids, uint32_t n, expert_handle * out) {
    assert_that(hot_ != nullptr, "fetch after init");
    assert_that(blk_.size() == hot_->hp().n_layer, "one pool per layer");
    std::vector<char> ready(n);
    if (!fetch_begin(layer, expert_ids, n, out, (bool *) ready.data())) return false;
    return fetch_end();
}

bool expert_cache::fetch_begin(uint32_t layer, const uint32_t * expert_ids, uint32_t n,
                               expert_handle * out, bool * ready) {
    // The miss bookkeeping below is sized for at most 64 experts per call (a verify step
    // asks for n_expert_used x its positions, 40 at most).
    if (layer >= blk_.size() || n > 64 || (n > 0 && (!expert_ids || !out || !ready))) return false;
    for (uint32_t i = 0; i < n; i++) if (expert_ids[i] >= blk_[layer].ef.size()) return false;
    const auto t_enter = std::chrono::steady_clock::now();
    // A speculative read for one of these experts that has not landed yet is
    // reported as not ready and settled in fetch_end(), after the demand reads
    // have been submitted, so the caller can compute the experts it already has
    // while both kinds of read land. The only waits here are for one slot's own
    // speculative read: a victim's (slot_wait_busy) or an upgraded block's
    // (fetch_one). In-flight slots are never chosen as victims.
    layer_pool & lp = blk_[layer];
    assert_that(lp.expert_slot.size() == lp.ef.size(), "per-expert tables agree");
    assert_that(lp.slot_expert.size() == lp.n_slots, "slot table sized to the layer");
    tick_++;
    fetch_epoch_ = tick_;
    fetch_count_++;
    // A fetch whose end was never called: its reads and reserved promotions first.
    if (has_inflight() && !fetch_end()) return false;
    live_.clear();
    pending_.clear();
    // Set before the lookups: speculative reads reaped during them (a victim's,
    // a cold upgrade's) must see this fetch's pending blocks (pf_finish).
    inflight_layer_ = layer;
    inflight_reqs_  = 0;
    inflight_slots_.clear();
    inflight_failed_ = false;
    inflight_experts_.clear();
    touch_vram_residents(lp, expert_ids, n);
    protect_ram_residents(lp, expert_ids, n);

    fetch_ctx c;
    const double promote_base = st_.t_promote;
    bool ok = true;
    for (uint32_t i = 0; i < n && ok; i++) ok = fetch_one(layer, lp, expert_ids[i], out[i], ready[i], c);
    if (ok && c.n_req == 0) {
        st_.t_submit += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_enter).count() - (st_.t_promote - promote_base);
        return true;
    }
    if (ok && submit_burst(c, t_enter, promote_base)) return true;
    abort_fetch(lp, c);
    return false;
}

// A fetch that failed before all its reads were submitted (a lookup found no
// slot or byte range, or the submit stalled): land what was submitted, then
// release what the lookups reserved -- the VRAM slots of deferred promotions,
// the RAM slots claimed for demand reads, the pending blocks that never landed
// -- so a later lookup misses cleanly instead of finding an invalid slot that
// no read will fill and no eviction will take. The demand slots are emptied
// only when both engines are idle: a short reap does not prove a read has left
// the kernel, and a slot freed under a live read would be refilled and then
// overwritten. Otherwise they stay claimed and invalid, which leaks them but
// cannot corrupt (slot_wait_busy never evicts them).
void expert_cache::abort_fetch(layer_pool & lp, const fetch_ctx & c) {
    assert_that(&lp == &blk_[inflight_layer_], "the pool of the fetched layer");
    assert_that(c.n_miss <= 64 && inflight_slots_.empty(), "a failed burst registered no in-flight slot");
    if (!drain_reads(io_hot_, inflight_reqs_) || !drain_reads(io_cold_, inflight_cold_reqs_))
        fprintf(stderr, "[qwfn] expert read wait returned short after a failed fetch\n");
    inflight_reqs_ = inflight_cold_reqs_ = 0;
    inflight_failed_ = false;
    for (size_t k = 0; k < c.n_miss; k++) {   // registered so drop_unlanded_pending leaves them alone
        inflight_slots_.push_back((int32_t) c.miss_slot[k]);
        inflight_experts_.push_back(c.miss_expert[k]);
    }
    if (demand_idle())
        for (size_t k = 0; k < c.n_miss; k++) empty_slot(lp, (int32_t) c.miss_slot[k], c.miss_expert[k]);
    finish_deferred(lp, false);   // no reserved VRAM slot may stay behind a failed fetch
    drop_unlanded_pending(lp);
}

// Release slot s of expert e, which no read will fill: a later lookup of e misses.
void expert_cache::empty_slot(layer_pool & lp, int32_t s, uint32_t e) {
    assert_that(s >= 0 && (uint32_t) s < lp.n_slots, "RAM slot within the layer");
    assert_that(e < lp.expert_slot.size(), "expert id within the layer");
    if (lp.slot_expert[s] != (uint16_t) e) return;   // taken by another expert since
    if (lp.expert_slot[e] == s) lp.expert_slot[e] = -1;
    lp.slot_expert[s] = SLOT_EMPTY;
    lp.slot_valid[s]  = 0;
}

// Pending blocks that never landed (a failed speculative read pf_finish kept
// for the settle, or a cold one that cannot be re-read): once pending_ is
// cleared nothing would read or evict them, so they are emptied. One whose
// read is still in flight is left to pf_finish. So is one that is this fetch's
// own demand slot (a duplicate id found it in flight): fetch_end or abort_fetch
// decides about it once the demand reads are drained.
void expert_cache::drop_unlanded_pending(layer_pool & lp) {
    assert_that(&lp == &blk_[inflight_layer_], "the pool of the fetched layer");
    assert_that(pending_.size() <= 64, "at most one pending entry per expert of the fetch");
    for (const pending_spec & p : pending_) {
        if (lp.slot_valid[p.slot]) continue;
        if (std::find(inflight_slots_.begin(), inflight_slots_.end(), p.slot) != inflight_slots_.end()) continue;
        if (pf_reads_outstanding_ > 0 && pf_slot_pending(lp, (uint32_t) p.slot)) continue;
        empty_slot(lp, p.slot, p.expert);
    }
    pending_.clear();
}

// Mark every VRAM-resident expert of this token as in use BEFORE any
// promotion runs: a promotion picks a recency victim, and an expert later
// in this same list still carried last token's tick, so it could be
// evicted moments before it was looked up -- a disk read if it had no RAM
// copy. Measured: 94.4% -> hit rate against 96.1% with the old rule.
void expert_cache::touch_vram_residents(layer_pool & lp, const uint32_t * expert_ids, uint32_t n) {
    assert_that(n == 0 || expert_ids != nullptr, "expert ids");
    assert_that(lp.g_used.size() == lp.g_slots && lp.g_used_fetch.size() == lp.g_slots, "VRAM slot tables sized");
    if (!(vram_buf_ && lp.g_slots > 0 && !lp.g_lent)) return;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t e = expert_ids[i];
        if (e >= lp.g_expert_slot.size()) continue;
        const int32_t gs = lp.g_expert_slot[e];
        if (gs >= 0 && lp.g_valid[gs] && lp.g_slot_expert[gs] == (uint16_t) e) { lp.g_used[gs] = tick_; lp.g_used_fetch[gs] = fetch_count_; }
    }
}

// The same for the RAM tier. A resident block of an expert later in this
// list still carries last token's recency and, for a tail expert, a low
// count, so a miss earlier in the list could pick it as the victim: a
// prefetched block evicted moments before it was looked up, and the miss
// that causes evicts another. Listing them as live protects them.
void expert_cache::protect_ram_residents(layer_pool & lp, const uint32_t * expert_ids, uint32_t n) {
    assert_that(n == 0 || expert_ids != nullptr, "expert ids");
    assert_that(live_.empty(), "live slots start empty for each fetch");
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t e = expert_ids[i];
        if (e >= lp.expert_slot.size()) continue;
        const int32_t s = lp.expert_slot[e];
        if (s >= 0 && lp.slot_expert[s] == (uint16_t) e) live_.push_back(s);
    }
}

// One expert of a fetch_begin() call: a hit in either tier, a speculative read
// still in flight, or a miss whose read is queued in `c`. False aborts the fetch.
bool expert_cache::fetch_one(uint32_t layer, layer_pool & lp, uint32_t e, expert_handle & h, bool & rdy,
                             fetch_ctx & c) {
    assert_that(e < lp.ef.size(), "expert id validated by fetch_begin");
    assert_that(c.n_miss < 64 && c.n_req + EXPERT_NPARTS <= fetch_ctx::MAX_REQ, "miss bookkeeping has room");
    st_.lookups++;
    lp.ef[e]++;
    if (++ef_ticks_ >= cfg_.age_every) {
        // Halve every counter so old popularity decays; ordering is
        // preserved, and it keeps an early-hot expert from pinning a slot.
        ef_ticks_ = 0;
        for (auto & lay : blk_) for (auto & f : lay.ef) f >>= 1;
    }
    if (gpu_hit(lp, e, h, rdy)) return true;
    // Held by the RAM tier, landed or still being read by a speculative read.
    // Both are handled alike -- whether a read has landed is I/O timing and must
    // not change what is computed where; it only decides when the data is used.
    const int32_t cs = lp.expert_slot[e];
    const bool held = cs >= 0 && lp.slot_expert[cs] == (uint16_t) e;

    // A block from the cold file is served as it is, unless this expert would
    // now earn a VRAM slot: then it is re-read at full precision on this fetch
    // so the promotion can follow. A cold block cannot be promoted, and the RAM
    // tier's frequency rule would keep it stuck there at CPU speed, holding a
    // slot that VRAM should hold (measured without the re-read: hit 97.7 ->
    // 89.8%, VRAM-served 64 -> 52%). The prefetch and the miss path read the
    // candidates hot by the same rule, so this re-read is left to the blocks
    // whose standing changed between the read and the reuse. (Re-reading every
    // reused cold block, when nothing was read hot yet, doubled the fill:
    // 12 -> 6 tok/s.) A block of a recent speculative claim (pf_guarded, landed
    // or not -- which is timing) is served cold this once, since the re-read
    // would go into the same slot and first have to wait for the cold read. It
    // is marked hotw, so its next read (a miss or a prefetch) is at full
    // precision; a later reuse re-reads it only if it still earns a VRAM slot.
    const bool earns = held && lp.slot_cold[cs] && would_promote(lp, e);
    if (earns) lp.hotw[e] = 1;
    const bool upgrade = earns && !pf_guarded(lp, (uint32_t) cs);
    // An old claim somehow still in flight: wait for it.
    while (upgrade && !lp.slot_valid[cs] && pf_reads_outstanding_ > 0 && pf_slot_pending(lp, (uint32_t) cs)) pf_reap(1);
    if (held && !upgrade && !lp.slot_valid[cs]) {
        inflight_hit(layer, lp, e, cs, h, rdy, c.promoted);
        return true;
    }
    const int32_t s = find_slot(lp, e);
    if (s >= 0 && !upgrade) {
        ram_hit(layer, lp, e, s, h, rdy, c.promoted);
        return true;
    }
    return claim_miss(layer, lp, e, s, upgrade && s >= 0, h, rdy, c);
}

// T0: already in VRAM, so this expert's matmul runs on the GPU.
bool expert_cache::gpu_hit(layer_pool & lp, uint32_t e, expert_handle & h, bool & rdy) {
    assert_that(e < lp.ef.size(), "expert id within the layer");
    assert_that(lp.g_slots == 0 || lp.g_expert_slot.size() == lp.ef.size(), "VRAM expert map sized");
    if (!(vram_buf_ && lp.g_slots > 0 && !lp.g_lent)) return false;
    const int32_t gs = lp.g_expert_slot[e];
    if (!(gs >= 0 && lp.g_valid[gs] && lp.g_slot_expert[gs] == (uint16_t) e)) return false;
    st_.hits++; st_.gpu_hits++;
    lp.g_freq[gs]++;
    lp.g_used[gs] = tick_;
    lp.g_used_fetch[gs] = fetch_count_;
    fill_gpu_handle(lp, (uint32_t) gs, h);
    rdy = true;
    return true;
}

// Claimed by a speculative read that has not landed: a hit, waited for in
// fetch_end. A cold block in flight counts too (it used to fall through to
// the miss path, which read the expert again into a second slot and left the
// landed block an orphan: with most prefetches cold, a third of them were
// wasted that way and the RAM tier lost the slots they held -- hit 97.7 ->
// 88%). A promotion candidate exactly as a landed block is (ram_hit).
void expert_cache::inflight_hit(uint32_t layer, layer_pool & lp, uint32_t e, int32_t s, expert_handle & h,
                                bool & rdy, uint32_t & promoted) {
    assert_that(s >= 0 && lp.slot_expert[s] == (uint16_t) e && !lp.slot_valid[s], "the slot is being read for the expert");
    assert_that(pending_.size() < 64, "at most one pending entry per expert of the fetch");
    st_.hits++;
    lp.slot_freq[s]++;
    lp.slot_used[s] = ++tick_;
    if (lp.slot_speculative[s]) { st_.pf_used++; lp.slot_speculative[s] = 0; }
    live_.push_back(s);
    fill_handle(lp, (uint32_t) s, h);
    rdy = false;
    pending_.push_back(pending_spec{ e, s });
    try_promote(layer, lp, e, s, h, promoted);
}

// A valid RAM-tier block: served from the arena, and pushed on to VRAM when the
// per-layer promotion budget allows.
void expert_cache::ram_hit(uint32_t layer, layer_pool & lp, uint32_t e, int32_t s, expert_handle & h, bool & rdy,
                           uint32_t & promoted) {
    assert_that(s >= 0 && (uint32_t) s < lp.n_slots, "RAM slot within the layer");
    assert_that(lp.slot_expert[s] == (uint16_t) e, "the slot holds the expert looked up");
    st_.hits++;
    lp.slot_freq[s]++;
    lp.slot_used[s] = ++tick_;
    if (lp.slot_speculative[s]) { st_.pf_used++; lp.slot_speculative[s] = 0; }
    live_.push_back(s);
    fill_handle(lp, (uint32_t) s, h);
    rdy = true;
    try_promote(layer, lp, e, s, h, promoted);
}

// Push a RAM-held block on to VRAM when the per-layer budget allows. Decided
// here, in lookup order, by the same rule whether the block's read has landed
// or not; only the copy waits for the data (settle_pending). So the victims,
// the budget, the counters and where the expert is computed never depend on
// I/O timing.
void expert_cache::try_promote(uint32_t layer, layer_pool & lp, uint32_t e, int32_t s, expert_handle & h,
                               uint32_t & promoted) {
    assert_that(s >= 0 && (uint32_t) s < lp.n_slots, "RAM slot within the layer");
    assert_that(lp.slot_expert[s] == (uint16_t) e, "the slot holds the expert looked up");
    if (promoted >= cfg_.max_promotions_per_layer) return;
    const auto tp = std::chrono::steady_clock::now();
    const bool landed = lp.slot_valid[s] != 0;
    if (promote(lp, e, landed ? slot_ptr(lp, (uint32_t) s) : nullptr)) {
        promoted++;
        // Hand back the VRAM copy. The block itself stays intact for this
        // token -- nothing can be admitted into the slot until this layer is
        // fetched again.
        const int32_t gs = lp.g_expert_slot[e];
        assert_that(gs >= 0, "a promoted expert has its VRAM slot");
        fill_gpu_handle(lp, (uint32_t) gs, h);
        h.late = true;
        st_.gpu_hits++;
        if (!landed) deferred_.push_back(deferred_promo{ s, gs, e });
        // The copy may still be reading the slot: keep it protected until
        // the next stream sync. It stays resident after that as a fallback
        // (see settle_promotions).
        if (cfg_.vram_backend && arena_pinned_) pending_release_.push_back(pending_rel{ layer, e, s });
    }
    st_.t_promote += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tp).count();
}

// A miss (or a cold block upgraded to full precision): claim a slot and queue its reads in `c`.
bool expert_cache::claim_miss(uint32_t layer, layer_pool & lp, uint32_t e, int32_t s, bool upgrade,
                              expert_handle & h, bool & rdy, fetch_ctx & c) {
    assert_that(!upgrade || s >= 0, "an upgrade re-reads the slot it has");
    assert_that(c.n_miss < 64 && c.n_req + EXPERT_NPARTS <= fetch_ctx::MAX_REQ, "miss bookkeeping has room");
    st_.misses++;
    if (upgrade) st_.upgrades++;

    // A miss reads the hot file when the expert would earn a VRAM slot now, by
    // the rule the prefetch applies. Without this a miss during the fill came
    // in cold, was reused while the tier still had empty slots (no re-read
    // then) and sat in RAM unpromotable for the rest of the run.
    if (cold_ && !upgrade && !lp.hotw[e] && would_promote(lp, e)) lp.hotw[e] = 1;
    const bool take_cold = cold_ != nullptr && !upgrade && !lp.hotw[e];
    lp.seen[e] = 1;
    const model_index * src = take_cold ? cold_ : hot_;
    if (take_cold) st_.cold_tier_reads++;

    const int32_t v = upgrade ? s : choose_victim(lp);
    if (v < 0) return false;
    claim_slot(lp, e, v, upgrade, take_cold);

    uint8_t * base = slot_ptr(lp, (uint32_t) v);
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        const byte_range br = src->expert_range(layer, e, (expert_part) q);
        // Claimed, never read (no read was submitted into it, so it is free now).
        // The miss, disk-byte and cold-read counts above stay: stats of a failed fetch.
        if (!br.valid()) { empty_slot(lp, v, e); return false; }
        c.req_cold[c.n_req] = take_cold;
        c.reqs[c.n_req++] = io_request{ br.shard, br.offset, br.nbytes, base + lp.part_off[q], (uint64_t) c.n_miss };
        st_.bytes_from_disk += br.nbytes;
    }
    fill_handle(lp, (uint32_t) v, h);
    c.miss_slot[c.n_miss]   = (uint32_t) v;
    c.miss_expert[c.n_miss] = e;
    rdy = false;
    c.n_miss++;
    return true;
}

// Hand slot v to expert e for a demand read: evict what it held (unless this is
// an upgrade of e's own cold block) and mark it in flight and live.
void expert_cache::claim_slot(layer_pool & lp, uint32_t e, int32_t v, bool upgrade, bool take_cold) {
    assert_that(v >= 0 && (uint32_t) v < lp.n_slots, "victim slot within the layer");
    assert_that(e < lp.expert_slot.size(), "expert id within the layer");
    if (!upgrade && lp.slot_expert[v] != SLOT_EMPTY) {
        if (lp.slot_speculative[v]) st_.pf_wasted++;   // prefetched, never used, evicted by a miss
        lp.expert_slot[lp.slot_expert[v]] = -1;
        st_.evictions++;
    }
    lp.slot_expert[v]  = (uint16_t) e;
    lp.slot_valid[v]   = 0;
    lp.slot_freq[v]    = 1;
    lp.slot_used[v]    = ++tick_;
    lp.expert_slot[e]  = v;
    lp.slot_cold[v]    = take_cold ? 1 : 0;
    lp.slot_speculative[v] = 0; lp.slot_pf_fetch[v] = 0;
    if (upgrade && e < lp.g_expert_slot.size()) {
        const int32_t gs = lp.g_expert_slot[e];   // discard the stale cold VRAM copy
        if (gs >= 0) { lp.g_slot_expert[gs] = SLOT_EMPTY; lp.g_valid[gs] = 0; lp.g_expert_slot[e] = -1; lp.g_ver++; }
    }
    live_.push_back(v);
}

// Submit and return. The device fills these while the caller computes the
// experts that were already resident. Hot and cold blocks live in different
// files: each request goes to its file's engine, and fetch_end drains both.
// (One engine for all of them read hot misses out of the cold shards.)
bool expert_cache::submit_burst(const fetch_ctx & c, std::chrono::steady_clock::time_point t_enter,
                                double promote_base) {
    assert_that(c.n_req > 0 && c.n_req <= fetch_ctx::MAX_REQ, "a burst has reads, within the table");
    assert_that(c.n_miss <= 64 && inflight_slots_.empty() && inflight_experts_.empty(), "in-flight lists reset");
    io_request hot_reqs[fetch_ctx::MAX_REQ], cold_reqs[fetch_ctx::MAX_REQ];
    size_t n_hot = 0, n_cold = 0;
    for (size_t k = 0; k < c.n_req; k++) (c.req_cold[k] ? cold_reqs[n_cold++] : hot_reqs[n_hot++]) = c.reqs[k];
    if (!submit_all(io_hot_, hot_reqs, n_hot, inflight_reqs_)) return false;
    if (n_cold && !submit_all(io_cold_, cold_reqs, n_cold, inflight_cold_reqs_)) return false;
    st_.n_bursts++;
    st_.n_reads += c.n_req;
    st_.t_submit += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_enter).count() - st_.t_promote + promote_base;
    for (size_t k = 0; k < c.n_miss; k++) {
        inflight_slots_.push_back((int32_t) c.miss_slot[k]);
        inflight_experts_.push_back(c.miss_expert[k]);
    }
    return true;
}

// Submit every request to `eng`; when its ring is full, drain one completion and go on.
bool expert_cache::submit_all(io_engine & eng, const io_request * rq, size_t n_rq, size_t & inflight) {
    assert_that(rq != nullptr, "request table");
    assert_that(n_rq <= fetch_ctx::MAX_REQ, "no more requests than a burst holds");
    size_t submitted = 0, reaped = 0; uint64_t tags[256];
    while (submitted < n_rq) {
        const size_t k = eng.submit(rq + submitted, n_rq - submitted);
        if (k == 0) {   // ring full: drain one, then keep going
            const size_t got = eng.reap(tags, 256, 1);
            // Reads already accepted may still be in the kernel: count them so the abort drains them.
            if (got == 0) { inflight = submitted - std::min(submitted, reaped); return false; }
            inflight_failed_ |= any_failed(tags, got);
            reaped += got; continue;
        }
        submitted += k;
    }
    inflight = n_rq - std::min(n_rq, reaped);
    return true;
}

bool expert_cache::read_block_now(layer_pool & lp, uint32_t layer, int32_t slot, uint32_t expert) {
    assert_that(slot >= 0 && (uint32_t) slot < lp.n_slots, "RAM slot within the layer");
    assert_that(lp.slot_expert[slot] == (uint16_t) expert, "the slot is claimed by the expert read into it");
    io_request reqs[EXPERT_NPARTS];
    uint8_t * base = slot_ptr(lp, (uint32_t) slot);
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        const byte_range br = hot_->expert_range(layer, expert, (expert_part) q);
        if (!br.valid()) return false;
        reqs[q] = io_request{ br.shard, br.offset, br.nbytes, base + lp.part_off[q], 0 };
        st_.bytes_from_disk += br.nbytes;
    }
    size_t submitted = 0; uint64_t tags[16]; size_t reaped = 0; bool failed = false;
    while (submitted < EXPERT_NPARTS) {
        const size_t k = io_hot_.submit(reqs + submitted, EXPERT_NPARTS - submitted);
        if (k == 0) { const size_t got = io_hot_.reap(tags, 16, 1); if (got == 0) return false; failed |= any_failed(tags, got); reaped += got; continue; }
        submitted += k;
    }
    while (reaped < EXPERT_NPARTS) { const size_t got = io_hot_.reap(tags, 16, EXPERT_NPARTS - reaped); if (got == 0) return false; failed |= any_failed(tags, got); reaped += got; }
    st_.n_reads += EXPERT_NPARTS;
    if (failed) { fprintf(stderr, "[qwfn] expert read failed (layer %u, expert %u)\n", layer, expert); return false; }
    lp.slot_valid[slot] = 1;
    lp.slot_cold[slot]  = 0;
    return true;
}

bool expert_cache::fetch_batch(uint32_t layer, const uint32_t * expert_ids, uint32_t n, expert_handle * out,
                               uint8_t * bounce, size_t bounce_bytes) {
    return fetch_batch_begin(layer, expert_ids, n, out, bounce, bounce_bytes) && fetch_batch_end();
}

// The two halves of fetch_batch: begin classifies and submits the misses' reads,
// end waits for them. One batch in flight at a time; the caller computes the
// previous chunk between the two (cbatch pipelining).
bool expert_cache::fetch_batch_begin(uint32_t layer, const uint32_t * expert_ids, uint32_t n, expert_handle * out,
                                     uint8_t * bounce, size_t bounce_bytes) {
    bf_.active = false;
    if (layer >= blk_.size()) return false;
    // Speculative reads left over from a decode: land them, so that which blocks
    // count as resident here does not depend on when they complete.
    if (pf_reads_outstanding_ > 0) prefetch_settle();
    if (has_inflight() && !fetch_end()) return false;   // and a decode fetch left unfinished
    layer_pool & lp = blk_[layer];
    if (!bounce || bounce_bytes < (size_t) n * lp.block_bytes) return false;
    assert_that(n == 0 || (expert_ids != nullptr && out != nullptr), "batch expert ids and handles");
    assert_that(hot_ != nullptr && lp.expert_slot.size() == hot_->hp().n_expert, "expert map sized");
    bf_.layer = layer; bf_.reqs.clear(); bf_.reqs.reserve((size_t) n * EXPERT_NPARTS); bf_.adopted.clear();
    bf_.submitted = bf_.reaped = 0; bf_.failed = false;
    uint32_t n_miss = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t e = expert_ids[i];
        out[i] = expert_handle{};
        if (e >= hot_->hp().n_expert) { bf_.active = true; bf_.failed = true; return batch_fail(); }
        if (vram_buf_ && lp.g_slots > 0 && !lp.g_lent) {
            const int32_t gs = lp.g_expert_slot[e];
            if (gs >= 0 && lp.g_valid[gs] && lp.g_slot_expert[gs] == (uint16_t) e) { fill_gpu_handle(lp, (uint32_t) gs, out[i]); continue; }
        }
        {
            const int32_t s = lp.expert_slot[e];
            if (s >= 0 && lp.slot_expert[s] == (uint16_t) e && lp.slot_valid[s] && !lp.slot_cold[s]) { fill_handle(lp, (uint32_t) s, out[i]); continue; }
        }
        const int32_t adopt = batch_empty_slot(lp);
        uint8_t * base = adopt >= 0 ? slot_ptr(lp, (uint32_t) adopt) : bounce + (size_t) n_miss * lp.block_bytes;
        if (adopt < 0) n_miss++;
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const byte_range br = hot_->expert_range(layer, e, (expert_part) q);
            if (!br.valid()) { bf_.active = true; bf_.failed = true; return batch_fail(); }
            bf_.reqs.push_back(io_request{ br.shard, br.offset, br.nbytes, base + lp.part_off[q], 0 });
            out[i].part[q] = base + lp.part_off[q] + lp.part_pay[q];
            out[i].type[q] = lp.part_type[q];
            st_.bytes_from_disk += br.nbytes;
        }
        out[i].buffer = adopt >= 0 ? arena_buf_ : nullptr; out[i].on_gpu = false; out[i].from_cold = false; out[i].slot = adopt;
        if (adopt >= 0) {
            lp.slot_expert[adopt] = (uint16_t) e; lp.slot_valid[adopt] = 1; lp.slot_freq[adopt] = 1;
            lp.slot_used[adopt] = ++tick_; lp.slot_cold[adopt] = 0; lp.slot_speculative[adopt] = 0; lp.slot_pf_fetch[adopt] = 0;
            lp.expert_slot[e] = adopt;
            bf_.adopted.push_back(adopt);
        }
    }
    st_.batch_lookups += n; st_.batch_misses += n_miss;
    bf_.active = true;
    batch_submit_now();
    return true;
}

// An empty RAM slot takes the block (a fresh session's tier fills with the
// prompt's experts, as it should); a full tier is left alone and the block
// goes to the bounce. Never an eviction, never a frequency count.
int32_t expert_cache::batch_empty_slot(layer_pool & lp) {
    assert_that(lp.slot_expert.size() == lp.n_slots, "slot table sized to the layer");
    assert_that(lp.slot_pinned.size() == lp.n_slots, "pin table sized to the layer");
    int32_t adopt = -1;
    static thread_local uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (int k = 0; k < 16 && lp.n_slots > 0; k++) {
        const uint32_t s = (uint32_t) (xorshift(rng) % lp.n_slots);
        if (lp.slot_expert[s] == SLOT_EMPTY && !lp.slot_pinned[s]) { adopt = (int32_t) s; break; }
    }
    return adopt;
}

// Submit what the queue takes now; fetch_batch_end submits the rest.
void expert_cache::batch_submit_now() {
    assert_that(bf_.active, "a batch is in flight");
    assert_that(bf_.submitted == 0 && bf_.reaped == 0, "nothing of this batch submitted yet");
    uint64_t tags[256];
    while (bf_.submitted < bf_.reqs.size()) {
        const size_t k = io_hot_.submit(bf_.reqs.data() + bf_.submitted, std::min<size_t>(64, bf_.reqs.size() - bf_.submitted));
        if (k == 0) break;
        bf_.submitted += k;
    }
    // Opportunistic reap so a full ring cannot stall the caller's compute.
    const size_t got = io_hot_.reap(tags, 256, 0);
    bf_.failed |= any_failed(tags, got); bf_.reaped += got;
}

// A read that does not land leaves an adopted slot valid over stale bytes: empty them.
bool expert_cache::batch_fail() {
    assert_that(bf_.active, "only a batch in flight fails");
    assert_that(bf_.adopted.size() <= bf_.reqs.size(), "every adopted slot has its reads");
    if (bf_.layer < blk_.size()) {
        layer_pool & lp = blk_[bf_.layer];
        for (int32_t v : bf_.adopted) { lp.expert_slot[lp.slot_expert[v]] = -1; lp.slot_expert[v] = SLOT_EMPTY; lp.slot_valid[v] = 0; }
    }
    bf_.active = false; bf_.adopted.clear();
    return false;
}

bool expert_cache::fetch_batch_end() {
    if (!bf_.active) return false;
    assert_that(bf_.layer < blk_.size(), "batch layer within the model");
    assert_that(bf_.submitted <= bf_.reqs.size() && bf_.adopted.size() <= bf_.reqs.size(), "batch counters consistent");
    uint64_t tags[256];
    while (bf_.submitted < bf_.reqs.size() || bf_.reaped < bf_.reqs.size()) {
        if (bf_.submitted < bf_.reqs.size()) {
            const size_t k = io_hot_.submit(bf_.reqs.data() + bf_.submitted, std::min<size_t>(64, bf_.reqs.size() - bf_.submitted));
            bf_.submitted += k;
            if (k > 0 && bf_.submitted < bf_.reqs.size()) continue;
        }
        const size_t got = io_hot_.reap(tags, 256, bf_.reaped < bf_.submitted ? 1 : 0);
        if (got == 0 && bf_.reaped < bf_.submitted) return batch_fail();
        bf_.failed |= any_failed(tags, got);
        bf_.reaped += got;
    }
    st_.n_reads += bf_.reqs.size();
    bf_.active = false;
    if (bf_.failed) { fprintf(stderr, "[qwfn] expert read failed in batch fetch (layer %u)\n", bf_.layer); bf_.active = true; return batch_fail(); }
    return true;
}

bool expert_cache::settle_pending(const uint32_t * expert_ids, uint32_t n, bool * ready) {
    if (pending_.empty()) return true;
    assert_that(inflight_layer_ < blk_.size(), "pending experts belong to a fetched layer");
    assert_that(pending_.size() <= 64 && (n == 0 || expert_ids != nullptr), "pending list and ids");
    const auto tw = std::chrono::steady_clock::now();
    layer_pool & lp = blk_[inflight_layer_];
    std::vector<uint32_t> ids; ids.reserve(pending_.size());
    for (const pending_spec & p : pending_) ids.push_back(p.expert);
    wait_state_ = 2;
    prefetch_settle_for(inflight_layer_, ids.data(), (uint32_t) ids.size());
    wait_state_ = 0;
    const bool ok = pending_landed(lp);
    finish_deferred(lp, ok);
    if (!ok) { drop_unlanded_pending(lp); return false; }
    for (const pending_spec & p : pending_)
        if (ready) for (uint32_t i = 0; i < n; i++) if (expert_ids[i] == p.expert) ready[i] = true;
    pending_.clear();
    st_.t_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
    return true;
}

// After the wait: every pending block valid, or false.
bool expert_cache::pending_landed(layer_pool & lp) {
    assert_that(&lp == &blk_[inflight_layer_], "the pool of the fetched layer");
    assert_that(!pending_.empty(), "called with pending blocks");
    for (const pending_spec & p : pending_) {
        // The speculative engine lost track of this read (its count drifted and
        // was reset): fetch the block now rather than compute on stale bytes.
        // A cold block cannot be re-read: read_block_now reads the hot layout,
        // and its handle is already out with the cold one.
        if (lp.slot_expert[p.slot] != (uint16_t) p.expert) return false;
        if (!lp.slot_valid[p.slot] && (lp.slot_cold[p.slot] || !read_block_now(lp, inflight_layer_, p.slot, p.expert))) return false;
    }
    return true;
}

// The promotions fetch_begin decided for blocks still in flight: copied now
// that the data has landed, in lookup order -- queued on the device stream
// before any graph that reads them. Without the data (a failed read) the
// reserved slots are released instead of being left valid over stale bytes,
// and the promotion and VRAM hit counted at the reservation are taken back.
void expert_cache::finish_deferred(layer_pool & lp, bool ok) {
    assert_that(deferred_.size() <= pending_.size(), "only pending blocks are promoted late");
    assert_that(deferred_.empty() || vram_buf_ != nullptr, "late promotions only with a VRAM tier");
    for (const deferred_promo & d : deferred_) {
        const uint32_t e = d.expert;
        if (!ok) { st_.promotions--; st_.gpu_hits--; }   // counted by promote and try_promote, never copied
        const bool same = lp.g_slot_expert[d.gslot] == (uint16_t) e && lp.g_expert_slot[e] == d.gslot;
        if (!same) continue;   // released already (the tier was lent, or the expert re-read)
        if (ok) {
            assert_that(lp.slot_expert[d.slot] == (uint16_t) e && lp.slot_valid[d.slot], "the promoted block has landed");
            copy_to_gpu(lp, d.gslot, slot_ptr(lp, (uint32_t) d.slot));
            lp.g_valid[d.gslot] = 1;
        } else {
            lp.g_expert_slot[e] = -1; lp.g_slot_expert[d.gslot] = SLOT_EMPTY; lp.g_valid[d.gslot] = 0;
        }
        lp.g_ver++;
    }
    deferred_.clear();
}

bool expert_cache::fetch_end() {
    const bool pend_ok = settle_pending(nullptr, 0, nullptr);   // on failure the demand reads still get reaped
    if (inflight_reqs_ == 0 && inflight_cold_reqs_ == 0) return pend_ok;
    assert_that(inflight_layer_ < blk_.size(), "in-flight reads belong to a fetched layer");
    assert_that(inflight_slots_.size() == inflight_experts_.size(), "one expert per in-flight slot");
    const auto tw = std::chrono::steady_clock::now();

    wait_state_ = 1;
    const bool ok = drain_reads(io_hot_, inflight_reqs_) && drain_reads(io_cold_, inflight_cold_reqs_);
    wait_state_ = 0;
    if (!ok) {
        fprintf(stderr, "[qwfn] expert read wait returned short (%zu hot + %zu cold expected): the I/O engine returned no completion\n",
                inflight_reqs_, inflight_cold_reqs_);
    }
    inflight_reqs_ = inflight_cold_reqs_ = 0;
    layer_pool & lp = blk_[inflight_layer_];
    // A completion does not say which block it belonged to, and failures are
    // rare: when one read of the burst failed, re-read every block of it. A
    // cold block cannot be: read_block_now reads the hot layout, and its
    // handle is already out with the cold one. Once one block cannot be had
    // (or the wait came back short) the rest are emptied, not left invalid --
    // but only once both engines are idle (see abort_fetch); else they leak.
    bool all = ok;
    for (size_t k = 0; k < inflight_slots_.size(); k++) {
        const int32_t v = inflight_slots_[k];
        if (lp.slot_expert[v] != (uint16_t) inflight_experts_[k]) continue;
        if (all && !inflight_failed_) lp.slot_valid[v] = 1;
        else if (!all || lp.slot_cold[v] || !read_block_now(lp, inflight_layer_, v, inflight_experts_[k])) {
            all = false;
            if (demand_idle()) empty_slot(lp, v, inflight_experts_[k]);
        }
    }
    if (!all) return false;
    st_.t_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
    st_.bytes_read  = io_hot_.stat_bytes + io_cold_.stat_bytes;
    st_.read_errors = io_hot_.stat_errors + io_cold_.stat_errors + io_pf_.stat_errors;
    st_.read_short  = io_hot_.stat_short + io_cold_.stat_short + io_pf_.stat_short;
    return pend_ok;
}

// Reap n demand reads off `eng`, noting any failure; false if the engine runs dry first.
bool expert_cache::drain_reads(io_engine & eng, size_t n) {
    assert_that(&eng == &io_hot_ || &eng == &io_cold_, "a demand-read engine");
    assert_that(n <= fetch_ctx::MAX_REQ, "no more reads than a burst holds");
    uint64_t tags[256]; size_t done = 0;
    while (done < n) {
        const size_t got = eng.reap(tags, 256, n - done);
        if (got == 0) return false;
        inflight_failed_ |= any_failed(tags, got);
        done += got;
    }
    return true;
}

void expert_cache::prefetch(uint32_t layer, const uint32_t * expert_ids, uint32_t n) {
    if (layer >= blk_.size()) return;
    assert_that(n == 0 || expert_ids != nullptr, "expert ids");
    assert_that(hot_ != nullptr, "prefetch after init");
    layer_pool & lp = blk_[layer];
    io_request reqs[64 * EXPERT_NPARTS];
    std::vector<int32_t> claimed;
    size_t n_req = 0;
    for (uint32_t i = 0; i < n && n_req + EXPERT_NPARTS <= sizeof(reqs) / sizeof(reqs[0]); i++) {
        const uint32_t e = expert_ids[i];
        claimed.push_back(-1);
        if (find_slot(lp, e) >= 0) continue;
        const int32_t v = choose_victim(lp);
        if (v < 0) continue;
        if (lp.slot_expert[v] != SLOT_EMPTY) lp.expert_slot[lp.slot_expert[v]] = -1;
        claimed.back() = v;
        lp.slot_expert[v] = (uint16_t) e;
        lp.slot_valid[v]  = 0;   // becomes valid only once the read is reaped
        lp.slot_freq[v]   = 1;
        lp.expert_slot[e] = v;
        uint8_t * base = slot_ptr(lp, (uint32_t) v);
        for (int q = 0; q < EXPERT_NPARTS; q++) {   // prefetches always take full precision
            const byte_range br = hot_->expert_range(layer, e, (expert_part) q);
            if (!br.valid()) continue;
            reqs[n_req++] = io_request{ br.shard, br.offset, br.nbytes, base + lp.part_off[q], 0 };
        }
    }
    if (!n_req) return;

    // Submit and reap here. True fire-and-forget prefetch needs a persistent
    // in-flight table so a later fetch() can wait on a specific block; until
    // that exists, marking a slot resident before its read lands would be the
    // same aliasing hazard choose_victim guards against.
    io_engine * eng = &io_hot_;
    size_t sub = 0;
    uint64_t tags[256];
    while (sub < n_req) {
        const size_t k = eng->submit(reqs + sub, n_req - sub);
        if (k == 0) { if (eng->reap(tags, 256, 1) == 0) return; continue; }
        sub += k;
    }
    size_t done = 0;
    while (done < n_req) {
        const size_t got = eng->reap(tags, 256, n_req - done);
        if (got == 0) return;
        done += got;
    }
    for (uint32_t i = 0; i < n && i < claimed.size(); i++) {
        const int32_t v = claimed[i];
        if (v >= 0) lp.slot_valid[v] = 1;
    }
}

void expert_cache::ram_resident_slices(uint32_t layer, std::vector<ram_slice> & out) const {
    out.clear();
    if (layer >= blk_.size()) return;
    const layer_pool & lp = blk_[layer];
    assert_that(lp.slot_valid.size() == lp.n_slots && lp.slot_expert.size() == lp.n_slots, "slot tables sized");
    assert_that(lp.base != nullptr, "RAM tier laid out");
    for (uint32_t s = 0; s < lp.n_slots; s++) {
        if (!lp.slot_valid[s] || lp.slot_cold[s] || lp.slot_expert[s] == SLOT_EMPTY) continue;
        const uint16_t e = lp.slot_expert[s];
        if (e >= lp.expert_slot.size() || lp.expert_slot[e] != (int32_t) s) continue;   // an orphan
        ram_slice r; r.expert = e;
        const uint8_t * base = lp.base + (size_t) s * lp.block_bytes;
        for (int q = 0; q < EXPERT_NPARTS; q++) r.part[q] = base + lp.part_off[q] + lp.part_pay[q];
        out.push_back(r);
    }
}

void expert_cache::settle_promotions() {
    if (pending_release_.empty()) return;
    assert_that(vram_buf_ != nullptr, "promotions only exist with a VRAM tier");
    assert_that(cfg_.vram_backend != nullptr && arena_pinned_, "only asynchronous promotions are deferred");
    if (cfg_.vram_backend) ggml_backend_synchronize(cfg_.vram_backend);
    // The RAM copy stays. With recency-based VRAM eviction a promoted expert
    // can leave VRAM again soon; if its RAM copy is gone that is a disk read
    // (measured: hit rate 96.1% -> 94.4%, 40% more reads). Lookups check VRAM
    // first, so the RAM copy is never touched again and recency retires it
    // on its own; while it lives it is a fallback. Only the eviction
    // protection is lifted here.
    pending_release_.clear();
}

void expert_cache::lend_begin() {
    // A prefill reads slot_valid (warm-up, RAM-resident slices, the refill at
    // lend_end): land the speculative reads first so none of that depends on timing.
    if (pf_reads_outstanding_ > 0) prefetch_settle();
    // A fetch left unfinished: its reserved promotions must not outlive the tier.
    if (has_inflight() && !fetch_end()) fprintf(stderr, "[qwfn] expert read failed before a prefill\n");
    if (!vram_extra_) return;
    assert_that(vram_buf_ != nullptr, "the dynamic part exists only beside the permanent one");
    assert_that(extra_bytes_ > 0, "a dynamic part has a size");
    settle_promotions();   // copies still landing in those slots must finish first
    for (layer_pool & lp : blk_) {
        if (!lp.g_in_lent) continue;
        lp.g_lent = true;
        std::fill(lp.g_slot_expert.begin(), lp.g_slot_expert.end(), SLOT_EMPTY);
        std::fill(lp.g_valid.begin(), lp.g_valid.end(), 0);
        std::fill(lp.g_expert_slot.begin(), lp.g_expert_slot.end(), -1);
        lp.g_ver++;
    }
    ggml_backend_buffer_free(vram_extra_);
    vram_extra_ = nullptr;
    tier_epoch_++;
}

void expert_cache::lend_end() {
    if (!extra_bytes_ || vram_extra_) return;
    assert_that(vram_buf_ != nullptr, "the dynamic part exists only beside the permanent one");
    assert_that(cfg_.vram_buft != nullptr, "device buffer type set");
    // A prefill just bumped the prompt's experts' counters (capped, but over
    // many ubatches); halve everything so the generation that follows can
    // displace them as its own routing settles.
    for (layer_pool & lp : blk_) for (auto & f : lp.ef) f >>= 1;
    vram_extra_ = ggml_backend_buft_alloc_buffer(cfg_.vram_buft, extra_bytes_);
    if (!vram_extra_) {
        static bool warned = false;
        if (!warned) { warned = true; fprintf(stderr, "[qwfn] could not take the dynamic VRAM tier back after the prefill; those layers stay CPU-served until the next one\n"); }
        return;
    }
    // Zeroed: slot 0 of every layer is the zero-weight dummy of the mul_mat_id
    // MoE, and stale bytes decoded as this layer's type can be NaN scales.
    ggml_backend_buffer_clear(vram_extra_, 0);
    uint8_t * gbase = (uint8_t *) ggml_backend_buffer_get_base(vram_extra_);
    for (layer_pool & lp : blk_) {
        if (!lp.g_in_lent) continue;
        lp.g_buf = vram_extra_;
        size_t goff = lp.g_off;
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            lp.g_part[q] = gbase + goff;
            goff += (size_t) lp.g_slots * lp.g_part_bytes[q];
        }
        lp.g_lent = false;
    }
    tier_epoch_++;
    if (pf_reads_outstanding_ > 0) prefetch_settle();   // the refill reads slot_valid
    refill_lent_layers();
}

// Those layers come back with empty VRAM tiers. Refill them from what the
// RAM tier holds -- the prefill's warm-up just put the prompt's experts
// there -- hottest first, up to the tier's size. Asynchronous copies from
// the pinned arena; the RAM slots are protected until the next settle.
void expert_cache::refill_lent_layers() {
    assert_that(vram_extra_ != nullptr, "the dynamic buffer is back");
    assert_that(vram_buf_ != nullptr, "the permanent buffer exists");
    for (uint32_t il = 0; il < blk_.size(); il++) {
        layer_pool & lp = blk_[il];
        if (!lp.g_in_lent || lp.g_slots == 0) continue;
        std::vector<std::pair<uint32_t, uint32_t>> cand;   // (ef, expert)
        for (uint32_t s = 0; s < lp.n_slots; s++)
            if (lp.slot_valid[s] && lp.slot_expert[s] != SLOT_EMPTY && !lp.slot_cold[s])
                cand.emplace_back(lp.ef[lp.slot_expert[s]], lp.slot_expert[s]);
        std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        uint32_t n = 0;
        for (const auto & c : cand) {
            if (n >= lp.g_slots) break;
            const uint32_t e = c.second;
            const int32_t  s = lp.expert_slot[e];
            if (s < 0 || !lp.slot_valid[s] || lp.slot_expert[s] != (uint16_t) e) continue;
            if (!promote(lp, e, slot_ptr(lp, (uint32_t) s))) continue;
            n++;
            st_.warm_promoted++;
            if (last_promote_async_) pending_release_.push_back(pending_rel{ il, e, s });
        }
    }
}

void expert_cache::warm(uint32_t layer, const warm_item * items, uint32_t n, uint32_t n_vram) {
    if (layer >= blk_.size() || n == 0) return;
    assert_that(items != nullptr, "warm items");
    assert_that(blk_[layer].expert_slot.size() == blk_[layer].ef.size(), "per-expert tables agree");
    const auto t0 = std::chrono::steady_clock::now();
    layer_pool & lp = blk_[layer];
    live_.clear();

    std::vector<warm_copy> copies;
    std::vector<int32_t>   admitted;      // slots filled by this call
    std::vector<uint32_t>  to_promote;

    for (uint32_t i = 0; i < n; i++) {
        const uint32_t e = items[i].expert;
        if (e >= lp.ef.size()) continue;
        // Capped: a prompt's raw use counts (hundreds over a long prefill)
        // would outrank anything decode admits afterwards -- promote() only
        // displaces a colder resident -- and the VRAM tier then freezes on the
        // prompt's experts. Measured after a 32K prefill: 34% VRAM-served.
        lp.ef[e]  += std::min<uint32_t>(items[i].count, 4);
        lp.seen[e] = 1;

        if (lp.g_slots > 0) {
            const int32_t gs = lp.g_expert_slot[e];
            if (gs >= 0 && lp.g_valid[gs] && lp.g_slot_expert[gs] == (uint16_t) e) {
                lp.g_freq[gs] += items[i].count;
                continue;                                    // already where it is most useful
            }
        }
        const int32_t s = warm_admit(lp, items[i], copies, admitted);
        if (s < 0) continue;
        lp.slot_used[s] = ++tick_;
        live_.push_back(s);
        if (i < n_vram) to_promote.push_back(e);
    }

    run_copies(copies);
    for (int32_t s : admitted) lp.slot_valid[s] = 1;
    st_.warm_admitted += admitted.size();

    for (uint32_t e : to_promote) {
        const int32_t s = lp.expert_slot[e];
        if (s < 0 || !lp.slot_valid[s] || lp.slot_expert[s] != (uint16_t) e) continue;
        if (!promote(lp, e, slot_ptr(lp, (uint32_t) s))) continue;
        st_.warm_promoted++;
        if (last_promote_async_) pending_release_.push_back(pending_rel{ layer, e, s });
    }
    live_.clear();
    st_.t_warm += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// The RAM slot a warm item lands in: its resident hot copy, or a claimed victim
// whose copy is queued in `copies`. -1 when no slot can be had.
int32_t expert_cache::warm_admit(layer_pool & lp, const warm_item & item, std::vector<warm_copy> & copies,
                                 std::vector<int32_t> & admitted) {
    const uint32_t e = item.expert;
    assert_that(e < lp.ef.size(), "warm expert within the layer");
    assert_that(admitted.size() <= lp.n_slots, "no more admissions than slots");
    int32_t s = find_slot(lp, e);
    if (s >= 0 && lp.slot_cold[s]) {                     // replace a cold copy outright
        lp.expert_slot[e] = -1; lp.slot_expert[s] = SLOT_EMPTY; lp.slot_valid[s] = 0;
        s = -1;
    }
    if (s < 0) {
        s = choose_victim(lp);
        if (s < 0) return -1;
        if (lp.slot_expert[s] != SLOT_EMPTY) {
            if (lp.slot_speculative[s]) st_.pf_wasted++;
            lp.expert_slot[lp.slot_expert[s]] = -1;
            st_.evictions++;
        }
        lp.slot_expert[s]      = (uint16_t) e;
        lp.slot_valid[s]       = 0;                      // valid once the copy is done
        lp.slot_cold[s]        = 0;
        lp.slot_speculative[s] = 0; lp.slot_pf_fetch[s] = 0;
        lp.slot_freq[s]        = 1;
        lp.expert_slot[e]      = s;
        uint8_t * base = slot_ptr(lp, (uint32_t) s);
        for (int q = 0; q < EXPERT_NPARTS; q++)
            if (item.part[q])
                copies.push_back(warm_copy{ base + lp.part_off[q] + lp.part_pay[q], item.part[q], lp.part_bytes[q] });
        admitted.push_back(s);
    } else {
        lp.slot_freq[s]++;
    }
    return s;
}

void expert_cache::pin(const std::vector<uint32_t> & packed_keys) {
    assert_that(hot_ != nullptr, "pin after init");
    assert_that(blk_.size() == hot_->hp().n_layer, "one pool per layer");
    for (uint32_t k : packed_keys) {
        const uint32_t l = k >> 16, e = k & 0xFFFF;
        if (l >= blk_.size() || e >= hot_->hp().n_expert) continue;
        const int32_t s = blk_[l].expert_slot[e];
        if (s >= 0) blk_[l].slot_pinned[s] = 1;
    }
}


// One speculative read has completed (or was never accepted: tag carries IO_TAG_FAILED).
// The block becomes valid when its last read lands; a failed block is emptied.
void expert_cache::pf_finish(uint64_t tag) {
    // A completion can outlive its table: after a count drift pf_reap resets, and the next
    // prefetch clears pf_pending_. Such a late, stale tag has nothing left to update.
    if ((tag & ~IO_TAG_FAILED) >= pf_pending_.size()) { st_.read_errors++; return; }
    pf_entry & e = pf_pending_[tag & ~IO_TAG_FAILED];
    assert_that(e.layer < blk_.size(), "pending entry's layer within the model");
    assert_that(e.slot >= 0 && (size_t) e.slot < blk_[e.layer].slot_expert.size(), "pending entry's slot within its layer");
    e.failed |= (tag & IO_TAG_FAILED) != 0;
    if (e.remaining == 0 || --e.remaining != 0) return;
    layer_pool & lp = blk_[e.layer];
    if (lp.slot_expert[e.slot] != (uint16_t) e.expert) return;
    if (!e.failed) { lp.slot_valid[e.slot] = 1; return; }
    fprintf(stderr, "[qwfn] speculative expert read failed (layer %u, expert %u)\n", e.layer, e.expert);
    // A fetch already waiting on it re-reads it (settle_pending), or empties it
    // when it cannot (drop_unlanded_pending). Otherwise empty it now: an
    // invalid slot counts as in flight and would never be evicted.
    bool waited_on = false;
    if (e.layer == inflight_layer_) for (const pending_spec & ps : pending_) waited_on |= ps.slot == e.slot;
    if (!waited_on) { lp.expert_slot[e.expert] = -1; lp.slot_expert[e.slot] = SLOT_EMPTY; }
}

// Drain completions off the speculative engine, marking each block valid the
// moment its last read has landed. Tags index pf_pending_.
void expert_cache::pf_reap(size_t min_complete) {
    if (pf_reads_outstanding_ == 0) return;
    assert_that(!pf_pending_.empty(), "outstanding reads have pending entries");
    uint64_t tags[256];
    size_t need = std::min(min_complete, pf_reads_outstanding_);
    assert_that(need <= pf_reads_outstanding_, "never wait for more than is outstanding");
    do {
        wait_state_ = need > 0 ? 2 : wait_state_;
        const size_t got = io_pf_.reap(tags, 256, need);
        wait_state_ = 0;
        if (got == 0) {
            if (need > 0) {
                // Asked for a completion and got none: the outstanding count has
                // drifted past what the engine holds. Resynchronise rather than
                // wait for a read that is not there.
                fprintf(stderr, "[qwfn] speculative read count drifted (%zu outstanding, engine idle); resetting\n", pf_reads_outstanding_);
                pf_reads_outstanding_ = 0;
            }
            break;
        }
        for (size_t k = 0; k < got; k++) pf_finish(tags[k]);
        pf_reads_outstanding_ -= std::min(got, pf_reads_outstanding_);
        need = need > got ? need - got : 0;
    } while (need > 0 && pf_reads_outstanding_ > 0);
}

bool expert_cache::pf_layer_pending(uint32_t layer) const {
    assert_that(layer < blk_.size(), "layer within the model");
    assert_that(pf_reads_outstanding_ > 0, "asked only while reads are outstanding");
    for (const pf_entry & e : pf_pending_)
        if (e.layer == layer && e.remaining > 0) return true;
    return false;
}

// Wait only for the entries targeting `layer` -- the layer whose fetch is about
// to run. Later layers' entries keep flying and go valid whenever they land.
void expert_cache::prefetch_settle_layer(uint32_t layer) {
    assert_that(layer < blk_.size(), "layer within the model");
    assert_that(pf_reads_outstanding_ == 0 || !pf_pending_.empty(), "outstanding reads have pending entries");
    while (pf_reads_outstanding_ > 0 && pf_layer_pending(layer)) pf_reap(1);
    if (pf_reads_outstanding_ == 0) pf_pending_.clear();
}

void expert_cache::prefetch_settle_for(uint32_t layer, const uint32_t * ids, uint32_t n) {
    assert_that(layer < blk_.size(), "layer within the model");
    assert_that(n == 0 || ids != nullptr, "expert ids");
    auto needed_pending = [&]() {
        for (const pf_entry & e : pf_pending_) {
            if (e.layer != layer || e.remaining == 0) continue;
            for (uint32_t i = 0; i < n; i++) if (ids[i] == e.expert) return true;
        }
        return false;
    };
    while (pf_reads_outstanding_ > 0 && needed_pending()) pf_reap(1);
    if (pf_reads_outstanding_ == 0) pf_pending_.clear();
}

uint64_t expert_cache::vram_version(uint32_t layer) const {
    assert_that(blk_.size() < (1u << 16), "layer count fits the packed key");
    assert_that(vram_buf_ != nullptr || total_gslots_ == 0, "VRAM slots only with a VRAM tier");
    return layer < blk_.size() ? blk_[layer].g_ver : 0;
}

void expert_cache::vram_table(uint32_t layer, int32_t * slot, float * mask) const {
    assert_that(slot != nullptr && mask != nullptr, "residency tables");
    assert_that(hot_ != nullptr, "vram_table after init");
    const uint32_t n_expert = hot_->hp().n_expert;
    std::fill(slot, slot + n_expert, 0);
    std::fill(mask, mask + n_expert, 0.0f);
    if (layer >= blk_.size()) return;
    const layer_pool & lp = blk_[layer];
    if (!vram_buf_ || lp.g_slots == 0 || lp.g_lent) return;
    for (uint32_t e = 0; e < n_expert && e < lp.g_expert_slot.size(); e++) {
        const int32_t gs = lp.g_expert_slot[e];
        if (gs >= 0 && lp.g_valid[gs] && lp.g_slot_expert[gs] == (uint16_t) e) { slot[e] = gs; mask[e] = 1.0f; }
    }
}

void expert_cache::prefetch_settle() {
    while (pf_reads_outstanding_ > 0) pf_reap(1);
    pf_pending_.clear();
    assert_that(pf_reads_outstanding_ == 0, "every speculative read settled");
    assert_that(pf_pending_.empty(), "pending table drained");
}

void expert_cache::prefetch_begin(const pf_set * sets, uint32_t n_sets) {
    assert_that(n_sets == 0 || sets != nullptr, "prediction sets");
    assert_that(hot_ != nullptr, "prefetch after init");
    pf_reap(0);                                   // opportunistic drain
    if (pf_reads_outstanding_ == 0) pf_pending_.clear();
    // The table is tag-addressed and append-only while reads are in flight;
    // this bound should be unreachable (two sets of <= n_expert_used each).
    if (pf_pending_.size() > 4096) prefetch_settle();

    io_request reqs[PF_MAX_REQ];
    size_t n_req = 0;

    for (uint32_t s = 0; s < n_sets; s++) {
        const uint32_t layer = sets[s].layer;
        if (layer >= blk_.size()) continue;
        layer_pool & lp = blk_[layer];

        for (uint32_t i = 0; i < sets[s].n && n_req + EXPERT_NPARTS * 8 <= sizeof(reqs)/sizeof(reqs[0]); i++)
            pf_claim(layer, lp, sets[s].ids[i], reqs, n_req);
    }
    pf_recent_[1] = pf_recent_[0];
    pf_recent_[0] = n_req;

    if (n_req == 0) return;
    pf_submit(reqs, n_req);
}

// Claim a slot for one predicted expert and queue its speculative reads.
void expert_cache::pf_claim(uint32_t layer, layer_pool & lp, uint32_t e, io_request * reqs, size_t & n_req) {
    assert_that(reqs != nullptr && n_req + EXPERT_NPARTS <= PF_MAX_REQ, "room for one block's reads");
    assert_that(layer < blk_.size() && &blk_[layer] == &lp, "the pool of the predicted layer");
    if (e >= hot_->hp().n_expert) return;
    if (lp.g_slots > 0 && lp.g_expert_slot[e] >= 0) return;   // already in VRAM
    // Resident, being fetched, or already inbound from an earlier
    // prediction (the L+1 and L+2 sets for one layer overlap heavily,
    // and a slot claimed by an in-flight read is not yet `valid`, so
    // find_slot alone would re-claim and re-read it).
    const int32_t cs = lp.expert_slot[e];
    if (cs >= 0 && lp.slot_expert[cs] == (uint16_t) e) return;

    const int32_t v = choose_victim(lp);
    if (v < 0) return;
    if (lp.slot_expert[v] != SLOT_EMPTY) {
        // A speculative block that never got used is pure waste; count it.
        if (lp.slot_speculative[v]) st_.pf_wasted++;
        lp.expert_slot[lp.slot_expert[v]] = -1;
        st_.evictions++;
    }
    lp.slot_expert[v]      = (uint16_t) e;
    lp.slot_valid[v]       = 0;      // valid once its reads land
    lp.slot_freq[v]        = 1;
    lp.slot_used[v]        = ++tick_;
    lp.slot_speculative[v] = 1;
    lp.slot_pf_fetch[v]    = fetch_count_ + 1;
    lp.expert_slot[e]      = v;
    lp.seen[e]             = 1;

    const uint64_t tag = (uint64_t) pf_pending_.size();
    pf_pending_.push_back(pf_entry{ layer, e, v, 0 });
    pf_entry & ent = pf_pending_.back();

    uint8_t * base = slot_ptr(lp, (uint32_t) v);
    if (cold_ && !lp.hotw[e] && would_promote(lp, e)) lp.hotw[e] = 1;   // a VRAM candidate is read at full precision
    const bool pf_cold = cold_ != nullptr && !lp.hotw[e];   // the tail comes from the cold file
    lp.slot_cold[v] = pf_cold ? 1 : 0;
    if (pf_cold) st_.cold_tier_reads++;
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        const byte_range br = (pf_cold ? cold_ : hot_)->expert_range(layer, e, (expert_part) q);
        if (!br.valid()) continue;
        reqs[n_req++] = io_request{ br.shard + (pf_cold ? (int) n_hot_shards_ : 0), br.offset, br.nbytes, base + lp.part_off[q], tag };
        ent.remaining++;
        st_.bytes_from_disk += br.nbytes;
    }
    if (ent.remaining == 0) {   // nothing to read: mark it now
        if (lp.slot_expert[v] == (uint16_t) e) lp.slot_valid[v] = 1;
        pf_pending_.pop_back();
        return;
    }
    st_.pf_issued++;
}

// Account each read as it is accepted, so pf_reap can make progress if the
// ring ever fills mid-submit.
void expert_cache::pf_submit(const io_request * reqs, size_t n_req) {
    assert_that(reqs != nullptr, "speculative request table");
    assert_that(n_req > 0 && n_req <= PF_MAX_REQ, "a non-empty batch within the table");
    size_t sub = 0;
    while (sub < n_req) {
        const size_t k = io_pf_.submit(reqs + sub, n_req - sub);
        sub += k;
        pf_reads_outstanding_ += k;
        if (k > 0) continue;
        if (pf_reads_outstanding_ == 0) {
            // The engine accepts nothing and has nothing left to complete, so waiting cannot
            // help: give up the rest, as failed reads, rather than spin.
            fprintf(stderr, "[qwfn] I/O engine accepted no speculative reads; dropping %zu\n", n_req - sub);
            for (size_t r = sub; r < n_req; r++) pf_finish(reqs[r].tag | IO_TAG_FAILED);
            break;
        }
        pf_reap(1);
    }
}

} // namespace qwfn
