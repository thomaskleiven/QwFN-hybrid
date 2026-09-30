#include "qwfn_weights.h"
#include "qwfn_check.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <pthread.h>
#include <sched.h>
#include <string>
#include "ggml-cpu.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace qwfn {

weights::~weights() {
    assert_that(map_base_.size() == map_size_.size(), "mapping tables sized together");
    assert_that(map_size_.size() == map_buf_.size(), "mapping buffers match the mappings");
    if (tpool_ && dev_) {
        auto tp_free = (void (*)(struct ggml_threadpool *)) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev_), "ggml_threadpool_free");
        if (backend_) {
            auto set_tp = (void (*)(ggml_backend_t, struct ggml_threadpool *)) ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(dev_), "ggml_backend_cpu_set_threadpool");
            if (set_tp) set_tp(backend_, nullptr);
        }
        if (tp_free) tp_free(tpool_);
    }
    for (auto b : map_buf_) if (b) ggml_backend_buffer_free(b);
    for (size_t i = 0; i < map_base_.size(); i++)
        if (map_base_[i]) munmap(map_base_[i], map_size_[i]);
    if (buf_)     ggml_backend_buffer_free(buf_);
    if (ctx_)     ggml_free(ctx_);
    if (backend_) ggml_backend_free(backend_);
}

// "a-b,c" (as QWFN_CPUS or sysfs writes it, trailing newline allowed) into `mask`; returns the
// number of cpus set. A malformed entry ends the list with a message.
static int parse_cpu_list(std::string list, bool * mask) {
    assert_that(mask != nullptr, "a cpu mask to fill");
    while (!list.empty() && (list.back() == '\n' || list.back() == ' ')) list.pop_back();
    int n_mask = 0;
    size_t i = 0;
    while (i < list.size()) {
        size_t c = list.find(',', i); if (c == std::string::npos) c = list.size();
        assert_that(c >= i && c <= list.size(), "the entry lies inside the list");
        const std::string tok = list.substr(i, c - i);
        const size_t d = tok.find('-');
        long long a = 0, b = 0;
        const bool ok = parse_int(tok.substr(0, d).c_str(), 0, 1 << 20, a) &&
                        (d == std::string::npos ? (b = a, true) : parse_int(tok.c_str() + d + 1, 0, 1 << 20, b));
        if (!ok) { fprintf(stderr, "[qwfn] ignoring the rest of the CPU list '%s' at '%s'\n", list.c_str(), tok.c_str()); break; }
        for (long long k = a; k <= b && k < GGML_MAX_N_THREADS; k++) { mask[k] = true; n_mask++; }
        i = c + 1;
    }
    return n_mask;
}

void weights::set_n_threads(int n) {
    assert_that(!tpool_ || (backend_ && dev_), "a threadpool implies a CPU backend");
    if (!backend_ || !dev_) return;
    auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev_), "ggml_backend_set_n_threads");
    if (fn) fn(backend_, n);
    if (ggml_backend_dev_type(dev_) != GGML_BACKEND_DEVICE_TYPE_CPU || getenv("QWFN_NO_TPOOL")) return;
    // Without a threadpool ggml-cpu builds a disposable one per graph compute: n-1
    // pthread_create + join for every CPU expert graph, several per layer. A
    // persistent pool pinned to the performance cores (Intel hybrid: cpu_core)
    // keeps the workers warm; the arithmetic is unchanged.
    auto reg = ggml_backend_dev_backend_reg(dev_);
    auto tp_new  = (struct ggml_threadpool * (*)(struct ggml_threadpool_params *)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto tp_free = (void (*)(struct ggml_threadpool *)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
    auto set_tp  = (void (*)(ggml_backend_t, struct ggml_threadpool *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
    if (!tp_new || !tp_free || !set_tp) return;
    ggml_threadpool_params p = ggml_threadpool_params_default(n);
    p.poll = 50;
    // CPU list: QWFN_CPUS="a-b,c" or the hybrid P-core list, else default affinity.
    std::string list;
    if (const char * e = getenv("QWFN_CPUS")) list = e;
    else if (FILE * f = fopen("/sys/devices/cpu_core/cpus", "r")) { char b[128] = {0}; if (fgets(b, sizeof b, f)) list = b; fclose(f); }
    const int n_mask = parse_cpu_list(list, p.cpumask);
    p.strict_cpu = n_mask >= n;
    if (tpool_) { set_tp(backend_, nullptr); tp_free(tpool_); tpool_ = nullptr; }
    assert_that(tpool_ == nullptr, "the old threadpool is gone before a new one");
    // ggml_threadpool_new pins the CALLING thread to worker 0's cpu; every thread
    // it creates later (readers, HTTP workers) would inherit that single core.
    // Keep the caller's affinity as it was.
    cpu_set_t prev; const bool have_prev = pthread_getaffinity_np(pthread_self(), sizeof prev, &prev) == 0;
    tpool_ = tp_new(&p);
    if (have_prev) {
        // Can fail at runtime (a cpuset change, CPU hotplug) when threads are re-tuned: warn, go on.
        if (pthread_setaffinity_np(pthread_self(), sizeof prev, &prev) != 0)
            fprintf(stderr, "[qwfn] could not restore the calling thread's CPU affinity\n");
    }
    if (tpool_) set_tp(backend_, tpool_);
    fprintf(stderr, "[qwfn] CPU threadpool: %d threads, poll %u, cpus %s%s\n", n, p.poll,
            list.empty() ? "default" : list.c_str(), p.strict_cpu ? " (pinned)" : "");
}

void weights::set_n_threads_nopool(int n) {
    assert_that(!tpool_ || (backend_ && dev_), "a threadpool implies a CPU backend");
    if (!backend_ || !dev_) return;
    assert_that(backend_ != nullptr && dev_ != nullptr, "backend initialised");
    auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev_), "ggml_backend_set_n_threads");
    if (fn) fn(backend_, n);
}

const char * weights::dev_name() const {
    assert_that(!backend_ || dev_, "a backend implies its device");
    assert_that(!on_gpu_ || dev_, "a GPU placement implies a device");
    return dev_ ? ggml_backend_dev_name(dev_) : "none";
}

bool weights::init(const model_index * mi, bool prefer_gpu,
                   const std::string & backend_dir, std::string & err) {
    assert_that(mi != nullptr, "weights need a model index");
    assert_that(ctx_ == nullptr && backend_ == nullptr, "weights initialised once");
    mi_ = mi;

    if (backend_dir.empty()) ggml_backend_load_all();
    else                     ggml_backend_load_all_from_path(backend_dir.c_str());

    if (prefer_gpu) {
        dev_ = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        on_gpu_ = dev_ != nullptr;
    }
    if (!dev_) {
        dev_ = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        on_gpu_ = false;
    }
    if (!dev_) { err = "no ggml backend device available"; return false; }

    backend_ = ggml_backend_dev_init(dev_, nullptr);
    if (!backend_) { err = "failed to init backend device"; return false; }
    buft_ = ggml_backend_dev_buffer_type(dev_);

    // no_alloc: tensors are declared first, then backed by one buffer in commit().
    // Headroom for ~1500 tensor descriptors.
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 4096;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ctx_ = ggml_init(ip);
    if (!ctx_) { err = "ggml_init failed"; return false; }
    return true;
}

ggml_tensor * weights::declare(const std::string & name) {
    assert_that(mi_ != nullptr && ctx_ != nullptr, "weights initialised before declaring");
    assert_that(by_name_.size() >= pending_.size(), "every pending tensor is named");
    auto it = by_name_.find(name);
    if (it != by_name_.end()) return it->second;

    const tensor_ref * ref = mi_->find(name);
    if (!ref) return nullptr;

    // GGUF stores ne[] already in ggml order, so this mirrors the file exactly.
    ggml_tensor * t = ggml_new_tensor_4d(ctx_, ref->type, ref->ne[0], ref->ne[1], ref->ne[2], ref->ne[3]);
    if (!t) return nullptr;
    ggml_set_name(t, name.c_str());

    by_name_[name] = t;
    pending_.emplace_back(t, ref);
    declared_bytes_ += ref->nbytes;
    return t;
}

bool weights::declare_dense_core(std::string & err) {
    assert_that(mi_ != nullptr, "weights initialised with a model index");
    assert_that(ctx_ != nullptr, "weights context created");
    for (const auto & kv : mi_->tensors()) {
        const std::string & n = kv.first;
        // Routed experts are streamed by expert_cache; the PLE table lives on NVMe.
        if (n.find("_exps.weight") != std::string::npos) continue;
        if (n == "per_layer_token_embd.weight")          continue;
        if (n == "token_embd.weight")                    continue;   // host mapping; rows are gathered there
        if (!declare(n)) { err = "failed to declare " + n; return false; }
    }
    return true;
}

bool weights::commit(std::string & err) {
    assert_that(ctx_ != nullptr && buft_ != nullptr, "weights initialised before commit");
    assert_that(buf_ == nullptr, "weights committed once");
    buf_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft_);
    if (!buf_) {
        err = "failed to allocate " + std::to_string(declared_bytes_ >> 20) + " MiB on " + dev_name();
        return false;
    }

    // One fd per shard, plain buffered reads: this is a single 5.35 GB pass at load
    // time, not a hot path, and the page cache warming here is harmless.
    std::vector<int> fds;
    for (const auto & p : mi_->shard_paths()) {
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) {
            for (int f : fds) ::close(f);
            err = "open failed: " + p + ": " + strerror(errno);
            return false;
        }
        fds.push_back(fd);
    }

    std::vector<uint8_t> staging;
    bool ok = true;
    for (auto & [t, ref] : pending_) {
        staging.resize(ref->nbytes);
        size_t done = 0;
        while (done < ref->nbytes) {
            const ssize_t n = ::pread(fds[ref->shard], staging.data() + done,
                                      ref->nbytes - done, (off_t) (ref->file_offset + done));
            if (n <= 0) { err = "short read on " + ref->name; ok = false; break; }
            done += (size_t) n;
        }
        if (!ok) break;
        ggml_backend_tensor_set(t, staging.data(), 0, ref->nbytes);
    }

    for (int f : fds) ::close(f);
    return ok;
}

ggml_tensor * weights::get(const std::string & name) const {
    auto it = by_name_.find(name);
    assert_that(it == by_name_.end() || it->second != nullptr, "only declared tensors are named");
    assert_that(by_name_.size() >= pending_.size(), "every pending tensor is named");
    return it == by_name_.end() ? nullptr : it->second;
}


bool weights::map_shards(std::string & err) {
    assert_that(mi_ != nullptr, "weights initialised with a model index");
    assert_that(map_base_.size() == map_buf_.size(), "mapping tables sized together");
    for (const auto & p : mi_->shard_paths()) {
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) { err = "open failed: " + p; return false; }
        const off_t sz = ::lseek(fd, 0, SEEK_END);
        if (sz <= 0) { ::close(fd); err = "cannot size " + p; return false; }
        void * base = ::mmap(nullptr, (size_t) sz, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (base == MAP_FAILED) { err = "mmap failed: " + p; return false; }

        // Expert access is scattered by the router; sequential readahead would
        // only evict pages we still want.
        (void) ::madvise(base, (size_t) sz, MADV_RANDOM);   // advisory: failure only costs readahead

        map_base_.push_back(base);
        map_size_.push_back((size_t) sz);
        map_buf_.push_back(ggml_backend_cpu_buffer_from_ptr(base, (size_t) sz));   // freed by the destructor either way
        if (!map_buf_.back()) { err = "failed to wrap the mapping of " + p; return false; }
        mapped_bytes_ += (size_t) sz;
    }
    return true;
}

ggml_tensor * weights::declare_mapped(const std::string & name) {
    assert_that(mi_ != nullptr && ctx_ != nullptr, "weights initialised before declaring");
    assert_that(map_base_.size() == map_buf_.size(), "mapping tables sized together");
    auto it = by_name_.find(name);
    if (it != by_name_.end()) return it->second;

    const tensor_ref * ref = mi_->find(name);
    if (!ref) return nullptr;
    if (ref->shard < 0 || (size_t) ref->shard >= map_base_.size()) return nullptr;

    ggml_tensor * t = ggml_new_tensor_4d(ctx_, ref->type, ref->ne[0], ref->ne[1], ref->ne[2], ref->ne[3]);
    if (!t) return nullptr;
    ggml_set_name(t, name.c_str());

    // Point the tensor at its bytes inside the mapping. No copy, no allocation.
    t->buffer = map_buf_[ref->shard];
    t->data   = (char *) map_base_[ref->shard] + ref->file_offset;

    by_name_[name] = t;
    declared_bytes_ += ref->nbytes;
    return t;
}

bool weights::declare_all_mapped(std::string & err) {
    assert_that(mi_ != nullptr, "weights initialised with a model index");
    if (map_base_.empty() && !map_shards(err)) return false;
    assert_that(map_base_.size() == mi_->shard_paths().size(), "every shard mapped");
    for (const auto & kv : mi_->tensors()) {
        if (!declare_mapped(kv.first)) { err = "failed to map " + kv.first; return false; }
    }
    return true;
}

} // namespace qwfn
