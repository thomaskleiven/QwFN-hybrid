#include "qwfn_io.h"
#include "qwfn_check.h"

#include <liburing.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace qwfn {

static uint64_t g_dio_align = 512;
uint64_t dio_align() { return g_dio_align; }
void     set_dio_align(uint64_t a) {
    g_dio_align = a == QWFN_DIO_PAGE ? QWFN_DIO_PAGE : 512;
    assert_that(g_dio_align == 512 || g_dio_align == QWFN_DIO_PAGE, "dio alignment is 512 or a page");
    assert_that((g_dio_align & (g_dio_align - 1)) == 0, "dio alignment is a power of two");
}

void * dio_alloc(size_t bytes) {
    void * p = nullptr;
    const size_t sz = dio_align_up(bytes);
    assert_that(sz >= bytes, "aligned size covers the request");
    if (posix_memalign(&p, 4096, sz) != 0) return nullptr;
    assert_that(((uintptr_t) p & 4095) == 0, "dio buffer is page-aligned");
    return p;
}

void dio_free(void * p) { free(p); }

uint64_t mem_available_bytes() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    assert_that(f != nullptr, "meminfo stream open");
    char line[256];
    uint64_t kb = 0;
    assert_that(sizeof(line) > 64, "meminfo line buffer holds a MemAvailable line");
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %lu kB", &kb) == 1) break;
    }
    fclose(f);
    return kb * 1024ull;
}

size_t clamp_to_available(size_t want, double frac, size_t headroom) {
    const uint64_t avail = mem_available_bytes();
    if (avail == 0) return want;                       // unknown: trust the caller
    const uint64_t budget = (uint64_t) ((double) avail * frac);
    const uint64_t safe   = budget > headroom ? budget - headroom : 0;
    assert_that(safe <= budget, "headroom never adds memory");
    if (safe == 0) return 0;
    if ((uint64_t) want <= safe) return want;
    assert_that(safe < (uint64_t) want, "clamping only ever shrinks the request");
    fprintf(stderr,
            "[qwfn] requested %.1f GB RAM tier but only %.1f GB is available; "
            "clamping to %.1f GB (%.0f%% of MemAvailable minus %.1f GB headroom)\n",
            want / 1e9, avail / 1e9, safe / 1e9, frac * 100, headroom / 1e9);
    return (size_t) safe;
}

io_engine::~io_engine() { shutdown(); }

bool io_engine::init(const std::vector<std::string> & paths, unsigned queue_depth,
                     bool direct_io, std::string & err, backend be) {
    shutdown();
    assert_that(fds_.empty() && ring_ == nullptr, "init starts from a shut-down engine");
    assert_that(workers_.empty() && in_flight_ == 0, "init starts with no workers or reads");
    direct_ = direct_io;
    be_     = be;
    qd_     = queue_depth ? queue_depth : 256;

    for (const auto & p : paths) {
        int flags = O_RDONLY;
        if (direct_) flags |= O_DIRECT;
        int fd = ::open(p.c_str(), flags);
        if (fd < 0 && direct_) {
            // Some filesystems refuse O_DIRECT; fall back rather than fail.
            fd = ::open(p.c_str(), O_RDONLY);
            if (fd >= 0) direct_ = false;
        }
        if (fd < 0) {
            err = "open failed for " + p + ": " + strerror(errno);
            shutdown();
            return false;
        }
        fds_.push_back(fd);
    }

    // With a 512-byte layout a direct read of the exact window would be served
    // buffered on a 4096-sector filesystem: the workers read a page-aligned
    // window into their own buffer and copy the payload into the slot instead.
    bounce_ = direct_ && dio_align() < QWFN_DIO_PAGE;
    if (be_ == backend::threads) {
        // pread is positional and thread-safe, so the shard fds are shared.
        const unsigned n = qd_ ? std::min(qd_, 32u) : 8u;
        stop_ = false;
        for (unsigned i = 0; i < n; i++) workers_.emplace_back([this] { worker_loop(); });
        return true;
    }
    return init_uring(err);
}

bool io_engine::init_uring(std::string & err) {
    assert_that(ring_ == nullptr && workers_.empty(), "no backend running yet");
    assert_that(qd_ > 0, "queue depth set");
    ring_ = (io_uring *) calloc(1, sizeof(io_uring));
    if (!ring_) { err = "out of memory allocating io_uring"; shutdown(); return false; }

    int rc = io_uring_queue_init(qd_, ring_, 0);
    if (rc < 0) {
        free(ring_);
        ring_ = nullptr;
        err = std::string("io_uring_queue_init failed: ") + strerror(-rc);
        shutdown();
        return false;
    }

    // Registering the fds removes a per-op file table lookup. If it fails we must
    // fall back to real fds: submitting with IOSQE_FIXED_FILE against an
    // unregistered table makes every read fail with -EBADF, and the destination
    // buffer then keeps whatever malloc left there.
    const int rr = io_uring_register_files(ring_, fds_.data(), (unsigned) fds_.size());
    registered_files = rr == 0;
    if (!registered_files) {
        fprintf(stderr, "[qwfn] io_uring_register_files failed (%s); using plain fds\n", strerror(-rr));
    }
    return true;
}

void io_engine::shutdown() {
    if (!workers_.empty()) {
        { std::lock_guard<std::mutex> lk(mtx_); stop_ = true; }
        cv_work_.notify_all();
        for (auto & t : workers_) if (t.joinable()) t.join();
        workers_.clear();
        q_.clear(); done_.clear();
        stop_ = false;
    }
    if (ring_) {
        io_uring_queue_exit(ring_);
        free(ring_);
        ring_ = nullptr;
    }
    for (int fd : fds_) if (fd >= 0) ::close(fd);
    fds_.clear();
    rejected_.clear();
    in_flight_ = 0;
    ctx_.clear(); ctx_free_.clear();
    assert_that(workers_.empty() && ring_ == nullptr, "shutdown stops every backend");
    assert_that(fds_.empty() && in_flight_ == 0, "shutdown closes every shard");
}

bool io_engine::bad_request(const io_request & r) const {
    return r.shard < 0 || (size_t) r.shard >= fds_.size() || !r.dst || r.nbytes == 0;
}

size_t io_engine::submit(const io_request * reqs, size_t n) {
    assert_that(n == 0 || reqs != nullptr, "submit has requests to read");
    assert_that(be_ == backend::threads || be_ == backend::uring, "known I/O backend");
    if (be_ == backend::threads) return submit_threads(reqs, n);
    if (!ring_) return 0;
    return submit_uring(reqs, n);
}

size_t io_engine::submit_threads(const io_request * reqs, size_t n) {
    assert_that(n == 0 || reqs != nullptr, "submit has requests to read");
    const auto t0 = std::chrono::steady_clock::now();
    size_t queued_jobs = 0;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (size_t i = 0; i < n; i++) {
            const io_request & r = reqs[i];
            if (bad_request(r)) {
                // Completed here as an error rather than dropped: see rejected_.
                // Returning n while quietly queueing fewer jobs is what used to
                // leave fetch_end() waiting on a completion nothing would produce.
                stat_errors++;
                // Not in_flight_++: here only a worker decrements it, so a job that
                // is already done would count as in flight for ever.
                done_.push_back(r.tag | IO_TAG_FAILED);
                continue;
            }
            uint64_t off = r.offset;
            uint32_t len = r.nbytes;
            if (direct_) { off = dio_align_down(r.offset); len = dio_padded_size(r.offset, r.nbytes); }
            q_.push_back(job{ r.shard, off, len, r.dst, r.tag, r.offset, r.nbytes });
            in_flight_++;
            queued_jobs++;
        }
    }
    assert_that(queued_jobs <= n, "no more jobs queued than requested");
    // notify_all(), not one notify_one() per job. Waking exactly as many
    // workers as there are jobs looks cheaper, but it was measured slower:
    // k notify_one() calls are k futex syscalls made serially by the
    // submitting thread, and the workers they wake start one after another,
    // while a single notify_all() starts them together. On an i9-12900H /
    // RTX 3080 Ti Laptop, UD-Q4_K_XL, 128-token decode, the per-job version
    // raised the per-token read wait from 63 to 68 ms (7.92 -> 7.75 tok/s).
    if (queued_jobs) cv_work_.notify_all();
    if (queued_jobs < n) cv_done_.notify_all();   // the rejected ones are already in done_
    stat_t_prep += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return n;
}

void io_engine::prep_sqe(io_uring_sqe * sqe, const io_request & r) {
    assert_that(sqe != nullptr, "an SQE to prepare");
    assert_that(r.shard >= 0 && (size_t) r.shard < fds_.size(), "request shard is valid");
    uint64_t off = r.offset;
    uint32_t len = r.nbytes;
    if (direct_) {
        off = dio_align_down(r.offset);
        len = dio_padded_size(r.offset, r.nbytes);
    }

    io_uring_prep_read(sqe, registered_files ? r.shard : fds_[r.shard], r.dst, len, off);
    if (registered_files) sqe->flags |= IOSQE_FIXED_FILE;
    // A padded window may run past the end of a shard; only the payload must land.
    const uring_ctx x{ r.tag, direct_ ? dio_pad(r.offset) + r.nbytes : r.nbytes };
    uint32_t ci;
    if (ctx_free_.empty()) { ci = (uint32_t) ctx_.size(); ctx_.push_back(x); }
    else                   { ci = ctx_free_.back(); ctx_free_.pop_back(); ctx_[ci] = x; }
    io_uring_sqe_set_data64(sqe, ci);
}

size_t io_engine::submit_uring(const io_request * reqs, size_t n) {
    assert_that(ring_ != nullptr, "io_uring ring initialised");
    assert_that(n == 0 || reqs != nullptr, "submit has requests to read");
    // `prepped` counts SQEs; `accepted` counts requests this call takes
    // responsibility for completing, which includes the rejected ones.
    size_t prepped = 0, accepted = 0;
    const auto t_prep0 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < n; i++) {
        const io_request & r = reqs[i];
        if (bad_request(r)) {
            // Completed here as an error rather than dropped: see rejected_.
            // Skipping it used to shift every later request's position in the
            // caller's retry loop, which then resubmitted the same bad request
            // for ever.
            stat_errors++;
            rejected_.push_back(r.tag | IO_TAG_FAILED);
            in_flight_++;
            accepted++;
            continue;
        }

        io_uring_sqe * sqe = io_uring_get_sqe(ring_);
        if (!sqe) break;   // ring full; caller should reap and retry
        prep_sqe(sqe, r);
        prepped++;
        accepted++;
    }

    const auto t_prep1 = std::chrono::steady_clock::now();
    stat_t_prep += std::chrono::duration<double>(t_prep1 - t_prep0).count();

    if (prepped) {
        const int rc = io_uring_submit(ring_);
        stat_t_submit_syscall += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_prep1).count();
        // A short or failed submit does NOT lose the entries: liburing has
        // already advanced the SQ tail, so whatever the kernel did not take
        // stays in the ring and goes out on the next io_uring_submit(). They
        // will therefore all complete, and all of them count as in flight --
        // what used to hang the decode loop was that nothing called submit()
        // again before the wait, so reap() flushes the ring before waiting.
        if (rc < 0 || (size_t) rc < prepped) stat_errors++;
        in_flight_ += prepped;
    }
    return accepted;
}

size_t io_engine::reap(uint64_t * tags_out, size_t max_tags, size_t min_complete) {
    assert_that(max_tags == 0 || tags_out != nullptr, "reap has somewhere to put tags");
    assert_that(be_ == backend::threads || be_ == backend::uring, "known I/O backend");
    if (be_ == backend::threads) return reap_threads(tags_out, max_tags, min_complete);
    if (!ring_ || in_flight_ == 0) return 0;
    return reap_uring(tags_out, max_tags, min_complete);
}

size_t io_engine::reap_threads(uint64_t * tags_out, size_t max_tags, size_t min_complete) {
    assert_that(max_tags == 0 || tags_out != nullptr, "reap has somewhere to put tags");
    size_t got = 0;
    std::unique_lock<std::mutex> lk(mtx_);
    while (got < max_tags) {
        if (done_.empty()) {
            if (got >= min_complete) break;
            // Nothing in flight and nothing done: a caller whose count has
            // drifted would wait here forever. Return short instead; the
            // caller reports a failed read, which beats a silent hang.
            if (in_flight_ == 0) break;
            cv_done_.wait(lk, [this] { return !done_.empty() || in_flight_ == 0; });
            if (done_.empty()) break;
        }
        tags_out[got++] = done_.front();
        done_.pop_front();
    }
    assert_that(got <= max_tags, "reap stays within the caller's tag buffer");
    return got;
}

size_t io_engine::reap_uring(uint64_t * tags_out, size_t max_tags, size_t min_complete) {
    assert_that(ring_ != nullptr && in_flight_ > 0, "reaping a live ring with reads in flight");
    assert_that(max_tags == 0 || tags_out != nullptr, "reap has somewhere to put tags");
    size_t got = 0;
    if (min_complete > in_flight_) min_complete = in_flight_;

    // Requests rejected at submit: complete in the only sense the caller tracks.
    while (got < max_tags && !rejected_.empty()) {
        tags_out[got++] = rejected_.back();
        rejected_.pop_back();
        in_flight_--;
    }
    if (in_flight_ == 0) return got;

    // io_uring_submit() takes as many SQEs as the kernel will accept and leaves
    // the rest in the ring for the next submit. Nothing else calls submit()
    // between the caller's last one and the wait below, so a short submit would
    // park entries the kernel never saw and io_uring_wait_cqe() would block on
    // completions that cannot arrive. Flush first.
    if (io_uring_sq_ready(ring_) > 0) {
        const int rc = io_uring_submit(ring_);
        if (rc <= 0 && io_uring_sq_ready(ring_) > 0) {
            // The ring will not drain. Hand back what is already there rather
            // than waiting for ever; the caller reports a short read, which is
            // recoverable, where a hang is not.
            stat_errors++;
            min_complete = 0;
        }
    }
    return reap_cqes(tags_out, max_tags, min_complete, got);
}

size_t io_engine::reap_cqes(uint64_t * tags_out, size_t max_tags, size_t min_complete, size_t got) {
    assert_that(ring_ != nullptr, "io_uring ring initialised");
    assert_that(got <= max_tags, "tags already reaped fit the buffer");
    int interrupted = 0;
    while (got < max_tags) {
        io_uring_cqe * cqe = nullptr;
        int rc;
        if (got < min_complete) {
            rc = io_uring_wait_cqe(ring_, &cqe);
            // A signal interrupting the wait is not a failed read: wait again (bounded), or the
            // caller would count the reads still in the kernel as lost.
            if (rc == -EINTR && interrupted++ < 64) continue;
        } else {
            rc = io_uring_peek_cqe(ring_, &cqe);
            if (rc == -EAGAIN || !cqe) break;
        }
        if (rc < 0) { stat_errors++; break; }

        const uint32_t ci = (uint32_t) io_uring_cqe_get_data64(cqe);   // set only by submit()
        const uring_ctx x = ctx_[ci];
        ctx_free_.push_back(ci);
        const bool failed = cqe->res < 0 || (uint32_t) cqe->res < x.need;
        if (failed) {
            stat_errors++;
            if (cqe->res >= 0) stat_short++;
        } else {
            stat_reads++;
            stat_bytes += (uint64_t) cqe->res;
        }
        tags_out[got++] = failed ? (x.tag | IO_TAG_FAILED) : x.tag;
        io_uring_cqe_seen(ring_, cqe);
        in_flight_--;
        if (in_flight_ == 0) break;
    }
    return got;
}


void io_engine::worker_loop() {
    assert_that(be_ == backend::threads, "workers run only on the thread backend");
    // rule 2 deviation: I/O worker service loop, runs until stop_ (its only wait honors it), see docs/CODING_RULES.md
    for (;;) {
        job j;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_work_.wait(lk, [this] { return stop_ || !q_.empty(); });
            if (stop_ && q_.empty()) return;
            j = q_.front();
            q_.pop_front();
        }
        assert_that(j.shard >= 0 && (size_t) j.shard < fds_.size() && j.dst, "queued job is valid");
        const ssize_t got = bounce_ ? read_bounced(j) : read_direct(j);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            // As on io_uring: only the payload must land, not the padded window.
            const bool failed = got < (ssize_t) (direct_ ? dio_pad(j.ooff) + j.onb : j.onb);
            if (failed) { stat_errors++; if (got > 0) stat_short++; }
            else { stat_reads++; stat_bytes += (uint64_t) got; }
            done_.push_back(failed ? (j.tag | IO_TAG_FAILED) : j.tag);
            in_flight_--;
        }
        cv_done_.notify_all();
    }
}

ssize_t io_engine::read_direct(const job & j) {
    assert_that(j.shard >= 0 && (size_t) j.shard < fds_.size(), "job shard is valid");
    assert_that(j.dst != nullptr, "job has a destination");
    ssize_t got = 0;
    while (got < (ssize_t) j.len) {
        const ssize_t r = ::pread(fds_[j.shard], (char *) j.dst + got,
                                  j.len - got, (off_t) (j.off + got));
        if (r <= 0) break;
        got += r;
    }
    return got;
}

// The page-aligned window around the requested range, into this worker's
// buffer; the payload then goes where the 512-byte layout expects it. A window
// past the end of a shard reads short, which is fine as long as the payload arrived.
ssize_t io_engine::read_bounced(const job & j) {
    assert_that(j.shard >= 0 && (size_t) j.shard < fds_.size(), "job shard is valid");
    assert_that(j.dst != nullptr, "job has a destination");
    ssize_t got = 0;
    static thread_local uint8_t * scratch = nullptr;
    static thread_local size_t    scratch_bytes = 0;
    const uint64_t w0 = j.ooff & ~(QWFN_DIO_PAGE - 1);
    const uint64_t w1 = (j.ooff + j.onb + QWFN_DIO_PAGE - 1) & ~(QWFN_DIO_PAGE - 1);
    const size_t   wl = (size_t) (w1 - w0);
    if (scratch_bytes < wl) {
        if (scratch) dio_free(scratch);
        scratch_bytes = wl + (1u << 20);
        scratch = (uint8_t *) dio_alloc(scratch_bytes);
    }
    const ssize_t need = (ssize_t) (j.ooff - w0 + j.onb);
    if (scratch) {
        while (got < (ssize_t) wl) {
            const ssize_t r = ::pread(fds_[j.shard], scratch + got, wl - got, (off_t) (w0 + got));
            if (r <= 0) break;
            got += r;
        }
    }
    if (got >= need) {
        memcpy((char *) j.dst + dio_pad(j.ooff), scratch + (j.ooff - w0), j.onb);
        got = (ssize_t) j.len;   // the caller's notion of a complete read
    } else {
        got = 0;
    }
    return got;
}

} // namespace qwfn
