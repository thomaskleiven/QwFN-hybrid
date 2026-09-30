# Coding rules

The C++ sources in `src/` and `tools/` follow the ten rules below, strictly. The only exceptions are where a
third-party API (ggml, cpp-httplib, nlohmann/json, liburing, POSIX) forces a construct, or where the rule itself cannot apply (the assertion primitive). Each exception
is listed below and marked at the site with a `// rule N deviation:` comment.

**Enforcement:** `scripts/check_rules.sh` builds every target with `-Werror`, then runs
`scripts/check_rules.py`, which needs `pip install lizard`. It fails on any unmarked violation. The tree
passes it.

| # | Rule | How it is enforced here |
|---|---|---|
| 1 | No goto, setjmp/longjmp or recursion | Keyword scan, plus a self-call check per function. Overloads that forward to each other were renamed or merged, so no function calls its own name |
| 2 | Every loop has a fixed upper bound | Loops over data are bounded by their container. Parser and drain loops are capped by input size. Waits either have timeouts (e.g. the prefill stage waits at most 120 s for its reader) or wait on work already in flight |
| 3 | No dynamic allocation after initialization | Decode-step scratch (expert handles, id/weight tables, prefetch sets, positions, residency tables, draft-head buffers) is sized once in `engine::init` and reused. See the deviations for ggml contexts and the server |
| 4 | Functions fit on one page (≤ 60 lines) | `check_rules.py`, using lizard's function boundaries; lambdas count |
| 5 | At least two assertions per function | `qwfn::assert_that` (`src/qwfn_check.h`). It stays on in release builds, and a failed assertion aborts with the file, line and function. Trivial accessors (≤ 5 lines, no branch or loop) are exempt. Assertions check programmer invariants, never external input |
| 6 | Data declared at the smallest scope | No mutable globals. Per-server state lives in `server`, per-step state in step structs |
| 7 | Check every return value and parameter | Numeric input goes through `parse_int` / `parse_float`, `arg_int` / `arg_float` (CLI: range-checked, exit 2 with the flag name) and `env_int` / `env_float` (environment). Server request fields are range-checked: a bad value gets a 400, not an abort. Allocations and file operations are checked |
| 8 | Preprocessor limited to includes and simple macros | No function-like macros. The assertion is a function, using `std::source_location` |
| 9 | Restricted pointers: no double indirection, no function pointers | The graph builder returns structs instead of `T **` out-parameters. Engine progress is an atomic snapshot instead of a callback. Server output goes to writer structs instead of `std::function` sinks |
| 10 | All warnings on and fixed | `-Wall -Wextra -Wpedantic -Wshadow` on every target, with zero warnings. `-DQWFN_WERROR=ON` makes them errors. Third-party headers are system includes |

## Deviations

| Rule | Where | Why | Mitigation |
|---|---|---|---|
| 2 | I/O worker threads (`io_engine::worker_loop`), the prefill reader thread, the server's stall watchdog, the interactive chat loop | Service loops that run for the life of the process by design | They exit on their stop flag, on `/quit`, or on end of input. Their waits honor the stop flag |
| 3 | ggml contexts per step: decode input context (non-QSA fallback), the draft head's per-draft contexts, the temporary graph context headers | ggml builds each graph in a context it allocates. The decode graphs themselves are cached and replayed, and the temporary contexts reuse one persistent buffer, so what remains is ggml's small context header | Allocation results are asserted, and the memory is bounded and freed per step |
| 3 | Server requests | cpp-httplib and nlohmann/json allocate per request | Request fields are validated. Session snapshots are capped (`QWFN_SNAP_MB`) |
| 5 | `qwfn::assert_that` | The assertion primitive cannot assert on itself | — |
| 5 | `engine::~engine` | A destructor also runs after a failed `init`, so it must not abort | It frees only what was allocated |
| 5 | `cubic()` in the vision resize | It runs about 60 times per output pixel | `resize_bicubic` asserts the tap range once per resize |
| 9 | `ggml_backend_reg_get_proc_address` function pointers, `main(argc, argv)`, `std::thread` entry points, cpp-httplib route handlers and content providers, POSIX signal handlers | That is the shape of the third-party or OS API | Null-checked where the API can return null |

## Verification that the rules changed nothing

Every phase of this refactor was checked with `bench/golden.sh`. Replays are deterministic, so each phase
had to reproduce the previous build byte for byte, and every phase had a separate senior review before
it was committed.

The regression check compares:
- per-token NLL and top-k log-probabilities on code, agent, reasoning, short chat and long context;
- a replay without MTP;
- a replay with every expert on the CPU;
- a warm second prompt's logits;
- io_uring reads, the speculative block, and a CPU-only engine.

Found and fixed along the way:
- two infinite loops: the server's tool-call parser, and the speculative read submit;
- an unbounded VRAM probe;
- unchecked fixed-size arrays in the expert fetch;
- an out-of-bounds CPU mask write;
- an empty-prompt server crash;
- `--multi-test` stats written out of bounds;
- NaN sampling weights from `repeat_penalty` 0;
- leaks on engine error paths.
