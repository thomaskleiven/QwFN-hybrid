# Design decisions

Baseline: QwFNfer 2d6837f. Target: RTX 5060 Ti 16 GB (PCIe Gen4 x8), Core Ultra 7 265F, 32 GB
DDR5, NVMe. Evidence lives in `bench/evidence/raw/` (final6 is the release build). Every change
below was measured on its own; rejected ones are listed with their numbers.

## Engine changes that stay

| Change | Why |
|---|---|
| Persistent ggml CPU threadpool. The caller's affinity is restored after creation. | Upstream created a thread pool on every CPU graph compute. |
| CPU experts at T ≥ 2 compute only the routed (expert, token) pairs | Upstream computed every CPU expert for every token and multiplied by a zero gate. The i-quant dot products are arithmetic-bound, so that doubled the work. |
| ggml `mul_mat_id` work stealing for 1-2 columns (`patches/`) | The 640-row gate/up got one static slice per thread, so the slowest core set the time. |
| Readback pack at every verify width, and host reduce | One device-to-host copy per layer instead of five. The reduce keeps the same F32 add order. |
| Late-fold row width (bug fix) | Rows aliased across tokens whenever the promotion budget was 4 or more. |
| MTP: 2 drafts, draft cost 0.4 (server default) | An extra position costs about 0.4 of a step on this machine; upstream's 0.7 kept it at 1 draft. |
| Session snapshots and shared-prefix checkpoints | An agent's side requests no longer force a full re-prefill. A new session restores its shared system/tools prefix (6.9K tokens in 0.02 s). |
| Deterministic expert placement | Promotion to VRAM is decided at lookup, so placement depends only on the token sequence, not on I/O timing. Before, the same prompt could give different text under load. Speed and NVMe reads unchanged (within 1%). It moves the reasoning parity shift (`docs/PARITY.md`). |
| Pipelined short-prompt (cbatch) prefill | Chunk c+1's NVMe reads overlap chunk c's uploads and matmuls. Tool-result prefill +39%. |
| Decode attention: q/k/v, gate, indexer and output projections computed once for all verify positions | One batched matmul per weight instead of one per position. GPU layer graphs −0.66 s per 800 tokens. |
| Persistent per-step graph contexts | No allocation of a fresh ggml context per CPU expert call. CPU MoE 7.9 → 7.1 s per 800 tokens. |
| Sparse attention gathers raw q8_0 cache bytes; flash attention reads them with its vector kernel | ~6× fewer bytes than dequantize → F16, and the same kernel llama.cpp runs over its cache. GPU layer graphs −10%, and the long-context NLL shift vs the aligned llama.cpp reference goes from −0.0035 (significant) to +0.0003. |
| Deferred speculative prefetch | With 3-token verify steps it wins: expert I/O wait −21% (code), −9% (agent). Output unchanged. |
| `--batch 32768` by default (server was 4096) | A long prompt streams every layer's experts once per batch. 21K tokens: 49.3 s at 16384 (two sweeps), 37.3 s at 32768 (one), same peak RAM, decode after it unchanged. |
| Prefill reads 2-4 deep instead of 32 | This drive streams 5.9 GB/s sequentially at depth 1 and ~2.4 GB/s at depth 16-32. Cold 4K-token prompt 16.4-19.7 → 11.3 s; warm 1.2K-token prompt 9.6 → 7.1 s. Decode keeps its 16 reader threads: at 4 its NVMe wait drops 30% but CPU experts slow 35% (net slower). |
| Optional Q4_K draft head (`bench/make_mtp_q4.sh`) | Draft head 2.28 → 1.83 s per 800 tokens at the same acceptance. Drafts are verified, so output is unchanged. |

## Configuration, not code

These speed up plain QwFN just as much. They are reported separately, never credited to the engine.

- RAM expert tier 9 → 19 GB (needs about 24 GB free).
- `--spec-block` off on this GPU.
- 5 CPU threads.

## Removed after measurement

| Removed | Reason |
|---|---|
| Split graph A (CPU experts overlapping the post-router GPU work) | Neutral on this machine, and adds a second graph per layer. |
| Measured-time draft policy | Inconclusive against the fixed 0.4. |
| 512-byte direct reads without the bounce buffer | Slower: 4K-unaligned requests. |
| Reader-thread affinity | Slower. |
| AVX2 Q2_0 kernel | GCC already vectorizes the generic kernel. |
| Promotion cap above 2, selective spec-block, `--spec-ahead 2`, deeper prefetch, KV in pinned RAM, OpenMP wait policies, a DRAM keeper thread | Neutral or worse. |
| ggml built without OpenMP (to use the pinned polling pool) | No gain. |
| `OMP_WAIT_POLICY=active` / `GOMP_SPINCOUNT` | Within noise on the replay (±1 tok/s from CPU heat); in the server the spinning threads starve the others (1-3 tok/s). |
| bf16 vector matmul kernel for 2-4 tokens (ggml dispatch) | GPU layer graphs 9.4 → 10.0 s. |
| A 4th draft (`MTP_MAX_DRAFTS` 4) | Accepted on 93% of code steps (4.84 tokens/step), yet slower (28.5-28.8 vs 29.2-29.5 tok/s): CPU expert work scales with tokens, not steps, so a longer step only amortizes the GPU part. |
| E-cores in the CPU expert pool (8 or 10 threads) | 28.3-29.0 vs 28.9-29.8 tok/s with 5 P-cores. |
| Q4_K draft head with 3 drafts as a recommended setting | final4: +6% on code (26.7 vs 25.2 tok/s), equal or slower elsewhere (agent 19.4 vs 20.1), equal in the server, and it fails the parity gate on agent (mean \|Δ\| 0.0116 vs ≤ 0.0104, KL 0.00147 vs ≤ 0.00109). The shipped default is the Q8_0 head with 2 drafts. |
| Server draft cap 3 | 3 drafts help only the code replay; server 20.7 vs 22.2 tok/s at cap 2. |
| LFU / hybrid expert eviction, 32 I/O threads, RAM tier 20 GB, prefetch depth 12 | Neutral or worse. |
| Prefill: upload RAM-tier experts straight from the pinned tier; read over short held runs | No gain (9.6-9.8 s vs 9.6 s for a warm 1.2K-token prompt); reading over held runs is slower (17.5 s). |
| Token-grouped MoE kernel (a ggml-cuda `mmvq` patch), not built | nsys over a code replay: the VRAM-MoE kernels are 8.0% of decode wall time and the GPU is busy 29% of it. Grouping duplicate expert reads could save 0.7-1.8% end to end, below a 3% bar, for a patch to maintain against upstream. Decode is bound by the CPU experts and NVMe reads, not by GPU kernels. |

## Prompt reading is bounded by the drive

A warm mid-size prompt (a tool result) reads every expert not in the RAM tier, ~25 GB. The
reader thread is busy for all of it at ~2.5-2.9 GB/s. `scripts/nvme_ceiling.py` shows the same:
this drive (PM9C1a) starts at 4-5 GB/s and settles at ~2.3-2.4 GB/s once reads span the whole
76 GB file, at normal temperature. The collapse is consistent with a DRAM-less controller's mapping
cache; that cause is not confirmed. Reading at depth 2-4 instead of 32 keeps it
near the sequential rate (see the table above).
