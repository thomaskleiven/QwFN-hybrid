# Parity with llama.cpp

## Method

Every engine is scored teacher-forced on identical tokens: the same model (Qwen3.8-Flash-Next
GSQ-RCO IQ3_XXS) and MTP head, prompt `bench/tokens/<scenario>.prompt`, continuation
`bench/tokens/<scenario>.replay` (generated once, greedily, by the original QwFN). At every position
we record the target token's NLL (full softmax over 248,320 tokens), the argmax and the top-32
log-probabilities.

**Reference:** unmodified llama.cpp (unslothai/llama.cpp ca1426903, `tools/llama_nll.cpp`) with a
q8_0 KV cache and flash attention, scoring one token per step (`--ubatch 1`), as every engine here
decodes. Files: `bench/evidence/raw/parity0/llama.*.u1.*`. The reference settings matter: q8_0 vs
f16 KV moves llama.cpp's own NLL by −0.0024 on code, ubatch 1 vs 512 by +0.0026.

| Scenario | Prompt | Tokens scored |
|---|---|---|
| code | Python LRU cache with tests, thinking off | 800 |
| agent | C++ header review and refactor | 800 |
| short | three-sentence explanation, thinking on | 300 |
| reasoning | constrained optimisation word problem, thinking on | 600 |
| longctx | 21K-token source file plus a grounded question | 300 |

Metrics against the reference: **shift** (mean NLL difference, token-bootstrap 95% CI), **mean |Δ|**
(mean absolute per-token difference), **KL** (top-k KL(llama.cpp ‖ engine)) and **argmax agreement**.

## Gate (v5, registered before its runs)

The yardstick is an independent engine for the same model: QwFNfer, the codebase this fork started
from, in three configurations (A, B1, B2). The gate is one-sided: it fails the hybrid only where it
is further from llama.cpp than normal. Per scenario:

| Metric | Passes when |
|---|---|
| shift | \|shift\| ≤ max(yardstick, llama.cpp null spread), or the 95% CI contains 0 |
| mean \|Δ\| | ≤ 1.10 × yardstick max |
| top-k KL | ≤ 1.25 × yardstick max |
| argmax | ≥ yardstick min − 0.5 points |

**Null spread:** llama.cpp scoring the same tokens 2, 3 and 4 per decode call
(`bench/evidence/null_runs.sh`, data in `null5`). Only the floating-point order changes, so this is
how far a correct implementation drifts by rounding alone. `bench/evidence/summarize.py RAW_DIR
--null NULL_DIR` computes the gate and exits non-zero on failure.

## Result (final6, the release build)

Four of five scenarios pass. **Reasoning fails on shift only** (`bench/evidence/raw/final6/summary_v5.md`):

| Scenario | \|shift\| (limit) | mean \|Δ\| (limit) | KL (limit) | argmax (limit) | |
|---|---|---|---|---|---|
| code | 0.0003 (0.0003) | 0.0022 (0.0027) | 0.00017 (0.00025) | 99.9% (≥ 99.1%) | pass |
| agent | 0.0005 (0.0011) | 0.0089 (0.0104) | 0.00089 (0.00109) | 99.5% (≥ 98.8%) | pass |
| short | 0.0017 (0.0039) | 0.0131 (0.0163) | 0.00185 (0.00292) | 99.7% (≥ 98.4%) | pass |
| longctx | 0.0010, CI covers 0 | 0.0129 (0.0145) | 0.00115 (0.00140) | 98.7% (≥ 98.2%) | pass |
| reasoning | **0.0031 (0.0021)** | 0.0192 (0.0206) | 0.00137 (0.00174) | 98.5% (≥ 97.9%) | **fail** |

Not re-scored; the failure stands. What we found:

- **Cause: where experts run, not a code error** (`raw/debate1`). The build before deterministic
  expert placement reproduces the previous release's reasoning NLLs byte for byte, and with no VRAM
  expert tier (`--vram 0`) the builds before and after it are byte-identical. Placement only decides
  which experts are summed on the GPU and which on the CPU. Changing only that path moves the
  reasoning shift across −0.0007 (1 draft), −0.0015 (`--vram 11`), −0.0017 (all experts on CPU),
  −0.0019 (no MTP) and −0.0031 (the gated configuration).
- **600 tokens is too short to separate a bias from noise.** A check registered before it ran
  scored the reasoning prompt continued to 3000 tokens (`raw/longreason`, `bench/tokens/reasoning3k.*`):

  | Engine | shift | 95% CI |
  |---|---|---|
  | llama.cpp, 4 per step | −0.0004 | [−0.0014, +0.0006] |
  | QwFN (yardstick A) | −0.0002 | [−0.0012, +0.0009] |
  | **QwFN-hybrid** | **−0.0014** | [−0.0024, −0.0004] |

  Within the 0.0021 limit, so noise by the registered rule. But the hybrid's CI excludes 0 and the
  others' do not: a small lean of about −0.001 is real at this length. It is below what the gate
  tolerates and is not yet attributed.

## Earlier findings

- **Attention gather** (parity1): dequantizing the chosen q8_0 cache cells to F16 shifted long
  context by −0.0035. The engine now reads raw q8_0 with llama.cpp's own vector kernel (+0.0003).
- **No separate parity mode:** switching off fusions, computing every expert on the CPU or disabling
  MTP does not bring the engine significantly closer to llama.cpp.
- **Gate history:** v1 (per run) and v2 (two-sided) were replaced because they failed runs for
  noise or for being *closer* to llama.cpp; v3 scored against a 512-token-batch reference, which
  uses a different attention kernel than decoding; v4 (one-token reference) passed the previous
  release (`raw/final4`) and failed final6's build on reasoning; v5 added the null spread.
  Each gate was fixed before the campaign it judged.

## What this shows

- **Shows:** under teacher forcing, the engine's per-token distributions are as close to llama.cpp's
  as an independent implementation's, except a small reasoning lean (above).
- **Does not show:** bit identity with llama.cpp, or identical free-running text. The engine itself
  is deterministic run to run (`bench/golden.sh`). Task success is measured separately (OpenCode runs).
