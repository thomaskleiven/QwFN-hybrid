#!/usr/bin/env python3
"""Per-run parity statistics against a llama.cpp reference, including KL from top-k dumps.

    parity_stats.py REF_PREFIX RUN_LOG [RUN_LOG ...]

REF_PREFIX: llama-nll outputs without extension (REF_PREFIX.nll from --dump, REF_PREFIX.topk from
--topk). RUN_LOG: a qwfn-gen log with QWFN_PPL_VERBOSE lines; its top-k dump is expected next to
it as RUN_LOG with .log replaced by .topk (QWFN_TOPK_DUMP), or another llama-nll prefix given as
"llama:PREFIX" (for the llama.cpp noise floor, e.g. ubatch 1 vs 512).

Per run: tokens, mean NLL, shift vs reference with a paired bootstrap 95% CI over tokens, mean
|delta| per token, top-k KL(ref || run) (approximate: ids outside the run's top-k get its k-th
log-probability, an upper bound on the true term), and argmax agreement.
"""
import math, random, re, statistics as st, sys

NLL_RE = re.compile(r"\[nll\] #(\d+) tok (\d+): ([\d.eE+-]+) argmax (\d+)")


def load_nll(path):
    if path.startswith("llama:"):
        rows = [l.split() for l in open(path[6:] + ".nll") if l.strip()]
        return [(int(t), float(v), int(a)) for _, t, v, a in rows]
    out = {}
    for m in NLL_RE.finditer(open(path, errors="replace").read()):
        out[int(m.group(1)) - 1] = (int(m.group(2)), float(m.group(3)), int(m.group(4)))
    return [out[i] for i in sorted(out)]


def load_topk(path):
    try:
        f = open(path[6:] + ".topk" if path.startswith("llama:") else re.sub(r"\.log$", ".topk", path))
    except OSError:
        return None
    rows = {}
    for line in f:
        p = line.split()
        if p:
            rows[int(p[0])] = {int(a): float(b) for a, b in (x.split(":") for x in p[1:])}
    return rows


def kl_topk(ref, run):
    """sum over the reference's top-k ids of p (log p - log q); q floored at the run's k-th value."""
    tot = 0.0
    floor = min(run.values())
    for i, lp in ref.items():
        tot += math.exp(lp) * (lp - run.get(i, floor))
    return max(tot, 0.0)


def main():
    ref_prefix, runs = sys.argv[1], sys.argv[2:]
    ref = load_nll("llama:" + ref_prefix); ref_tk = load_topk("llama:" + ref_prefix)
    ref_mean = st.mean(v for _, v, _ in ref)
    print(f"reference {ref_prefix}: {len(ref)} tokens, NLL {ref_mean:.5f}")
    print(f"{'run':44s} {'n':>4s} {'NLL':>8s} {'shift':>9s} {'95% CI (tokens)':>21s} {'mean|d|':>8s} {'KL':>9s} {'argmax':>7s}")
    random.seed(0)
    for path in runs:
        rows = load_nll(path)
        n = min(len(rows), len(ref))
        if any(rows[i][0] != ref[i][0] for i in range(n)):
            sys.exit(f"{path}: token sequence differs from the reference")
        d = [rows[i][1] - ref[i][1] for i in range(n)]
        boots = sorted(st.mean(random.choices(d, k=n)) for _ in range(2000))
        tk = load_topk(path)
        kl = st.mean(kl_topk(ref_tk[i], tk[i]) for i in range(n) if i in tk and i in ref_tk) if tk and ref_tk else float("nan")
        name = path if len(path) <= 44 else "..." + path[-41:]
        print(f"{name:44s} {n:4d} {st.mean(r[1] for r in rows[:n]):8.5f} {st.mean(d):+9.5f} "
              f"[{boots[50]:+.5f}, {boots[1949]:+.5f}] {st.mean(abs(x) for x in d):8.5f} {kl:9.6f} "
              f"{100 * sum(rows[i][2] == ref[i][2] for i in range(n)) / n:6.1f}%")


if __name__ == "__main__":
    main()
