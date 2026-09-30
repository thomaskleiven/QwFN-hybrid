#!/usr/bin/env python3
"""Summarize an evidence campaign directory: speed statistics and teacher-forced parity
against llama.cpp, with a regression gate.

    summarize.py RAW_DIR [--json out.json] [--md out.md] [--null NULL_DIR]

Inputs (written by campaign_replay.sh):
  llama.<scenario>.r<k>.nll            per-token dump of llama-nll: "i token nll argmax"
  <cfg>.<scenario>.r<k>.log            qwfn-gen replay log with QWFN_PPL_VERBOSE lines
  <cfg>.<scenario>.r<k>.gpu.csv        nvidia-smi samples (timestamp, util %, MiB, W)

Configurations: A = QwFN as deployed for the aider runs, B1/B2 = QwFN tuned (1/2 drafts),
C = hybrid. A, B1 and B2 are the same original engine (2d6837f), so together they define how
far plain QwFN's numerics sit from llama.cpp; the hybrid is gated against that envelope.

D = hybrid with the Q4_K draft head at 3 drafts (the fastest configuration).

Gate v1 (per run) is reported. Gate v3 (docs/PARITY.md, registered before the campaign it judges)
decides the exit status; it is one-sided -- only "worse than plain QwFN" fails:
  per scenario and hybrid configuration, over its runs (QwFN = mean of each of A, B1, B2)
    |shift|    <= max over QwFN configs |shift|, or the run's token-bootstrap 95% CI contains 0
    mean |d|   <= 1.10 * max over QwFN configs
    top-k KL   <= 1.25 * max over QwFN configs          (when top-k dumps are present)
    argmax     >= min over QwFN configs - 0.5 percentage points
Gate v5 (--null NULL_DIR with llama.<scenario>.u<k>.nll): the shift limit becomes
    max(max over QwFN configs |shift|, max over k |shift of llama.cpp at ubatch k vs its u1 reference|)
"""
import glob, json, os, random, re, statistics as st, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity_stats import kl_topk, load_topk

NLL_RE = re.compile(r"\[nll\] #(\d+) tok (\d+): ([\d.eE+-]+) argmax (\d+)")
TOK_RE = re.compile(r"decode: \d+ tokens in [\d.]+ s\s+\(([\d.]+) tok/s\)")
PRE_RE = re.compile(r"prefill: (\d+) tokens in ([\d.]+) s")
RSS_RE = re.compile(r"maxrss (\d+) KB")
CFG_NAMES = {"A": "QwFN, aider flags", "B1": "QwFN tuned, 1 draft", "B2": "QwFN tuned, 2 drafts", "C": "hybrid",
             "D": "hybrid, Q4K draft head, 3 drafts"}
QWFN = ("A", "B1", "B2")
HYB = ("C", "D")
CFGS = QWFN + HYB


def llama_dump(path):
    out = []
    for line in open(path):
        p = line.split()
        if len(p) == 4:
            out.append((int(p[1]), float(p[2]), int(p[3])))
    return out


def engine_log(path):
    txt = open(path, errors="replace").read()
    tk = load_topk(path)
    toks = {}
    for m in NLL_RE.finditer(txt):
        toks[int(m.group(1)) - 1] = (int(m.group(2)), float(m.group(3)), int(m.group(4)))
    rows = [toks[i] for i in sorted(toks)]
    tps = TOK_RE.search(txt); pre = PRE_RE.search(txt); rss = RSS_RE.search(txt)
    return {"rows": rows, "topk": tk,
            "tok_s": float(tps.group(1)) if tps else None,
            "prefill_tok_s": int(pre.group(1)) / float(pre.group(2)) if pre and float(pre.group(2)) > 0 else None,
            "maxrss_gb": int(rss.group(1)) / 1e6 if rss else None}


def gpu_stats(path):
    util, mem = [], []
    if os.path.exists(path):
        for line in open(path):
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                try:
                    util.append(float(p[1].split()[0])); mem.append(float(p[2].split()[0]))
                except ValueError:
                    pass
    return {"gpu_util_mean": st.mean(util) if util else None, "vram_max_mib": max(mem) if mem else None}


def compare(rows, ref, tk=None, ref_tk=None):
    n = min(len(rows), len(ref))
    if n == 0:
        return None
    if any(rows[i][0] != ref[i][0] for i in range(n)):
        raise SystemExit("token sequences differ between an engine log and the llama.cpp dump")
    d = [rows[i][1] - ref[i][1] for i in range(n)]
    ad = sorted(abs(x) for x in d)
    random.seed(0)
    boots = sorted(st.mean(random.choices(d, k=n)) for _ in range(2000))
    kls = [kl_topk(ref_tk[i], tk[i]) for i in range(n) if tk and ref_tk and i in tk and i in ref_tk]
    return {"n": n,
            "ci": (boots[50], boots[1949]),
            "kl": st.mean(kls) if kls else None,
            "nll": st.mean(r[1] for r in rows[:n]),
            "mean_abs_delta": st.mean(ad),
            "p99_abs_delta": ad[int(0.99 * (n - 1))],
            "max_abs_delta": ad[-1],
            "argmax_agree": sum(rows[i][2] == ref[i][2] for i in range(n)) / n}


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "mean": st.mean(xs), "min": min(xs), "max": max(xs), "std": st.stdev(xs) if len(xs) > 1 else 0.0}


def main():
    raw = sys.argv[1]
    args = sys.argv[2:]
    out_json = args[args.index("--json") + 1] if "--json" in args else None
    out_md = args[args.index("--md") + 1] if "--md" in args else None
    null_dir = args[args.index("--null") + 1] if "--null" in args else None
    gate = "v5" if null_dir else "v3"

    llama = defaultdict(dict); llama_tk = defaultdict(dict)
    for p in glob.glob(os.path.join(raw, "llama.*.r*.nll")):
        _, sc, r, _ = os.path.basename(p).split(".")
        llama[sc][r] = llama_dump(p)
        llama_tk[sc][r] = load_topk("llama:" + p[:-4])
    runs = defaultdict(lambda: defaultdict(dict))
    for p in glob.glob(os.path.join(raw, "*.*.r*.log")):
        b = os.path.basename(p)
        if b.startswith("llama."):
            continue
        cfg, sc, r, _ = b.split(".")
        e = engine_log(p); e.update(gpu_stats(p[:-4] + ".gpu.csv"))
        runs[sc][cfg][r] = e

    result = {"scenarios": {}, "gate": {"pass": True, "failures": []}, "gate_v3": {"pass": True, "failures": []}}
    for sc in sorted(runs):
        if "r1" not in llama.get(sc, {}):
            continue
        ref = llama[sc]["r1"]
        ref_nll = [st.mean(v for _, v, _ in llama[sc][k]) for k in sorted(llama[sc])]
        ref_mean = st.mean(ref_nll)
        s = {"llama_nll_runs": ref_nll, "tokens": len(ref)}
        if "r2" in llama[sc]:
            s["llama_vs_llama"] = compare(llama[sc]["r2"], ref)
        ref_tk = llama_tk[sc]["r1"]
        per_cfg = {}
        for cfg in sorted(runs[sc]):
            reps = runs[sc][cfg]
            par = {r: compare(e["rows"], ref, e["topk"], ref_tk) for r, e in reps.items() if e["rows"]}
            for r, c in par.items():
                if c:
                    c["signed_shift"] = c["nll"] - ref_mean
            per_cfg[cfg] = {
                "tok_s": stats([e["tok_s"] for e in reps.values()]),
                "prefill_tok_s": stats([e["prefill_tok_s"] for e in reps.values()]),
                "maxrss_gb": stats([e["maxrss_gb"] for e in reps.values()]),
                "gpu_util": stats([e["gpu_util_mean"] for e in reps.values()]),
                "vram_max_mib": stats([e["vram_max_mib"] for e in reps.values()]),
                "parity": par,
            }
        s["configs"] = per_cfg
        # envelope from the plain-QwFN runs
        q = [c for cfg in QWFN if cfg in per_cfg for c in per_cfg[cfg]["parity"].values() if c]
        if q:
            ll_shift = abs(ref_nll[0] - ref_nll[1]) if len(ref_nll) > 1 else 0.0
            env = {"shift": max(1.25 * max(abs(c["signed_shift"]) for c in q), ll_shift),
                   "mean_abs_delta": 1.25 * max(c["mean_abs_delta"] for c in q),
                   "argmax_agree": min(c["argmax_agree"] for c in q) - 0.005}
            s["envelope"] = env
            for r, c in per_cfg.get("C", {}).get("parity", {}).items():
                if not c:
                    continue
                bad = []
                if abs(c["signed_shift"]) > env["shift"]:
                    bad.append(f"shift {c['signed_shift']:+.4f} > {env['shift']:.4f}")
                if c["mean_abs_delta"] > env["mean_abs_delta"]:
                    bad.append(f"mean|d| {c['mean_abs_delta']:.4f} > {env['mean_abs_delta']:.4f}")
                if c["argmax_agree"] < env["argmax_agree"]:
                    bad.append(f"argmax {c['argmax_agree']:.3f} < {env['argmax_agree']:.3f}")
                if bad:
                    result["gate"]["pass"] = False
                    result["gate"]["failures"].append(f"{sc} {r}: " + "; ".join(bad))
        # Gate v3 (one-sided, docs/PARITY.md): a hybrid configuration fails a metric only when it
        # is worse than every plain-QwFN configuration by more than the stated margin.
        cfgs = [[c for c in per_cfg[k]["parity"].values() if c] for k in QWFN if k in per_cfg]
        cfgs = [x for x in cfgs if x]
        if cfgs:
            qm = lambda key: [st.mean(c[key] for c in x) for x in cfgs if all(c[key] is not None for c in x)]
            null = []
            for p in sorted(glob.glob(os.path.join(null_dir, f"llama.{sc}.u*.nll"))) if null_dir else []:
                c0 = compare(llama_dump(p), ref)
                null.append(c0["nll"] - st.mean(v for _, v, _ in ref[:c0["n"]]))
            s["null_shifts"] = null
            lim = {"shift": max([abs(v) for v in qm("signed_shift")] + [abs(v) for v in null]),
                   "mean_abs_delta": 1.10 * max(qm("mean_abs_delta")),
                   "kl": 1.25 * max(qm("kl")) if qm("kl") else None,
                   "argmax_agree": min(qm("argmax_agree")) - 0.005}
            s["limits_v3"] = lim
            s["gate_v3"] = {}
            for h in HYB:
                hyb = [c for c in per_cfg.get(h, {}).get("parity", {}).values() if c]
                if not hyb:
                    continue
                m = lambda key: st.mean(c[key] for c in hyb)
                chk = {"shift": abs(m("signed_shift")) <= lim["shift"] or all(c["ci"][0] <= 0 <= c["ci"][1] for c in hyb),
                       "mean_abs_delta": m("mean_abs_delta") <= lim["mean_abs_delta"],
                       "argmax_agree": m("argmax_agree") >= lim["argmax_agree"]}
                if lim["kl"] is not None and all(c["kl"] is not None for c in hyb):
                    chk["kl"] = m("kl") <= lim["kl"]
                s["gate_v3"][h] = chk
                for k, ok in chk.items():
                    if not ok:
                        result["gate_v3"]["pass"] = False
                        result["gate_v3"]["failures"].append(f"{sc} {h} {k}")
        result["scenarios"][sc] = s

    # markdown
    L = []
    L.append("## Decode speed (tok/s, teacher-forced replay)\n")
    L.append("| scenario | " + " | ".join(CFG_NAMES[c] for c in CFGS) + " |")
    L.append("|---|" + "---|" * len(CFGS))
    for sc, s in result["scenarios"].items():
        cells = []
        for c in CFGS:
            t = s["configs"].get(c, {}).get("tok_s")
            cells.append(f"{t['mean']:.2f} ({t['min']:.2f}-{t['max']:.2f}, n={t['n']})" if t else "-")
        L.append(f"| {sc} | " + " | ".join(cells) + " |")
    L.append("\n## Teacher-forced parity against llama.cpp\n")
    L.append("| scenario | tokens | engine | NLL/token | shift vs llama.cpp | mean abs delta | top-k KL | max abs delta | argmax agreement |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for sc, s in result["scenarios"].items():
        L.append(f"| {sc} | {s['tokens']} | llama.cpp (runs) | {', '.join(f'{x:.4f}' for x in s['llama_nll_runs'])} | - | - | - | - | - |")
        if s.get("llama_vs_llama"):
            c = s["llama_vs_llama"]
            L.append(f"| {sc} | {c['n']} | llama.cpp r2 vs r1 | {c['nll']:.4f} | - | {c['mean_abs_delta']:.4f} | - | {c['max_abs_delta']:.4f} | {100*c['argmax_agree']:.1f}% |")
        for cfg in CFGS:
            par = [c for c in s["configs"].get(cfg, {}).get("parity", {}).values() if c]
            if not par:
                continue
            f = lambda k, p=4: (f"{st.mean(c[k] for c in par):.{p}f} ({min(c[k] for c in par):.{p}f}..{max(c[k] for c in par):.{p}f})"
                                if all(c[k] is not None for c in par) else "-")
            L.append(f"| {sc} | {par[0]['n']} | {CFG_NAMES[cfg]} (n={len(par)}) | {f('nll')} | {f('signed_shift')} | {f('mean_abs_delta')} | "
                     f"{f('kl', 6)} | {f('max_abs_delta')} | {100*min(c['argmax_agree'] for c in par):.1f}-{100*max(c['argmax_agree'] for c in par):.1f}% |")
        if "envelope" in s:
            e = s["envelope"]
            L.append(f"| {sc} | | *envelope* | | abs <= {e['shift']:.4f} | <= {e['mean_abs_delta']:.4f} | | | >= {100*e['argmax_agree']:.1f}% |")
    L.append(f"\n**Parity gate v1 (per run, pre-registered): {'PASS' if result['gate']['pass'] else 'FAIL'}**")
    for f in result["gate"]["failures"]:
        L.append(f"- {f}")
    L.append(f"\n**Parity gate (one-sided, {gate} rules; registered before this campaign): {'PASS' if result['gate_v3']['pass'] else 'FAIL'}**")
    for sc, s in result["scenarios"].items():
        if "limits_v3" in s:
            lim = s["limits_v3"]
            for h, chk in s["gate_v3"].items():
                par = [c for c in s["configs"][h]["parity"].values() if c]
                m = lambda k: st.mean(c[k] for c in par)
                kl = f"KL {m('kl'):.6f} <= {lim['kl']:.6f}" if "kl" in chk else "KL -"
                L.append(f"- {sc} {h}: |shift| {abs(m('signed_shift')):.4f} <= {lim['shift']:.4f} or CI covers 0{' (null ' + ', '.join(f'{v:+.4f}' for v in s['null_shifts']) + ')' if s.get('null_shifts') else ''}; "
                         f"mean|d| {m('mean_abs_delta'):.4f} <= {lim['mean_abs_delta']:.4f}; {kl}; "
                         f"argmax {100*m('argmax_agree'):.1f}% >= {100*lim['argmax_agree']:.1f}% -> "
                         + ("ok" if all(chk.values()) else "FAIL: " + ", ".join(k for k, v in chk.items() if not v)))
    md = "\n".join(L)
    print(md)
    if out_md:
        open(out_md, "w").write(md + "\n")
    if out_json:
        json.dump(result, open(out_json, "w"), indent=1, default=float)
    # Exit status follows gate v3; v1 is reported.
    sys.exit(0 if result["gate_v3"]["pass"] else 1)


if __name__ == "__main__":
    main()
