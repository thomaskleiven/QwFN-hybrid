#!/usr/bin/env python3
"""End-to-end server benchmark (bench/serverbench.py): start a qwfn-server, send fixed coding requests
(greedy, thinking on as in the aider runs), report decode/prefill tok/s from the
server's own timings, stop the server.
  serverbench.py NAME BINARY [server flags...]      (env LIBDIR for LD_LIBRARY_PATH)
"""
import json, os, subprocess, sys, time, urllib.request
def load_env():
    """bench/local.env (KEY=VALUE lines) under the process environment."""
    env = {}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local.env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); env[k.strip()] = v.strip()
    env.update({k: v for k, v in os.environ.items() if k in ("MODEL", "MTP_MODEL")})
    for k in ("MODEL", "MTP_MODEL"):
        if not env.get(k): sys.exit(f"bench: {k} is not set (see bench/env.example -> bench/local.env)")
    return env
_E = load_env()
MODEL, MTP = _E["MODEL"], _E["MTP_MODEL"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("OUT_DIR", os.path.join(HERE, "logs"))
BASE = ["--ctx", "73728", "--kv", "q8_0", "--batch", "32768", "--reserve", "1100", "--mtp", MTP,
        "--think", "medium", "--think-budget", "5500", "--alias", "q", "--host", "127.0.0.1", "--port", "8081"]
SRC = open(os.environ.get("SB_SRC") or os.path.join(HERE, "..", "src", "qwfn_io.h")).read()[:6000]   # SB_SRC: a frozen copy for regression checks
REQS = [
  [{"role": "user", "content": "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H5M' into total seconds. Include type hints and a few doctests."}],
  [{"role": "user", "content": "In Go, implement a thread-safe bounded LRU cache with Get/Put and a TTL per entry. Keep it under 80 lines."}],
  [{"role": "user", "content": "Here is a C++ header:\n\n" + SRC + "\n\nList three concrete bugs or risks you see in it and propose a fix for each, with code."}],
  [{"role": "user", "content": "Refactor this JavaScript to async/await and add error handling:\n\nfunction load(u, cb){ fetch(u).then(r=>r.json()).then(j=>cb(null,j)).catch(e=>cb(e)) }\nfunction all(us, cb){ var out=[], n=us.length; us.forEach((u,i)=>load(u,(e,j)=>{ if(e) return cb(e); out[i]=j; if(--n===0) cb(null,out) })) }"}],
]
def post(path, body, timeout=900):
    r = urllib.request.Request("http://127.0.0.1:8081" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def main():
    name, binary, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    env = dict(os.environ)
    if "LIBDIR" in env: env["LD_LIBRARY_PATH"] = env["LIBDIR"]
    outdir = os.environ.get("SB_OUT", OUTDIR); os.makedirs(outdir, exist_ok=True)
    log = open(f"{outdir}/server.{name}.log", "w")
    p = subprocess.Popen([binary, MODEL] + BASE + extra, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        for _ in range(600):
            time.sleep(2)
            if p.poll() is not None: print("server died"); return
            try: urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=2); break
            except Exception: pass
        rows = []; outs = []
        for rep in range(int(os.environ.get("REPS", "1"))):
            for i, msgs in enumerate(REQS):
                t0 = time.time()
                r = post("/v1/chat/completions", {"model": "q", "messages": msgs, "temperature": 0, "max_tokens": int(os.environ.get("MAXTOK", "700"))})
                tm = r.get("timings", {})
                outs.append(r["choices"][0]["message"].get("content", ""))
                rows.append((i, tm.get("predicted_n"), tm.get("predicted_per_second"), tm.get("prompt_n"), tm.get("prompt_per_second"), time.time() - t0))
                print(f"{name} req{i} gen {tm.get('predicted_n')} @ {tm.get('predicted_per_second', 0):.2f} tok/s | prompt {tm.get('prompt_n')} @ {tm.get('prompt_per_second', 0):.1f} tok/s | wall {time.time()-t0:.1f}s", flush=True)
        json.dump({"outputs": outs, "rows": rows}, open(f"{outdir}/server.{name}.out.json", "w"))
        gn = sum(r[1] or 0 for r in rows); gt = sum((r[1] or 0) / (r[2] or 1e9) for r in rows)
        print(f"RESULT {name}: decode {gn} tok in {gt:.1f}s = {gn/gt:.2f} tok/s (aggregate)")
        try: print("stats:", json.dumps(json.load(urllib.request.urlopen("http://127.0.0.1:8081/stats", timeout=5)))[:600])
        except Exception as e: print("stats n/a", e)
    finally:
        p.terminate()
        try: p.wait(60)
        except Exception: p.kill()
main()
