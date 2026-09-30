#!/usr/bin/env python3
"""OpenCode agentic-coding benchmark against a qwfn-server: fix a module until its
unittest suite passes (tool calls, multi-turn, prefix reuse). Reports wall time,
success, and the server's /stats deltas.
  opencodebench.py NAME BINARY [server flags...]   env: LIBDIR, RUNS (1), OC_TIMEOUT (1200)
"""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
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
OUTDIR = os.environ.get("OUT_DIR", os.path.join(HERE, "logs"))
BASE = ["--ctx", "73728", "--kv", "q8_0", "--batch", "32768", "--reserve", "1100", "--mtp", MTP,
        "--think", "medium", "--think-budget", "5500", "--alias", "qwen3.8-flash-next", "--host", "127.0.0.1", "--port", "8081"]
PROMPT = ("Some tests in test_inventory.py fail. Run the test suite with `python3 -m unittest -v test_inventory`, "
          "fix inventory.py so that every test passes without changing the tests, run the suite again to confirm, "
          "then reply with one line saying what you changed.")
CONFIG = {"$schema": "https://opencode.ai/config.json",
          "provider": {"bench": {"npm": "@ai-sdk/openai-compatible", "name": "bench",
                                 "options": {"baseURL": "http://127.0.0.1:8081/v1"},
                                 "models": {"qwen3.8-flash-next": {"name": "q", "limit": {"context": 65536, "output": 16384}}}}},
          "model": "bench/qwen3.8-flash-next", "small_model": "bench/qwen3.8-flash-next"}

def stats():
    return json.load(urllib.request.urlopen("http://127.0.0.1:8081/stats", timeout=10))

def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items(): out.update(flat(v, f"{p}{k}."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool): out[p[:-1]] = d
    return out

def main():
    name, binary, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    env = dict(os.environ)
    if "LIBDIR" in env: env["LD_LIBRARY_PATH"] = env["LIBDIR"]
    log = open(os.path.join(os.environ.get("OC_OUT", OUTDIR), f"opencode.{name}.server.log"), "w")
    srv = subprocess.Popen([binary, MODEL] + BASE + extra, stdout=log, stderr=subprocess.STDOUT, env=env)
    root = tempfile.mkdtemp(prefix="ocbench-", dir=os.environ.get("OC_TMP"))
    try:
        for _ in range(600):
            time.sleep(2)
            if srv.poll() is not None: print("server died"); return
            try: urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=2); break
            except Exception: pass
        cfgdir = os.path.join(root, "config", "opencode"); os.makedirs(cfgdir)
        json.dump(CONFIG, open(os.path.join(cfgdir, "opencode.json"), "w"))
        ocenv = dict(os.environ, XDG_CONFIG_HOME=os.path.join(root, "config"), XDG_DATA_HOME=os.path.join(root, "data"),
                     XDG_STATE_HOME=os.path.join(root, "state"), XDG_CACHE_HOME=os.path.join(root, "cache"))
        for run in range(int(os.environ.get("RUNS", "1"))):
            work = os.path.join(root, f"work{run}"); shutil.copytree(os.path.join(HERE, "opencode_task"), work)
            subprocess.run(["git", "init", "-q"], cwd=work); subprocess.run(["git", "add", "."], cwd=work)
            subprocess.run(["git", "-c", "user.email=b@b", "-c", "user.name=b", "commit", "-qm", "task"], cwd=work)
            s0 = flat(stats()); t0 = time.time()
            try:
                oc = subprocess.run(["opencode", "run", "--auto", "--pure", "--format", "json", "--dir", work, "-m", "bench/qwen3.8-flash-next", PROMPT],
                                    cwd=work, env=ocenv, capture_output=True, text=True, timeout=int(os.environ.get("OC_TIMEOUT", "1200")))
                events_path = os.path.join(os.environ.get("OC_OUT", OUTDIR), f"opencode.{name}.run{run}.events.jsonl")
                open(events_path, "w").write(oc.stdout)
                out = oc.stdout[-400:]
            except subprocess.TimeoutExpired:
                out = "TIMEOUT"
            wall = time.time() - t0
            s1 = flat(stats())
            t = subprocess.run(["python3", "-m", "unittest", "-q", "test_inventory"], cwd=work, capture_output=True, text=True)
            ok = t.returncode == 0
            tests_untouched = subprocess.run(["git", "diff", "--quiet", "--", "test_inventory.py"], cwd=work).returncode == 0
            d = {k: s1[k] - s0.get(k, 0) for k in s1 if isinstance(s1[k], (int, float)) and s1[k] != s0.get(k, 0)}
            # Agent behaviour from the event stream: tool calls by tool, failed calls, turns.
            tools, failed_tools, texts = {}, 0, []
            if out != "TIMEOUT":
                for line in oc.stdout.splitlines():
                    try: ev = json.loads(line)
                    except Exception: continue
                    blob = json.dumps(ev)
                    part = ev.get("part") or {}
                    if ev.get("type") == "tool_use" or part.get("type") == "tool":
                        tn = part.get("tool") or ev.get("tool") or "?"
                        tools[tn] = tools.get(tn, 0) + 1
                        if '"status": "error"' in blob: failed_tools += 1
                    if ev.get("type") == "text" and part.get("text"): texts.append(part["text"])
            diff = subprocess.run(["git", "diff", "--stat"], cwd=work, capture_output=True, text=True).stdout.strip()
            rec = {"wall": wall, "ok": ok, "tests_untouched": tests_untouched, "tool_calls": tools, "failed_tool_calls": failed_tools,
                   "turns": d.get("totals.requests"), "diff_stat": diff, "delta": d, "reply": (texts[-1] if texts else out)[-400:]}
            open(os.path.join(os.environ.get("OC_OUT", OUTDIR), f"opencode.{name}.run{run}.json"), "w").write(json.dumps(rec, indent=1))
            print(f"   tool calls {sum(tools.values())} {tools} failed {failed_tools} turns {d.get('totals.requests')} | {diff.splitlines()[-1] if diff else 'no diff'}")
            keys = [k for k in d if any(w in k for w in ("generat", "prompt", "cache", "hit", "second", "decode", "token", "spec", "accept"))]
            print(f"RESULT {name} run{run}: wall {wall:.1f}s, tests {'PASS' if ok else 'FAIL'}{'' if tests_untouched else ' (tests modified!)'}")
            for k in sorted(keys)[:30]: print(f"   {k}: {d[k]:.4g}")
            print("   reply:", out.strip().splitlines()[-1] if out.strip() else "")
    finally:
        srv.terminate()
        try: srv.wait(60)
        except Exception: srv.kill()
        shutil.rmtree(root, ignore_errors=True)
main()
