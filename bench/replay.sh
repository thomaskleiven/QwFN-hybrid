#!/usr/bin/env bash
# Teacher-forced replay: one fixed prompt + continuation through qwfn-gen, so every binary and
# configuration is scored on identical tokens (decode tok/s, per-token NLL/argmax with
# QWFN_PPL_VERBOSE=1).
#   bench/replay.sh NAME [extra qwfn-gen flags...]
# env: SCENARIO=code|agent|short|reasoning|longctx (code), GEN (per scenario), RAM (GB, 19), NO_MTP,
#      GEN_BIN (build/qwfn-gen). Flags after NAME override the defaults below.
set -u
. "$(dirname "$0")/lib.sh"; need MODEL MTP_MODEL
cd "$REPO_DIR"
SCEN=${SCENARIO:-code}; GEN=${GEN:-${SCEN_GEN[$SCEN]}}; RAM=${RAM:-19}
GEN_BIN=${GEN_BIN:-$REPO_DIR/build/qwfn-gen}
BASE=(--ctx 73728 --kv q8_0 --batch 32768 --threads 8 --reserve 1100 --vram 12 --ram "$RAM" --spec-block)
[ -n "${NO_MTP:-}" ] || BASE+=(--mtp "$MTP_MODEL")   # NO_MTP=1: plain decode, one token per step
[ $# -ge 1 ] || { echo "usage: $0 NAME [flags]" >&2; exit 1; }
name=$1; shift
P=$TOKENS/$SCEN.prompt; R=$TOKENS/$SCEN.replay
[ -s "$P" ] && [ -s "$R" ] || { echo "missing $P or $R (see bench/tokens/README.md)" >&2; exit 2; }
log=$LOGS/$SCEN.$name.log
/usr/bin/time -f "wall %e s maxrss %M KB" "$GEN_BIN" "$MODEL" --prompt-file "$P" --replay-file "$R" --ppl --gen "$GEN" "${BASE[@]}" "$@" > "$log" 2>&1
python3 - "$log" "$SCEN" "$name" <<'PY'
import re, sys
log, scen, name = sys.argv[1:4]
t = open(log, errors="replace").read()
g = lambda p: (re.search(p, t) or [None, "-"])[1]
print(f"scenario={scen} name={name} tok_s={g(r'decode: \d+ tokens in [\d.]+ s\s+\(([\d.]+) tok/s\)')} "
      f"NLL={g(r'replay NLL: ([\d.]+)')} hit={g(r'expert cache: ([\d.]+)% hit')} log={log}")
PY
