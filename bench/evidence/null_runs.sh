#!/usr/bin/env bash
# Gate v5 null spread (docs/PARITY.md): llama.cpp scoring the same tokens 2/3/4 per decode call,
# compared by summarize.py --null against its one-token-per-step reference. Resumable: a
# scenario whose log already has its result line is skipped.
#   bench/evidence/null_runs.sh [OUT_DIR]   (default bench/evidence/raw/null5)
set -u
cd "$(dirname "$0")/../.."
. bench/lib.sh
O=${1:-bench/evidence/raw/null5}; mkdir -p "$O"
for u in 2 3 4; do for sc in code agent short reasoning longctx; do
  f=$O/llama.$sc.u$u
  grep -q 'llama.cpp replay' "$f.log" 2>/dev/null && { echo "u$u $sc: done"; continue; }
  LD_LIBRARY_PATH=$REF_LLAMA_LIBS build/llama-nll "$MODEL" "$TOKENS/$sc.prompt" "$TOKENS/$sc.replay" \
    --n "${SCEN_GEN[$sc]}" --kv q8_0 --ubatch "$u" --dump "$f.nll" --topk "$f.topk" > "$f.log" 2>&1
  echo "$(date +%T) u$u $sc: $(grep -h 'llama.cpp replay' "$f.log")"
done; done
