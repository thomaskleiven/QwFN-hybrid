#!/usr/bin/env bash
# Enforce docs/CODING_RULES.md: rule 10 by a warnings-as-errors build of every target, rules 1-9 by
# scripts/check_rules.py. Exits non-zero on any violation.
#   scripts/check_rules.sh [LLAMA_CPP_ROOT]     (default: the value in bench/local.env)
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=${1:-}
if [ -z "$LLAMA" ] && [ -f "$ROOT/bench/local.env" ]; then
    LLAMA=$(sed -n 's/^LLAMA_CPP_ROOT=//p' "$ROOT/bench/local.env" | sed 's/[[:space:]]*#.*//')
fi
[ -n "$LLAMA" ] || { echo "usage: $0 LLAMA_CPP_ROOT" >&2; exit 2; }
B="$ROOT/build-rules"
cmake -S "$ROOT" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_ROOT="$LLAMA" -DQWFN_WERROR=ON > /dev/null
cmake --build "$B" -j
echo "rule 10: warnings-as-errors build passed"
python3 "$ROOT/scripts/check_rules.py"
