# Sourced by the bench scripts: repository root, local configuration, required variables.
BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$BENCH_DIR/.." && pwd)
# bench/local.env fills in what the environment does not set (the environment wins).
if [ -f "$BENCH_DIR/local.env" ]; then
  while IFS='=' read -r k v; do
    [[ "$k" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    [ -n "${!k:-}" ] || export "$k=$v"
  done < "$BENCH_DIR/local.env"
fi
need() { for v in "$@"; do [ -n "${!v:-}" ] || { echo "bench: $v is not set (see bench/env.example -> bench/local.env)" >&2; exit 2; }; done; }
TOKENS=$BENCH_DIR/tokens
LOGS=${LOGS:-$BENCH_DIR/logs}; mkdir -p "$LOGS"
declare -A SCEN_GEN=([code]=800 [agent]=800 [short]=300 [reasoning]=600 [longctx]=300)
