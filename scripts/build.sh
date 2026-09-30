#!/usr/bin/env bash
# One-step build: fetches ggml/llama.cpp at the validated commit next to this repo, applies
# patches/, builds it with CUDA, then builds the engine into build/.
#   scripts/build.sh [LLAMA_CPP_DIR]     (default: ../llama.cpp; reused if already there)
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=${1:-$ROOT/../llama.cpp}
COMMIT=ca1426903
if [ ! -d "$LLAMA/.git" ]; then
    git clone https://github.com/unslothai/llama.cpp "$LLAMA"
    git -C "$LLAMA" checkout -q $COMMIT
    git -C "$LLAMA" apply "$ROOT"/patches/*.patch
fi
cmake -S "$LLAMA" -B "$LLAMA/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build "$LLAMA/build" -j
cmake -S "$ROOT" -B "$ROOT/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_ROOT="$(cd "$LLAMA" && pwd)"
cmake --build "$ROOT/build" -j
echo "built: $ROOT/build/qwfn-server"
