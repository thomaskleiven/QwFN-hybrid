#!/usr/bin/env bash
# llama-bench-style numbers for the README: pp512, pp16384, tg128, tg128 @ 16K depth.
#   bench/llamabench.sh OUT_DIR       (LLAMA_BENCH: llama-bench binary, default $REF_LLAMA_LIBS/llama-bench)
# llama.cpp is measured with llama-bench itself (routed experts on the CPU, the usual setup for a
# model larger than VRAM); QwFN-hybrid with qwfn-gen on the first 512 / 16384 tokens of the
# long-context prompt, generating freely (greedy). tg is reported without and with MTP drafts.
set -u
. "$(dirname "$0")/lib.sh"; need MODEL MTP_MODEL REF_LLAMA_LIBS
cd "$REPO_DIR"; OUT=$(realpath -m "$1"); mkdir -p "$OUT"
free_wait() { until [ $(free -g | awk '/Mem:/{print $7}') -ge 24 ]; do sleep 5; done; sleep 15; }
for n in 512 16384; do tr ' ' '\n' < $TOKENS/longctx.prompt | head -$n | tr '\n' ' ' > $OUT/p$n.prompt; done
BASE=(--ctx 73728 --kv q8_0 --batch 32768 --reserve 1100 --vram 12)

# engine name, binary, extra flags
run() { local name=$1 bin=$2; shift 2
  for n in 512 16384; do for mtp in off on; do for r in 1 2; do
    local f=$OUT/$name.p$n.mtp_$mtp.r$r.log M=(); [ $mtp = on ] && M=(--mtp "$MTP_MODEL")
    free_wait; "$bin" "$MODEL" --prompt-file $OUT/p$n.prompt --gen 128 "${BASE[@]}" "$@" "${M[@]}" > $f 2>&1
    echo "$(date +%T) $name p$n mtp=$mtp r$r: $(grep -oE 'prefill: .*|decode: .*' $f | tr '\n' ' ')"
  done; done; done; }
run hybrid build/qwfn-gen --ram 19 --ram-frac 0.95 --threads 5

free_wait
LD_LIBRARY_PATH=$REF_LLAMA_LIBS ${LLAMA_BENCH:-$REF_LLAMA_LIBS/llama-bench} -m "$MODEL" -ngl 99 -ot "exps=CPU" -fa 1 -ctk q8_0 -ctv q8_0 \
  -t 8 -p 512,16384 -n 128 -r 2 -o md > $OUT/llama.md 2> $OUT/llama.err
free_wait
LD_LIBRARY_PATH=$REF_LLAMA_LIBS ${LLAMA_BENCH:-$REF_LLAMA_LIBS/llama-bench} -m "$MODEL" -ngl 99 -ot "exps=CPU" -fa 1 -ctk q8_0 -ctv q8_0 \
  -t 8 -p 0 -n 128 -d 16384 -r 1 -o md >> $OUT/llama.md 2>> $OUT/llama.err
cat $OUT/llama.md
