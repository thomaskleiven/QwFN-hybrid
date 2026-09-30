#!/usr/bin/env bash
# Bit-exact regression check. Replays are deterministic, so a refactor must reproduce the same bytes.
#   bench/golden.sh capture DIR [quick|full]   record the reference outputs into DIR
#   bench/golden.sh check   DIR [quick|full]   rerun and compare byte for byte (exit 1 on any difference)
# quick: code + long-context replays and a warm second prompt. full adds agent, short, reasoning,
# a replay without MTP, one with every expert on the CPU (--vram 0), and the server's responses.
# extra (on its own) checks the non-default engine paths: io_uring reads, the speculative block and a
# CPU-only engine. (--spec-ahead 2 is left out: its prefetch timing still leaks into expert placement.) GOLDEN_BUILD selects the build directory (default build/).
set -u
. "$(dirname "$0")/lib.sh"; need MODEL MTP_MODEL
cd "$REPO_DIR"
mode=$1; DIR=$(realpath -m "$2"); level=${3:-quick}
BUILD=$(realpath -m "${GOLDEN_BUILD:-$REPO_DIR/build}")
[ "$mode" = capture ] || [ "$mode" = check ] || { echo "usage: $0 capture|check DIR [quick|full]" >&2; exit 2; }
OUT=$DIR/$mode; mkdir -p "$OUT"
H=(--ram-frac 0.95 --spec-block-layers 99 --threads 5)
fw() { until [ "$(free -g | awk '/Mem:/{print $7}')" -ge 24 ]; do sleep 5; done; sleep 10; }
fail=0
cmp_file() {   # name: compare OUT/name against DIR/capture/name
    [ "$mode" = check ] || return 0
    if cmp -s "$DIR/capture/$1" "$OUT/$1"; then echo "  same  $1"; else echo "  DIFF  $1"; fail=1; fi
}
replay() {     # name scenario [env...] -- [flags...]
    local name=$1 sc=$2; shift 2; local envs=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
    fw
    env "${envs[@]}" GEN_BIN="$BUILD/qwfn-gen" QWFN_PPL_VERBOSE=1 QWFN_TOPK_DUMP="$OUT/$name.topk" LOGS="$OUT" SCENARIO=$sc bench/replay.sh "golden_$name" "${H[@]}" "$@" > /dev/null
    # stdout (token ids) and stderr share the log, so an [nll] line can start mid-line
    grep -oE '\[nll\] #[0-9]+ tok [0-9]+: [0-9.eE+-]+ argmax [0-9]+' "$OUT/$sc.golden_$name.log" > "$OUT/$name.nll"
    echo "$name: $(grep -oE 'decode: .*tok/s\)' "$OUT/$sc.golden_$name.log")"
    cmp_file "$name.nll"; cmp_file "$name.topk"
}
second() {     # warm engine, then a streamed 1226-token prompt; fingerprint of the final logits
    head -c 100000 "$TOKENS/agent.prompt" | tr ' ' '\n' | tail -n +2 | head -1226 | tr '\n' ' ' > "$OUT/second.prompt"
    fw
    QWFN_SECOND_PROMPT="$OUT/second.prompt" "$BUILD/qwfn-gen" "$MODEL" --prompt-file "$TOKENS/code.prompt" --replay-file "$TOKENS/code.replay" \
        --gen 100 --ctx 73728 --kv q8_0 --batch 32768 --reserve 1100 --vram 12 --ram 19 --mtp "$MTP_MODEL" --mtp-drafts 2 "${H[@]}" > "$OUT/second.log" 2>&1
    grep -oE 'second prompt logits .*' "$OUT/second.log" > "$OUT/second.fp"
    echo "second: $(grep -oE 'second prompt: [0-9]+ tokens .* tok/s\)' "$OUT/second.log")"
    cmp_file second.fp
}
server() {     # temperature-0 responses of the 4 server requests (prompt text frozen at capture)
    [ "$mode" = capture ] && cp src/qwfn_io.h "$DIR/serverbench_src.txt"
    fw
    SB_SRC="$DIR/serverbench_src.txt" SB_OUT="$OUT" REPS=1 MAXTOK=300 python3 bench/serverbench.py golden "$BUILD/qwfn-server" --threads 5 --ram 19 --ram-frac 0.95 | grep -E 'RESULT|died'
    python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['outputs']))" "$OUT/server.golden.out.json" > "$OUT/server.outputs"
    cmp_file server.outputs
}
if [ "$level" != extra ]; then
    replay code code -- --mtp-drafts 2
    replay longctx longctx -- --mtp-drafts 2
    second
fi
if [ "$level" = full ]; then
    replay agent agent -- --mtp-drafts 2
    replay short short -- --mtp-drafts 2
    replay reasoning reasoning -- --mtp-drafts 2
    replay nomtp short NO_MTP=1 --
    replay vram0 code GEN=200 -- --mtp-drafts 2 --vram 0
    server
fi
if [ "$level" = extra ]; then
    replay iouring code GEN=200 -- --mtp-drafts 2 --io-uring
    replay specblock code GEN=200 -- --mtp-drafts 2 --spec-block-layers 1-47
    replay cpu short GEN=24 NO_MTP=1 -- --cpu
fi
[ "$mode" = check ] && { [ $fail = 0 ] && echo "GOLDEN: IDENTICAL" || echo "GOLDEN: DIFFERENT"; }
exit $fail
