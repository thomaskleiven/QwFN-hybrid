#!/usr/bin/env bash
# (Re)create a scenario's prompt tokens and, with the original QwFN, its greedy continuation.
# The published evidence uses the committed files; regenerating changes the continuation.
#   bench/make_tokens.sh SCENARIO     env: REF_QWFN_BUILD (for the continuation)
set -eu
. "$(dirname "$0")/lib.sh"; need MODEL MTP_MODEL
cd "$REPO_DIR"; sc=$1; TOK=build/qwfn-tok; P=$TOKENS/$sc.prompt; R=$TOKENS/$sc.replay
case $sc in
  short)     $TOK "$MODEL" --chat "Explain in three sentences why mixture-of-experts models are cheaper to run than dense models with the same parameter count." > $P ;;
  code)      $TOK "$MODEL" --chat "Write a Python module with a class LRUCache(capacity) offering get(key) and put(key, value) in O(1) using a doubly linked list and a dict, plus a small unittest suite covering eviction order, update-on-access and capacity 1. Code only, no explanations." --think off > $P ;;
  agent)     $TOK "$MODEL" --chat "Review the attached C++ header. Propose a concrete refactor of the configuration struct into smaller structs and write the new code." --file $TOKENS/agent_file.txt --think off > $P ;;
  reasoning) $TOK "$MODEL" --chat "A warehouse ships 3 kinds of boxes: small (2 kg), medium (5 kg) and large (11 kg). A truck carries at most 400 kg and at most 60 boxes. Every shipment must contain at least twice as many small boxes as large ones. What is the largest total weight a single truck can carry? Show your reasoning and give the counts of each box." > $P ;;
  longctx)   $TOK "$MODEL" --chat "Using only the attached source file, explain how the RAM tier chooses an eviction victim and how RAM-to-VRAM promotion decides whether to replace a resident expert. Cite the function names." --file src/qwfn_expert_cache.cpp --think off > $P ;;
  *) echo "unknown scenario $sc" >&2; exit 1 ;;
esac
need REF_QWFN_BUILD
"$REF_QWFN_BUILD/qwfn-gen" "$MODEL" --prompt-file $P --save-replay $R --gen ${SCEN_GEN[$sc]} \
  --ctx 73728 --kv q8_0 --batch 16384 --threads 8 --reserve 1100 --vram 12 --ram 9 --spec-block --mtp "$MTP_MODEL" > $LOGS/$sc.make_tokens.log 2>&1
echo "$P ($(wc -w < $P) tokens), $R ($(wc -l < $R) tokens)"
