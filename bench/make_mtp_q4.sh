#!/usr/bin/env bash
# Draft head with Q4_K routed experts (dense part stays Q8_0): 2.79 -> 1.63 GB, ~20% cheaper
# drafts at unchanged acceptance. Drafts are verified by the model, so output is unaffected.
#   bench/make_mtp_q4.sh IN_MTP_Q8_0.gguf OUT.gguf     (needs llama-quantize from LLAMA_CPP_ROOT)
set -eu
. "$(dirname "$0")/lib.sh"; need LLAMA_CPP_ROOT
"$LLAMA_CPP_ROOT/build/bin/llama-quantize" --allow-requantize \
  --tensor-type ffn_gate_exps=q4_K --tensor-type ffn_up_exps=q4_K --tensor-type ffn_down_exps=q4_K \
  "$1" "$2" Q8_0
