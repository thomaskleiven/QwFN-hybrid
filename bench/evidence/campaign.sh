#!/usr/bin/env bash
# Evidence campaign. Stages:
#   replay  aligned llama.cpp reference (q8_0 KV, one token per step; reused from REF_NLL_DIR when present) +
#           teacher-forced replays of the 5 scenarios under A (original QwFN, aider flags),
#           B1/B2 (original QwFN tuned, 1/2 drafts), C (hybrid), D (hybrid, Q4K draft head, 3 drafts),
#           x3 interleaved, with top-k dumps for KL
#   e2e     server end to end (A/B/C/D x3), OpenCode (3 sessions each for A/B/C), system information
#   bench/evidence/campaign.sh replay|e2e|all OUT_DIR
# env: MTP_Q4_MODEL (D's draft head, from bench/make_mtp_q4.sh), REF_NLL_DIR (llama.<sc>.u1.{nll,topk}),
#      CFGS (replay configurations, default "A B1 B2 C D")
set -u
. "$(dirname "$0")/../lib.sh"; need MODEL MTP_MODEL REF_QWFN_BUILD REF_QWFN_INSTR_BUILD REF_LLAMA_LIBS
cd "$REPO_DIR"; stage=$1; OUT=$(realpath -m "$2"); mkdir -p "$OUT"
HYB=$REPO_DIR/build
mon_start() { nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -lms 1000 > "$1.gpu.csv" 2>/dev/null & MON1=$!; vmstat -n 1 > "$1.vmstat.txt" & MON2=$!; }
mon_stop() { kill $MON1 $MON2 2>/dev/null; wait $MON1 $MON2 2>/dev/null; }
cfg_flags() { case $1 in   # binary dir, RAM GB, extra qwfn-gen flags
  A)  echo "$REF_QWFN_INSTR_BUILD 9" ;;
  B1) echo "$REF_QWFN_INSTR_BUILD 19 --ram-frac 0.95 --spec-block-layers 99" ;;
  B2) echo "$REF_QWFN_INSTR_BUILD 19 --ram-frac 0.95 --spec-block-layers 99 --mtp-drafts 2" ;;
  C)  echo "$HYB 19 --ram-frac 0.95 --spec-block-layers 99 --threads 5 --mtp-drafts 2" ;;
  D)  echo "$HYB 19 --ram-frac 0.95 --spec-block-layers 99 --threads 5 --mtp-drafts 3" ;; esac; }
: "${MTP_Q4_MODEL:?set MTP_Q4_MODEL (bench/make_mtp_q4.sh)}"
mtp_for() { [ $1 = D ] && echo "$MTP_Q4_MODEL" || echo "$MTP_MODEL"; }
replay_run() { # cfg scenario rep
  set -- $1 $2 $3 $(cfg_flags $1); local cfg=$1 sc=$2 rep=$3 bin=$4 ram=$5; shift 5
  local f=$OUT/$cfg.$sc.r$rep; mon_start $f
  until [ $(free -g | awk '/Mem:/{print $7}') -ge 24 ]; do sleep 5; done; sleep 15   # previous run's memory released
  MTP_MODEL=$(mtp_for $cfg) QWFN_PPL_VERBOSE=1 QWFN_TOPK_DUMP=$f.topk LOGS=$OUT/logs RAM=$ram SCENARIO=$sc GEN_BIN=$bin/qwfn-gen \
    bench/replay.sh ${cfg}_r$rep "$@" > $f.summary 2>&1
  mon_stop; cp $OUT/logs/$sc.${cfg}_r$rep.log $f.log; echo "$(date +%T) $cfg $sc r$rep: $(cut -d' ' -f3-5 $f.summary)"; }
if [ $stage = replay ] || [ $stage = all ]; then
  { echo "hybrid $(git rev-parse HEAD)"; echo "ref-instr $(git -C $REF_QWFN_INSTR_BUILD/.. rev-parse HEAD 2>/dev/null)"; } > $OUT/versions.txt
  for sc in code agent short reasoning longctx; do   # deterministic: one run (two were bit-identical)
    f=$OUT/llama.$sc.r1; R=${REF_NLL_DIR:-}/llama.$sc.u1
    if [ -n "${REF_NLL_DIR:-}" ] && [ -s $R.nll ]; then cp $R.nll $f.nll; cp $R.topk $f.topk; echo "llama $sc: reused $R"; continue; fi
    LD_LIBRARY_PATH=$REF_LLAMA_LIBS /usr/bin/time -f "wall %e s maxrss %M KB" build/llama-nll "$MODEL" $TOKENS/$sc.prompt $TOKENS/$sc.replay \
      --n ${SCEN_GEN[$sc]} --kv q8_0 --ubatch 1 --dump $f.nll --topk $f.topk > $f.log 2>&1
    echo "$(date +%T) llama $sc: $(grep -h 'llama.cpp replay' $f.log)"
  done
  for rep in 1 2 3; do for sc in code agent short reasoning longctx; do for cfg in ${CFGS:-A B1 B2 C D}; do replay_run $cfg $sc $rep; done; done; done
  python3 bench/evidence/summarize.py $OUT --json $OUT/summary.json --md $OUT/summary.md; echo "parity gate exit $?"
fi
if [ $stage = e2e ] || [ $stage = all ]; then
  export SB_OUT=$OUT OC_OUT=$OUT
  srv() { case $2 in
    A) LIBDIR=$REF_LLAMA_LIBS REPS=1 MAXTOK=700 python3 bench/serverbench.py $1 $REF_QWFN_BUILD/qwfn-server --spec-block --threads 8 --ram 9 ;;
    B) LIBDIR=$REF_LLAMA_LIBS QWFN_DRAFT_COST=0.4 REPS=1 MAXTOK=700 python3 bench/serverbench.py $1 $REF_QWFN_BUILD/qwfn-server --threads 8 --ram 19 --ram-frac 0.95 --mtp-drafts 2 ;;
    C) REPS=1 MAXTOK=700 python3 bench/serverbench.py $1 $HYB/qwfn-server --threads 5 --ram 19 --ram-frac 0.95 ;;
    D) MTP_MODEL=$MTP_Q4_MODEL REPS=1 MAXTOK=700 python3 bench/serverbench.py $1 $HYB/qwfn-server --threads 5 --ram 19 --ram-frac 0.95 ;; esac; }
  for rep in 1 2 3; do for cfg in A B C D; do
    until [ $(free -g | awk '/Mem:/{print $7}') -ge 24 ]; do sleep 5; done; sleep 15
    srv srv_${cfg}_r$rep $cfg 2>&1 | grep -E "RESULT|died"; done; done
  RUNS=3 LIBDIR=$REF_LLAMA_LIBS python3 bench/opencodebench.py oc_A $REF_QWFN_BUILD/qwfn-server --spec-block --threads 8 --ram 9
  RUNS=3 LIBDIR=$REF_LLAMA_LIBS QWFN_DRAFT_COST=0.4 python3 bench/opencodebench.py oc_B $REF_QWFN_BUILD/qwfn-server --threads 8 --ram 19 --ram-frac 0.95 --mtp-drafts 2
  RUNS=3 python3 bench/opencodebench.py oc_C $HYB/qwfn-server --threads 5 --ram 19 --ram-frac 0.95
  bench/evidence/sysinfo.sh $OUT/sysinfo.txt --hash-model
fi
