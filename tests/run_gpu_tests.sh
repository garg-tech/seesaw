#!/usr/bin/env bash
# Run every correctness test on the GPU machine and print one line per test.
# Needs both GPUs free (stop the engine or vLLM first) and the virtualenv active.
#
#   tests/run_gpu_tests.sh            # full output in logs/tests_<timestamp>.log
#
# What each checks (expected: every line ends in PASS):
#   test_scheduler.py           switch policy and request handling, fake workers, no GPU
#   test_layers_vs_hf.py 0|1    first 6 layers on one GPU vs transformers, both GPUs
#   test_tp_golden.py tp|pp     full model in TP2 / PP2 vs vLLM's greedy outputs (teacher-forced)
#   test_tp_golden.py tp-lean   TP2 with the split embedding Seesaw uses
#   test_pp_decode.py 26        pipelined PP decode at 26/22 vs vLLM
#   test_reshard.py 24|26       PP prefill -> re-shard -> TP decode -> back, twice, vs vLLM
#   test_reshard.py 28 lean     the same with KV parking and the split embedding (as served)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/tests_$(date +%Y%m%d_%H%M%S).log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
fails=0
for t in "test_scheduler.py" "test_layers_vs_hf.py 0" "test_layers_vs_hf.py 1" \
         "test_tp_golden.py tp" "test_tp_golden.py pp" "test_tp_golden.py tp-lean" \
         "test_pp_decode.py 26" "test_reshard.py 24" "test_reshard.py 26" "test_reshard.py 28 lean"; do
  start=$(date +%s)
  out=$(python tests/$t 2>&1)
  code=$?
  printf '===== tests/%s (exit %s)\n%s\n' "$t" $code "$out" >> "$LOG"
  if [ $code = 0 ] && grep -qw PASS <<<"$out" && ! grep -qw FAIL <<<"$out"; then res=PASS
  else res=FAIL; fails=$((fails + 1)); fi
  printf '%-28s %4ss  %s\n' "$t" $(( $(date +%s) - start )) "$res"
done
echo "log: $LOG"
exit $fails
