#!/usr/bin/env bash
# Prosody probe runs (RQ3) on the PC GPU once both TP-GRU queues have finished.
set -uo pipefail
cd "$(dirname "$0")/.."
. tools/prism_env.sh
export PYTHONUNBUFFERED=1
until grep -q QUEUE_DONE outputs/logs/queue_tp_gru.log 2>/dev/null && grep -q QUEUE_DONE outputs/logs/queue_tp_gru2.log 2>/dev/null; do
  sleep 120
done
for seed in 42 43 44 45 46; do
  for proto in speaker_split prism; do
    name="prosody_gru_${proto}_s${seed}"
    [ -f "results/runs/$name/metrics.json" ] && continue
    echo "$(date +%H:%M) start $name"
    python scripts/run_prosody.py --protocol $proto --seeds $seed --device cuda > "outputs/logs/$name.log" 2>&1 \
      && tail -1 "outputs/logs/$name.log" || echo "FAILED $name"
  done
done
echo PROSODY_DONE
