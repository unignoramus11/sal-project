#!/usr/bin/env bash
# Transcript-probe GRU runs on a CUDA machine, seed by seed so every condition gets seed 42 first.
# Skips runs whose results/runs/<name>/metrics.json exists, so it can be restarted at any time.
#   bash tools/queue_tp_gru.sh            (waits for both transcript files first)
set -uo pipefail
cd "$(dirname "$0")/.."
. tools/prism_env.sh
for m in w2v2 zipa; do
  until [ -f outputs/transcripts/${m}_rebuild.parquet ]; do sleep 60; done
done
for seed in 42 43 44 45 46; do
  for m in w2v2 zipa; do
    for spec in "char gru" "phone gru" "phone shuffle"; do
      set -- $spec
      for proto in prism speaker_split; do
        name="tp_${m}_rebuild_${1}_${2}_${proto}_s${seed}"
        [ -f "results/runs/$name/metrics.json" ] && continue
        echo "$(date +%H:%M) start $name"
        python scripts/run_tp.py --transcripts outputs/transcripts/${m}_rebuild.parquet --tokens $1 --probe $2 \
          --protocol $proto --seeds $seed --device cuda > "outputs/logs/$name.log" 2>&1 \
          && tail -1 "outputs/logs/$name.log" || echo "FAILED $name"
      done
    done
  done
done
echo QUEUE_DONE
