#!/usr/bin/env bash
# PC: TP-GRU on the corrected official Vaani-Hi (results/official/*_clean.parquet), after tools/pc_official.sh:
# char GRU, phone GRU and shuffled phone GRU x {released split on clean rows, pseudo-speaker split} x seeds 42-44.
set -uo pipefail
cd "$(dirname "$0")/.."
. tools/prism_env.sh
export PYTHONUNBUFFERED=1
R=results/official
until grep -q OFFICIAL_PC_DONE outputs/logs/pc_official.log 2>/dev/null; do sleep 120; done
for seed in 42 43 44; do
  for spec in "char gru" "phone gru" "phone shuffle"; do
    set -- $spec
    for P in speaker_split prism; do
      name=tp_w2v2_official_${1}_${2}_${P}_clean_s$seed
      [ -f results/runs/$name/metrics.json ] && continue
      echo "$(date +%H:%M) start $name"
      python scripts/run_tp.py --transcripts outputs/transcripts/w2v2_official.parquet --dataset $R/meta_clean.parquet \
        --splits $R/splits_clean.parquet --tokens $1 --probe $2 --protocol $P --tag _clean --seeds $seed --device cuda \
        > outputs/logs/$name.log 2>&1 && tail -n 1 outputs/logs/$name.log || echo "FAILED $name"
    done
  done
done
echo OFFICIAL_CLEAN_DONE
