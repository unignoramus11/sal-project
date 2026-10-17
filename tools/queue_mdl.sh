#!/usr/bin/env bash
# Online-coding MDL for the bag probes on the Mac CPU (scripts/mdl.py). Skips finished (config, seed) files.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
for proto in speaker_split prism; do
  for m in w2v2 zipa; do
    for tok in phone char; do
      for probe in bigram unigram; do
        for seed in 0 1 2; do
          [ -f "results/mdl/mdl_${m}_rebuild_${tok}_${probe}_${proto}_s${seed}.json" ] && continue
          echo "$(date +%H:%M) mdl $m $tok $probe $proto s$seed"
          uv run python scripts/mdl.py --transcripts outputs/transcripts/${m}_rebuild.parquet --tokens $tok \
            --probe $probe --protocol $proto --seeds $seed 2>&1 | grep -v -i warn | grep online
        done
      done
    done
  done
done
echo MDL_DONE
