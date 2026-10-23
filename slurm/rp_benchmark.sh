#!/usr/bin/env bash
# Final evaluation (November): RP-Attn on the cross-validated benchmark, 3 models x {all, shared} x 5 folds = 30 jobs.
# Needs step 3 (features) and `git pull` for results/benchmark/{meta,splits}.parquet.
cd "$(dirname "$0")/.."
for M in w2v2 zipa whisper; do
  for T in all shared; do
    for F in 0 1 2 3 4; do
      bash slurm/submit.sh slurm/rp_benchmark.sbatch "$M" "$T" "$F"
    done
  done
done
