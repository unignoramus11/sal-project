#!/usr/bin/env bash
# Final evaluation: RP-Attn on the cross-validated benchmark, 3 models x {all, shared} x 5 folds, seeds 42-44.
# 12 jobs: each extracts the frames once and runs two or three folds (all: 6-9 h; shared: about a third of that).
cd "$(dirname "$0")/.."
for M in w2v2 zipa whisper; do
  for T in all shared; do
    bash slurm/submit.sh slurm/rp_benchmark.sbatch "$M" "$T" "0 1 2"
    bash slurm/submit.sh slurm/rp_benchmark.sbatch "$M" "$T" "3 4"
  done
done
