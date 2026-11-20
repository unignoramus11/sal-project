#!/usr/bin/env bash
#SBATCH --job-name=rp_bench
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp_benchmark.sbatch <model> <all|shared> "<folds>"
# PRiSM's RP-Attn (last layer, seeds 42-44) on folds of one track of the cross-validated benchmark
# (results/benchmark/, docs/benchmark.md). The frames are extracted once per job, so one job runs several folds.
. slurm/common.sh
. slurm/frames.sh
B=results/benchmark
use_frames "$1" rebuild "-1"
for f in $3; do
  python scripts/run_rp.py --features "$FEAT" --dataset $B/meta.parquet --splits $B/splits.parquet \
    --protocol "${2}_f${f}" --tag _bench --probe attn --layers -1 --seeds 42 43 44 --device cuda --in-memory ${RP_ARGS:-}
done
