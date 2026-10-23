#!/usr/bin/env bash
#SBATCH --job-name=rp_bench
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp_benchmark.sbatch <model> <all|shared> <fold 0-4>
# PRiSM's RP-Attn (last layer, seeds 42-44) on one fold of one track of the cross-validated benchmark
# (results/benchmark/, docs/benchmark.md). Uses the rebuild features from step 3.
. slurm/common.sh
B=results/benchmark
python scripts/run_rp.py --features "outputs/features/${1}_rebuild" --dataset $B/meta.parquet --splits $B/splits.parquet \
  --protocol "${2}_f${3}" --tag _bench --probe attn --layers -1 --seeds 42 43 44 --device cuda --in-memory
