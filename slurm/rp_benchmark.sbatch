#!/usr/bin/env bash
#SBATCH --job-name=rp_bench
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp_benchmark.sbatch <model> <all|shared> "<folds>" ["<probes>"] ["<layers>"]
# Representation probes (default: PRiSM's RP-Attn, last layer) on folds of one track of the cross-validated
# benchmark (results/benchmark/, docs/benchmark.md), seeds 42-44. The frames are extracted once per job, so one
# job runs several folds and probes. Runs with results are skipped.
. slurm/common.sh
. slurm/frames.sh
B=results/benchmark
PROBES=${4:-attn}
LAYERS=${5:--1}
left=""
for f in $3; do for p in $PROBES; do
  [ -n "$(rp_todo "$1" rebuild "$LAYERS" "$p" "${2}_f${f}_bench" "42 43 44")" ] && left=1
done; done
[ -n "$left" ] || { echo "all runs done"; exit 0; }
use_frames "$1" rebuild "$LAYERS"
for f in $3; do
  for p in $PROBES; do
    S=$(rp_todo "$1" rebuild "$LAYERS" "$p" "${2}_f${f}_bench" "42 43 44")
    if [ -n "$S" ]; then
      python scripts/run_rp.py --features "$FEAT" --dataset $B/meta.parquet --splits $B/splits.parquet \
        --protocol "${2}_f${f}" --tag _bench --probe "$p" --layers $LAYERS --seeds $S --device cuda --in-memory ${RP_ARGS:-}
    fi
  done
done
