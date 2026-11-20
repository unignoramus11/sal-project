#!/usr/bin/env bash
#SBATCH --job-name=bench_gru
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/bench_gru.sbatch <w2v2|zipa> <all|shared> "<folds>"
# Transcript BiGRUs on the cross-validated benchmark: PRiSM's character BiGRU, the phone BiGRU and the shuffled
# phone BiGRU, seeds 42-44. Finished runs are skipped.
. slurm/common.sh
B=results/benchmark
for f in $3; do
  for spec in "char gru" "phone gru" "phone shuffle"; do
    set -- "$1" "$2" "$3" $spec
    S=$(todo "tp_${1}_rebuild_${4}_${5}_${2}_f${f}_bench" 42 43 44)
    if [ -n "$S" ]; then
      python scripts/run_tp.py --transcripts "outputs/transcripts/${1}_rebuild.parquet" \
        --dataset $B/meta.parquet --splits $B/splits.parquet --tokens "$4" --probe "$5" --protocol "${2}_f${f}" \
        --tag _bench --seeds $S --device cuda
    fi
  done
done
