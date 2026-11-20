#!/usr/bin/env bash
#SBATCH --job-name=bench_light
#SBATCH --time=24:00:00
# Usage: GPU=0 bash slurm/submit.sh slurm/bench_light.sbatch <all|shared> <fold 0-4>
# CPU probes on one fold of the cross-validated benchmark (docs/benchmark.md), seeds 42-44, run in parallel:
# bag probes {W2V2, ZIPA} x {phone, char} x {unigram, bigram}, the ECAPA voice embedding and pooled prosody
# statistics. Needs outputs/transcripts/{w2v2,zipa}_rebuild.parquet, outputs/features/ecapa_rebuild (sent by
# Mohit) and outputs/features/prosody_rebuild. Finished runs are skipped.
. slurm/common.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
B=results/benchmark
P=${1}_f$2
common="--dataset $B/meta.parquet --splits $B/splits.parquet --protocol $P --tag _bench"
jobs=$(mktemp)
for s in 42 43 44; do
  for m in w2v2 zipa; do for tok in phone char; do for probe in unigram bigram; do
    [ -f results/runs/tp_${m}_rebuild_${tok}_${probe}_${P}_bench_s$s/metrics.json ] || echo "python scripts/run_tp.py \
--transcripts outputs/transcripts/${m}_rebuild.parquet --tokens $tok --probe $probe $common --seeds $s" >> "$jobs"
  done; done; done
  [ -f results/runs/dense_ecapa_rebuild_${P}_bench_s$s/metrics.json ] || \
    echo "python scripts/run_dense.py --features outputs/features/ecapa_rebuild $common --seeds $s" >> "$jobs"
  [ -f results/runs/dense_prosody_rebuild_${P}_bench_s$s/metrics.json ] || echo "python scripts/run_dense.py \
--features outputs/features/prosody_rebuild --file pooled.npy --zscore $common --seeds $s" >> "$jobs"
done
echo "$(wc -l < "$jobs") runs on ${SLURM_CPUS_PER_TASK:-4} CPUs"
tr '\n' '\0' < "$jobs" | xargs -0 -P "${SLURM_CPUS_PER_TASK:-4}" -n 1 sh -c 'eval "$1" > /dev/null 2>&1 || echo "FAILED $1"' _
rm -f "$jobs"
echo "$(find results/runs -maxdepth 1 -name "*_${P}_bench_s*" | wc -l) benchmark runs of $P in results/runs"
