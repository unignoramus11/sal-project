#!/usr/bin/env bash
#SBATCH --job-name=bench_light
#SBATCH --time=24:00:00
# Usage: GPU=0 bash slurm/submit.sh slurm/bench_light.sbatch <all|shared> <fold 0-4>
# CPU probes on one fold of the cross-validated benchmark (docs/benchmark.md), seeds 42-44, run in parallel:
# bag probes {W2V2, ZIPA} x {phone, char} x {unigram, bigram}, the ECAPA voice embedding and pooled prosody
# statistics. Needs outputs/transcripts/{w2v2,zipa}_rebuild.parquet, outputs/features/ecapa_rebuild and
# outputs/features/prosody_rebuild. Finished runs are skipped. Each run logs to outputs/logs/bench/<run>.log; the
# job lists every failed run with the end of its log and exits non-zero if any failed.
. slurm/common.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
B=results/benchmark
P=${1}_f$2
L=outputs/logs/bench
mkdir -p $L
common="--dataset $B/meta.parquet --splits $B/splits.parquet --protocol $P --tag _bench"
jobs=$(mktemp)
add() { [ -f "results/runs/$1/metrics.json" ] || printf '%s\t%s\n' "$1" "$2" >> "$jobs"; }
for s in 42 43 44; do
  for m in w2v2 zipa; do for tok in phone char; do for probe in unigram bigram; do
    add "tp_${m}_rebuild_${tok}_${probe}_${P}_bench_s$s" "python scripts/run_tp.py \
--transcripts outputs/transcripts/${m}_rebuild.parquet --tokens $tok --probe $probe $common --seeds $s"
  done; done; done
  add "dense_ecapa_rebuild_${P}_bench_s$s" "python scripts/run_dense.py --features outputs/features/ecapa_rebuild $common --seeds $s"
  add "dense_prosody_rebuild_${P}_bench_s$s" "python scripts/run_dense.py --features outputs/features/prosody_rebuild \
--file pooled.npy --zscore $common --seeds $s"
done
echo "$(wc -l < "$jobs") runs to do on ${SLURM_CPUS_PER_TASK:-4} CPUs"
export L
cut -f1,2 "$jobs" | tr '\t' '\n' | tr '\n' '\0' | xargs -0 -n 2 -P "${SLURM_CPUS_PER_TASK:-4}" sh -c \
  'eval "$2" > "$L/$1.log" 2>&1 && echo "done   $1" || echo "FAILED $1"' _ | tee "$jobs.out"
rm -f "$jobs"
failed=$(grep -c '^FAILED' "$jobs.out" || true)
grep '^FAILED' "$jobs.out" | while read -r _ name; do echo "--- $name"; tail -n 5 "$L/$name.log"; done
rm -f "$jobs.out"
echo "$(find results/runs -maxdepth 1 -name "*_${P}_bench_s*" | wc -l) benchmark runs of $P in results/runs; $failed failed"
[ "$failed" -eq 0 ]
