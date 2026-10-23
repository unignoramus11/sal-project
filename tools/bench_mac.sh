#!/usr/bin/env bash
# Mac (CPU), final evaluation: bag probes, ECAPA and pooled prosody on the cross-validated benchmark
# (docs/benchmark.md), 9 runs at a time with one thread each, then the pooled evaluation.
# Bags: {W2V2, ZIPA} x {phone, char} x {unigram, bigram}; dense: ECAPA, prosody statistics (z-scored);
# x {all, shared} tracks x 5 folds x seeds 42-44. Finished runs are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1
B=results/benchmark
jobs=$(mktemp)
for s in 42 43 44; do for track in all shared; do for f in 0 1 2 3 4; do
  P=${track}_f$f
  for m in w2v2 zipa; do for tok in phone char; do for probe in unigram bigram; do
    [ -f results/runs/tp_${m}_rebuild_${tok}_${probe}_${P}_bench_s$s/metrics.json ] || \
      echo "uv run python scripts/run_tp.py --transcripts outputs/transcripts/${m}_rebuild.parquet --dataset $B/meta.parquet --splits $B/splits.parquet --tokens $tok --probe $probe --protocol $P --tag _bench --seeds $s" >> $jobs
  done; done; done
  [ -f results/runs/dense_ecapa_rebuild_${P}_bench_s$s/metrics.json ] || \
    echo "uv run python scripts/run_dense.py --features outputs/features/ecapa_rebuild --dataset $B/meta.parquet --splits $B/splits.parquet --protocol $P --tag _bench --seeds $s" >> $jobs
  [ -f results/runs/dense_prosody_rebuild_${P}_bench_s$s/metrics.json ] || \
    echo "uv run python scripts/run_dense.py --features outputs/features/prosody_rebuild --file pooled.npy --zscore --dataset $B/meta.parquet --splits $B/splits.parquet --protocol $P --tag _bench --seeds $s" >> $jobs
done; done; done
echo "$(wc -l < $jobs) runs"
tr '\n' '\0' < $jobs | xargs -0 -P 9 -n 1 sh -c 'eval "$1" 2>&1 | grep -E "^(tp|dense)_|Error|Traceback"' _
rm -f $jobs
uv run python scripts/benchmark_eval.py
echo BENCH_MAC_DONE
