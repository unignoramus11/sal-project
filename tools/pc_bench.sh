#!/usr/bin/env bash
# PC (GPU), final evaluation: TP-GRU on the cross-validated benchmark (scripts/make_benchmark.py, docs/benchmark.md):
# {W2V2, ZIPA} x {char GRU, phone GRU, shuffled phone GRU} x {all, shared} tracks x 5 folds x seeds 42-43.
# Two copies can run on one GPU: SEEDS="43 42" for the second; a lock directory per run stops both taking it.
# Needs results/benchmark/{meta,splits}.parquet and outputs/transcripts/{w2v2,zipa}_rebuild.parquet.
set -uo pipefail
cd "$(dirname "$0")/.."
. tools/prism_env.sh
export PYTHONUNBUFFERED=1
B=results/benchmark
LOCKS=/tmp/sal_locks_bench
mkdir -p $LOCKS outputs/logs
for seed in ${SEEDS:-42 43}; do
  for track in all shared; do
    for f in 0 1 2 3 4; do
      for m in w2v2 zipa; do
        for spec in "phone gru" "phone shuffle" "char gru"; do
          set -- $spec
          name=tp_${m}_rebuild_${1}_${2}_${track}_f${f}_bench_s$seed
          [ -f results/runs/$name/metrics.json ] && continue
          mkdir $LOCKS/$name 2>/dev/null || continue
          echo "$(date +%H:%M) start $name"
          python scripts/run_tp.py --transcripts outputs/transcripts/${m}_rebuild.parquet --dataset $B/meta.parquet \
            --splits $B/splits.parquet --tokens $1 --probe $2 --protocol ${track}_f$f --tag _bench --seeds $seed \
            --device cuda > outputs/logs/$name.log 2>&1 && tail -n 1 outputs/logs/$name.log || echo "FAILED $name"
        done
      done
    done
  done
done
echo BENCH_PC_DONE
