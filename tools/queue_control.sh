#!/usr/bin/env bash
# Control-task runs on the Mac (speaker-constant random locations; sal.data.control).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
for P in prism speaker_split; do
  echo "$(date +%H:%M) ecapa control $P"
  uv run python scripts/run_dense.py --features outputs/features/ecapa_rebuild --protocol $P --control \
    --seeds 42 43 44 45 46 2>&1 | grep -v -i warn | tail -5
done
for P in prism speaker_split; do
  for M in w2v2 zipa; do
    for TOK in phone char; do
      echo "$(date +%H:%M) $M $TOK bigram control $P"
      uv run python scripts/run_tp.py --transcripts outputs/transcripts/${M}_rebuild.parquet --tokens $TOK \
        --probe bigram --protocol $P --control --seeds 42 43 44 45 46 2>&1 | grep '^tp_'
    done
  done
done
echo CONTROL_DONE
