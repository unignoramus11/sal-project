#!/usr/bin/env bash
# PC GPU follow-ups once both TP-GRU queues are done: control task for the BiGRU (does a sequence model recognise
# speakers through transcripts?) and online-coding MDL for BiGRU vs shuffled BiGRU (W2V2 phones, both splits).
set -uo pipefail
cd "$(dirname "$0")/.."
. tools/prism_env.sh
export PYTHONUNBUFFERED=1
until grep -q QUEUE_DONE outputs/logs/queue_tp_gru.log 2>/dev/null && grep -q QUEUE_DONE outputs/logs/queue_tp_gru2.log 2>/dev/null; do
  sleep 120
done
for P in prism speaker_split; do
  for M in w2v2 zipa; do
    name="tp_${M}_rebuild_phone_gru_${P}_control"
    ls -d results/runs/${name}_s43 >/dev/null 2>&1 && continue
    echo "$(date +%H:%M) $name"
    python scripts/run_tp.py --transcripts outputs/transcripts/${M}_rebuild.parquet --tokens phone --probe gru \
      --protocol $P --control --seeds 42 43 --device cuda > outputs/logs/$name.log 2>&1 && tail -n 2 outputs/logs/$name.log
  done
done
for P in speaker_split prism; do
  for probe in gru shuffle; do
    echo "$(date +%H:%M) mdl w2v2 phone $probe $P"
    python scripts/mdl.py --transcripts outputs/transcripts/w2v2_rebuild.parquet --tokens phone --probe $probe \
      --protocol $P --seeds 0 1 2 --device cuda --max-epochs 40 --patience 4 > outputs/logs/mdl_${probe}_${P}.log 2>&1
    grep online outputs/logs/mdl_${probe}_${P}.log
  done
done
echo AFTER_DONE
