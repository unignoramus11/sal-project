#!/usr/bin/env bash
# PC GPU work after both TP-GRU queues, one job at a time (three concurrent jobs overcommit the 6 GB GPU and
# slow everything ~10x): district transcription (both recognizers; its DISTRICT_TRANSCRIPTS_DONE marker is
# what tools/gemination_extra.sh waits for), then BiGRU control + MDL (tools/queue_pc_after.sh), then the
# prosody BiGRU runs that Ada has not delivered (tools/queue_pc_prosody.sh skips finished runs).
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q QUEUE_DONE outputs/logs/queue_tp_gru.log 2>/dev/null && grep -q QUEUE_DONE outputs/logs/queue_tp_gru2.log 2>/dev/null; do
  sleep 120
done
bash tools/queue_pc_district.sh >> outputs/logs/queue_pc_district.log 2>&1
bash tools/queue_pc_after.sh >> outputs/logs/queue_pc_after.log 2>&1
bash tools/queue_pc_prosody.sh >> outputs/logs/queue_pc_prosody.log 2>&1
echo POST_DONE
