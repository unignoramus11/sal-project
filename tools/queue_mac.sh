#!/usr/bin/env bash
# Mac CPU queue (run in the background). Skips runs that already have results.
set -uo pipefail
cd "$(dirname "$0")/.."
run() {  # name, command...
  local name=$1; shift
  [ -f "results/runs/${name}_s46/metrics.json" ] && return
  echo "$(date +%H:%M) start $name"; "$@" > "outputs/logs/$name.log" 2>&1 || echo "FAILED $name"
}
# wait for the running w2v2 bag job
while pgrep -f "run_tp.py --transcripts outputs/transcripts/w2v2_rebuild.parquet" >/dev/null; do sleep 30; done
S="--seeds 42 43 44 45 46 --device cpu"
for P in prism speaker_split; do
  run dense_ecapa_rebuild_${P}_generic uv run python scripts/run_dense.py --features outputs/features/ecapa_rebuild --protocol $P --prompts generic $S
  for PR in unigram bigram; do
    run tp_w2v2_rebuild_phone_${PR}_${P}_generic uv run python scripts/run_tp.py --transcripts outputs/transcripts/w2v2_rebuild.parquet --tokens phone --probe $PR --protocol $P --prompts generic $S
  done
done
# ZIPA, once its transcripts are copied over
until [ -f outputs/transcripts/zipa_rebuild.parquet ]; do
  scp -q pc:~/developer/sal-project/outputs/transcripts/zipa_rebuild.parquet outputs/transcripts/ 2>/dev/null || sleep 120
done
uv run python scripts/dpr.py outputs/transcripts/zipa_rebuild.parquet > outputs/logs/dpr_zipa.log 2>&1
for PR in unigram bigram; do for T in phone char; do for P in prism speaker_split; do
  run tp_zipa_rebuild_${T}_${PR}_${P} uv run python scripts/run_tp.py --transcripts outputs/transcripts/zipa_rebuild.parquet --tokens $T --probe $PR --protocol $P $S
done; done; done
for P in prism speaker_split; do for PR in unigram bigram; do
  run tp_zipa_rebuild_phone_${PR}_${P}_generic uv run python scripts/run_tp.py --transcripts outputs/transcripts/zipa_rebuild.parquet --tokens phone --probe $PR --protocol $P --prompts generic $S
done; done
echo MAC_QUEUE_DONE
