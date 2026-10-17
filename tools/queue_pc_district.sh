#!/usr/bin/env bash
# On the PC: transcribe the district candidate audio with both recognizers once it has been copied over.
set -uo pipefail
cd "$(dirname "$0")/.."
. tools/prism_env.sh
A=data/vaani_districts/candidates_audio.parquet
until [ -s $A ] && ! pgrep -f "rsync.*candidates_audio" >/dev/null; do sleep 60; done
sleep 120   # let the copy settle
for m in w2v2 zipa; do
  [ -f outputs/transcripts/${m}_districts.parquet ] && continue
  python scripts/transcribe.py --model $m --data $A --id-col audio_path \
    --out outputs/transcripts/${m}_districts.parquet --check 20 > outputs/logs/transcribe_${m}_districts.log 2>&1
done
echo DISTRICT_TRANSCRIPTS_DONE
