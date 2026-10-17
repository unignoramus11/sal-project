#!/usr/bin/env bash
# Gemination chain (run in the background on the Mac): download -> align on the PC GPU -> measure on the Mac.
set -uo pipefail
cd "$(dirname "$0")/.."
A=data/vaani_districts/candidates_audio.parquet
echo "$(date +%H:%M) waiting for download"
while pgrep -f scripts/district_audio.py >/dev/null; do sleep 60; done
[ -s $A ] || { echo "download produced no file"; exit 1; }
echo "$(date +%H:%M) download done: $(du -h $A | cut -f1)"
tools/sync_pc.sh >/dev/null
ssh pc 'mkdir -p ~/developer/sal-project/data/vaani_districts ~/developer/sal-project/results/gemination'
rsync -a $A pc:~/developer/sal-project/data/vaani_districts/
rsync -a results/gemination/candidate_words.csv pc:~/developer/sal-project/results/gemination/
echo "$(date +%H:%M) copied to pc; aligning on GPU"
ssh pc 'cd ~/developer/sal-project && . tools/prism_env.sh && python scripts/align_districts.py --device cuda > outputs/logs/align_districts.log 2>&1'
tail -2 <(ssh pc 'cat ~/developer/sal-project/outputs/logs/align_districts.log')
rsync -a pc:~/developer/sal-project/data/vaani_districts/alignments.parquet data/vaani_districts/
echo "$(date +%H:%M) alignments back; measuring"
uv run python scripts/measure_closure.py --workers 6 > outputs/logs/measure_closure.log 2>&1
tail -8 outputs/logs/measure_closure.log
echo "$(date +%H:%M) CHAIN_DONE"
