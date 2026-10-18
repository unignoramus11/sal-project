#!/usr/bin/env bash
# Add the district shards fetched after the first gemination pass: wait for the download and the first pass,
# rebuild candidates_audio.parquet from all parts, align only the new shards' utterances (2 Mac shards) while
# the PC transcribes them with both recognizers, merge alignments and transcripts with the first pass, then
# rerun closure measurement and analysis. The audit set is not re-drawn.
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/vaani_districts
R=~/developer/sal-project
while pgrep -f scripts/district_audio_files.py >/dev/null; do sleep 60; done
until grep -q GEMINATION_DONE outputs/logs/finish_gemination.log 2>/dev/null; do sleep 60; done
uv run python - <<'PY'
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
D = Path("data/vaani_districts")
first = set((D / "parts_first_pass.txt").read_text().split())
new = [p for p in sorted((D / "candidates_parts").glob("*.parquet")) if p.name not in first]
print(len(new), "new shards")
if new:
    pq.write_table(pa.concat_tables([pq.read_table(p) for p in new], promote_options="default"), D / "candidates_extra.parquet")
PY
[ -s $D/candidates_extra.parquet ] || { echo "no new shards"; exit 0; }
uv run python scripts/district_audio_files.py --assemble-only

# PC: transcribe the extra utterances once the first-pass district transcription is finished
rsync -a $D/candidates_extra.parquet pc:$R/$D/
ssh pc "cd $R && nohup bash -c '. tools/prism_env.sh; export PYTHONUNBUFFERED=1; \
  until grep -q DISTRICT_TRANSCRIPTS_DONE outputs/logs/queue_pc_district.log; do sleep 60; done; \
  for m in w2v2 zipa; do python scripts/transcribe.py --model \$m --data $D/candidates_extra.parquet --id-col audio_path \
    --out outputs/transcripts/\${m}_districts_extra.parquet --check 20 > outputs/logs/transcribe_\${m}_districts_extra.log 2>&1; done; \
  echo EXTRA_TRANSCRIPTS_DONE > outputs/logs/extra_transcripts.done' > /dev/null 2>&1 &"

for i in 0 1; do
  PYTHONUNBUFFERED=1 uv run --with uroman python scripts/align_districts.py --device cpu --data $D/candidates_extra.parquet \
    --shard $i/2 --out $D/alignments_extra_$i.parquet > outputs/logs/align_extra_$i.log 2>&1 &
done
wait
uv run python -c "
import pandas as pd
parts = ['$D/alignments_0.parquet', '$D/alignments_1.parquet', '$D/alignments_extra_0.parquet', '$D/alignments_extra_1.parquet']
a = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
a.to_parquet('$D/alignments.parquet', index=False); print(len(a), 'tokens merged')"
uv run python scripts/measure_closure.py --workers 6 > outputs/logs/measure_closure.log 2>&1
tail -8 outputs/logs/measure_closure.log
uv run python scripts/analyse_gemination.py > outputs/logs/analyse_gemination.log 2>&1   # without recognizer doubling
tail -20 outputs/logs/analyse_gemination.log
echo "$(date +%H:%M) EXTRA_MEASURED"

until ssh pc "test -f $R/outputs/logs/extra_transcripts.done"; do sleep 120; done
for m in w2v2 zipa; do
  rsync -a pc:$R/outputs/transcripts/${m}_districts.parquet pc:$R/outputs/transcripts/${m}_districts_extra.parquet outputs/transcripts/
  uv run python -c "
import pandas as pd
a = pd.concat([pd.read_parquet('outputs/transcripts/${m}_districts.parquet'), pd.read_parquet('outputs/transcripts/${m}_districts_extra.parquet')], ignore_index=True)
a = a.drop_duplicates('audio_path'); a.to_parquet('outputs/transcripts/${m}_districts_all.parquet', index=False); print('$m', len(a), 'transcripts')"
  cp outputs/transcripts/${m}_districts_all.parquet outputs/transcripts/${m}_districts.parquet
done
uv run python scripts/analyse_gemination.py > outputs/logs/analyse_gemination.log 2>&1
tail -20 outputs/logs/analyse_gemination.log
echo "$(date +%H:%M) EXTRA_DONE"
