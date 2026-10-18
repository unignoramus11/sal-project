#!/usr/bin/env bash
# Add the district shards fetched after the first gemination pass: wait for the download and the first pass,
# rebuild candidates_audio.parquet from all parts, align only the new shards' utterances (2 Mac shards), merge
# with the first-pass alignments, then rerun closure measurement and analysis. The audit set is not re-drawn.
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/vaani_districts
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
uv run python scripts/analyse_gemination.py > outputs/logs/analyse_gemination.log 2>&1
tail -20 outputs/logs/analyse_gemination.log
echo "$(date +%H:%M) EXTRA_DONE"
