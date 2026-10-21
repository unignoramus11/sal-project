#!/usr/bin/env bash
# PC: ZIPA transcription of the official Vaani-Hi after the TP-GRU queues (for ZIPA bag probes on the Mac).
set -uo pipefail
cd "$(dirname "$0")/.."
. tools/prism_env.sh
export PYTHONUNBUFFERED=1
until grep -q OFFICIAL_CLEAN_DONE outputs/logs/pc_official_clean.log 2>/dev/null && grep -q OFFICIAL_CLEAN_DONE outputs/logs/pc_official_clean2.log 2>/dev/null; do sleep 120; done
for f in data/vaani_hi_official/data/train-*.parquet; do
  n=$(basename "$f" .parquet)
  [ -f outputs/transcripts/official/zipa_$n.parquet ] && continue
  python scripts/transcribe.py --model zipa --data "$f" --out outputs/transcripts/official/zipa_$n.parquet --check 20 \
    > outputs/logs/transcribe_official_zipa_$n.log 2>&1 || echo "FAILED $n"
done
python - <<'PY'
import glob, pyarrow as pa, pyarrow.parquet as pq
t = pa.concat_tables([pq.read_table(p) for p in sorted(glob.glob("outputs/transcripts/official/zipa_train-*.parquet"))])
pq.write_table(t, "outputs/transcripts/zipa_official.parquet"); print("merged", t.num_rows)
PY
echo OFFICIAL_ZIPA_DONE
