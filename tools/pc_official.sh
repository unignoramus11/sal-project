#!/usr/bin/env bash
# PC: W2V2 transcription of the official Vaani-Hi (16 shards copied from the Mac), merged into one file,
# then PRiSM's char TP-GRU on PRiSM's own split (reproduction of the 45.7 Recall@1 anchor).
set -uo pipefail
cd "$(dirname "$0")/.."
. tools/prism_env.sh
export PYTHONUNBUFFERED=1
D=data/vaani_hi_official
mkdir -p outputs/transcripts/official
for f in $D/data/train-*.parquet; do
  n=$(basename "$f" .parquet)
  [ -f outputs/transcripts/official/w2v2_$n.parquet ] && continue
  echo "$(date +%H:%M) transcribe $n"
  python scripts/transcribe.py --model w2v2 --data "$f" --out outputs/transcripts/official/w2v2_$n.parquet --check 20 \
    > outputs/logs/transcribe_official_$n.log 2>&1 || echo "FAILED $n"
done
python - <<'PY'
import glob, pyarrow as pa, pyarrow.parquet as pq
t = pa.concat_tables([pq.read_table(p) for p in sorted(glob.glob("outputs/transcripts/official/w2v2_train-*.parquet"))])
pq.write_table(t, "outputs/transcripts/w2v2_official.parquet"); print("merged", t.num_rows)
PY
for seed in 42 43 44 45 46; do
  name=tp_w2v2_official_char_gru_prism_s$seed
  [ -f results/runs/$name/metrics.json ] && continue
  echo "$(date +%H:%M) start $name"
  python scripts/run_tp.py --transcripts outputs/transcripts/w2v2_official.parquet --dataset $D/meta.parquet \
    --splits $D/splits.parquet --tokens char --probe gru --protocol prism --seeds $seed --device cuda \
    > outputs/logs/$name.log 2>&1 && tail -n 1 outputs/logs/$name.log || echo "FAILED $name"
done
echo OFFICIAL_PC_DONE
