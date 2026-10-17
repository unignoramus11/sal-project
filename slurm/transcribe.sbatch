#!/usr/bin/env bash
#SBATCH --job-name=transcribe
#SBATCH --time=06:00:00
# Usage: bash slurm/submit.sh slurm/transcribe.sbatch <w2v2|zipa>
# Greedy CTC transcription of the rebuilt Vaani-Hi; prints a hash of the transcripts to compare with the PC's.
. slurm/common.sh
python scripts/transcribe.py --model "$1" --data data/vaani_hi_rebuild/vaani_hi_rebuild.parquet \
  --out "outputs/transcripts/${1}_rebuild.parquet"
python - "$1" <<'PY'
import hashlib, sys
import pandas as pd
m = sys.argv[1]
expected = {"w2v2": "93b2b814b23b234c1a4dd36fa155e0fa3933ecf0e1c0be08a86cbbc5b20b33e8",
            "zipa": "d4543d638c02f50046b362f6278a196b7fe1a829ddfff6ec722aedd5e9e4615f"}[m]
t = pd.read_parquet(f"outputs/transcripts/{m}_rebuild.parquet", columns=["metadata_idx", "processed_transcript"])
t = t.sort_values("metadata_idx")
h = hashlib.sha256("\n".join(t.processed_transcript).encode()).hexdigest()
print("transcripts", len(t), h)
print("expected   ", 22551, expected, "MATCH" if h == expected else "DIFFERENT (send the log; GPU kernels can flip rare argmax ties)")
PY
