#!/usr/bin/env bash
# Second half of the gemination chain: wait for both alignment shards (both on the Mac), merge them,
# measure closures on the Mac, pull the PC's district transcripts, run the analysis, export the audit clips.
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/vaani_districts
until [ -s $D/alignments_0.parquet ] && [ -s $D/alignments_1.parquet ]; do sleep 60; done
echo "$(date +%H:%M) both shards aligned"
uv run python -c "
import pandas as pd
a = pd.concat([pd.read_parquet('$D/alignments_0.parquet'), pd.read_parquet('$D/alignments_1.parquet')], ignore_index=True)
a.to_parquet('$D/alignments.parquet', index=False); print(len(a), 'tokens merged')"
echo "$(date +%H:%M) measuring"
uv run python scripts/measure_closure.py --workers 6 > outputs/logs/measure_closure.log 2>&1
tail -8 outputs/logs/measure_closure.log
rsync -a pc:~/developer/sal-project/outputs/transcripts/{w2v2,zipa}_districts.parquet outputs/transcripts/ 2>/dev/null
uv run python scripts/analyse_gemination.py > outputs/logs/analyse_gemination.log 2>&1
tail -20 outputs/logs/analyse_gemination.log
uv run python scripts/audit_tokens.py --n 50 > outputs/logs/audit_tokens.log 2>&1
tail -6 outputs/logs/audit_tokens.log
echo "$(date +%H:%M) GEMINATION_DONE"
