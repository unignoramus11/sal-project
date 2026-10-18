#!/usr/bin/env bash
# Lexical voiceless geminates (अच्छा, बच्चे, मिट्टी, ...) in the district audio, after the second gemination pass:
# align them (Mac CPU) and measure their closures. Uses: validity check (lexical geminates must show longer
# closures than single stops) and the positive class of the layer-wise geminate probe.
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/vaani_districts
until grep -q "EXTRA_DONE\|no new shards" outputs/logs/gemination_extra.log 2>/dev/null; do sleep 120; done
PYTHONUNBUFFERED=1 uv run --with uroman python scripts/align_districts.py --device cpu --targets geminate \
  --out $D/alignments_geminate.parquet > outputs/logs/align_geminate.log 2>&1
tail -1 outputs/logs/align_geminate.log
uv run python scripts/measure_closure.py --workers 6 --alignments $D/alignments_geminate.parquet \
  --out $D/closures_geminate.parquet > outputs/logs/measure_geminate.log 2>&1
tail -6 outputs/logs/measure_geminate.log
echo "$(date +%H:%M) GEMINATE_PASS_DONE"
