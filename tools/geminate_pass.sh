#!/usr/bin/env bash
# Lexical voiceless geminates (अच्छा, बच्चे, मिट्टी, ...) in the district audio, after the second gemination pass:
# align them (Mac CPU) and measure their closures, then export and push the token table for the layer-wise
# geminate probe (docs/ada.md step 10). Lexical geminates must also show longer closures than single stops,
# which checks the measurement.
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
uv run python scripts/geminate_layers.py --export-tokens > outputs/logs/export_tokens.log 2>&1
tail -3 outputs/logs/export_tokens.log
git add results/gemination/layer_probe_tokens.parquet && git commit -qm "Add token table for the layer-wise geminate probe" \
  && timeout 120 git push -q
uv run python scripts/analyse_gemination.py > outputs/logs/analyse_gemination.log 2>&1   # adds the validity check
grep -A1 "validity" outputs/logs/analyse_gemination.log | head -3
echo "$(date +%H:%M) GEMINATE_PASS_DONE"
