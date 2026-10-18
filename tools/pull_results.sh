#!/usr/bin/env bash
# Copy run directories produced on the PC (results/runs/) back to this checkout. Never deletes local files and
# never replaces a local file that is newer: tables, figures and baselines are made on the Mac, and the PC's
# copies of them are stale.
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -az --update pc:~/developer/sal-project/results/runs/ results/runs/
# per-run MDL files only; the summary table is rebuilt on the Mac from all of them (scripts/mdl.py)
rsync -az --update --include='mdl_*_s*.json' --exclude='*' pc:~/developer/sal-project/results/mdl/ results/mdl/
