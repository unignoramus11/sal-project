#!/usr/bin/env bash
# Copy results produced on the PC (results/) back to this checkout. Never deletes local files.
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -az pc:~/developer/sal-project/results/ results/
echo "pulled results from pc"
