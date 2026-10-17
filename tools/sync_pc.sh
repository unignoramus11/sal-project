#!/usr/bin/env bash
# Push code (not data, results, outputs, environments or notes) to the PC. Excludes are anchored to
# the root. Results flow the other way: tools/pull_results.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -az --delete \
  --exclude '/.venv' --exclude '/data/' --exclude '/internal/' --exclude '/external/' \
  --exclude '/outputs/' --exclude '/.git/' --exclude '/proposal/' --exclude '/envs/' \
  --exclude '/results/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
  ./ pc:~/developer/sal-project/
echo "synced to pc:~/developer/sal-project"
