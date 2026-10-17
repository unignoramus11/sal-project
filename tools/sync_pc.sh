#!/usr/bin/env bash
# Push code (not data, outputs, environments or notes) to the PC. Excludes are anchored to the root.
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -az --delete \
  --exclude '/.venv' --exclude '/data/' --exclude '/internal/' --exclude '/external/' \
  --exclude '/outputs/' --exclude '/.git/' --exclude '/proposal/' --exclude '/envs/' \
  --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '/CLAUDE.md' \
  ./ pc:~/developer/sal-project/
echo "synced to pc:~/developer/sal-project"
