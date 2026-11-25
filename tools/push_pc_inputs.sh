#!/usr/bin/env bash
# Copy code and the small result tables a PC queue needs: the benchmark and official split tables, and every
# finished run and MDL file, so the PC skips what is already done. Never deletes or overwrites newer files there.
set -euo pipefail
cd "$(dirname "$0")/.."
bash tools/sync_pc.sh
ssh pc 'mkdir -p ~/developer/sal-project/results/{runs,mdl,benchmark,official}'
for d in benchmark official runs mdl; do rsync -az --update results/$d/ pc:~/developer/sal-project/results/$d/; done
echo "pushed results/{benchmark,official,runs,mdl} to the PC"
