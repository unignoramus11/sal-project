#!/usr/bin/env bash
# Copy code and the small tables a PC queue needs: the rebuild's split and prompt tables, the benchmark and official
# split tables, and every finished run and MDL file, so the PC skips what is already done. Never deletes or
# overwrites newer files there. The transcripts, prosody features and the 4.6 GB dataset file are copied once by hand;
# tools/pc_november.sh checks for every input before it starts.
set -euo pipefail
cd "$(dirname "$0")/.."
bash tools/sync_pc.sh
ssh pc 'mkdir -p ~/developer/sal-project/{results/{runs,mdl,benchmark,official},data/vaani_hi_rebuild}'
rsync -az --update data/vaani_hi_rebuild/{splits,prompts}.parquet pc:~/developer/sal-project/data/vaani_hi_rebuild/
for d in benchmark official runs mdl; do rsync -az --update results/$d/ pc:~/developer/sal-project/results/$d/; done
echo "pushed data/vaani_hi_rebuild/{splits,prompts}.parquet and results/{benchmark,official,runs,mdl} to the PC"
