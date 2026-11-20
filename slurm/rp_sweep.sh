#!/usr/bin/env bash
# Main representation-probe sweep for one model (RQ1: does order help the RP, at matched size, with the layer
# choice held fixed?). Needs the layer sweep CSV for this model (step 4 of docs/ada.md).
#   1. reads the top 3 layers by validation km (speaker-disjoint split) from results/layer_sweep/
#   2. submits the probe grid; every job extracts its own frames into node-local scratch (slurm/frames.sh):
#        probes   attn (PRiSM), attn_pe, gru, transformer, chunkshuffle
#        layers   last layer (-1), and the learned weighted sum of the top 3 layers
#        splits   speaker_split first, then prism
#        seeds    42-46 in one job (5 runs, ~3-5 h, plus 1-2 h of extraction)
#   = 19 jobs (95 runs) per model; last-layer RP-Attn comes from slurm/rp_baselines.sh.
# Usage: bash slurm/rp_sweep.sh <w2v2|whisper|zipa>          (dry run: DRY=1 bash slurm/rp_sweep.sh w2v2)
set -euo pipefail
cd "$(dirname "$0")/.."
M=$1
. tools/prism_env.sh
TOP=$(python slurm/top_layers.py "$M" 3)
echo "$M: top layers by validation km: $TOP"
run() { if [ "${DRY:-0}" = 1 ]; then echo "would run: $*"; else "$@"; fi; }
for P in speaker_split prism; do
  for probe in attn attn_pe gru transformer chunkshuffle; do
    # last-layer RP-Attn is already run by slurm/rp_baselines.sh (step 5)
    [ "$probe" = attn ] || run bash slurm/submit.sh slurm/rp.sbatch "$M" "$probe" "$P" "-1" "42 43 44 45 46"
    # three layers in RAM (~15 GB each for W2V2/ZIPA): ask for more memory
    MEM=${MEM_WEIGHTED:-100G} run bash slurm/submit.sh slurm/rp.sbatch "$M" "$probe" "$P" "$TOP" "42 43 44 45 46"
  done
done
