#!/usr/bin/env bash
#SBATCH --job-name=extract
#SBATCH --time=12:00:00
# Usage: bash slurm/submit.sh slurm/extract.sbatch <w2v2|whisper|zipa> [frame layers...]
# Default: time-averaged vectors of every layer only (pooled.npy, at most 2.3 GB per model), which fit in /home2 and
# are all the layer sweep needs. Representation-probe jobs extract their frames into scratch themselves
# (slurm/frames.sh); pass layers here only to cache frames in /home2 on purpose (~15 GB per layer).
. slurm/common.sh
MODEL=$1; shift
python scripts/extract_features.py --model "$MODEL" --data data/vaani_hi_rebuild/vaani_hi_rebuild.parquet \
  --out-dir "outputs/features/${MODEL}_rebuild" --frame-layers "$@"
