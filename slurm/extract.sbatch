#!/usr/bin/env bash
#SBATCH --job-name=extract
#SBATCH --time=12:00:00
# Usage: bash slurm/submit.sh slurm/extract.sbatch <w2v2|whisper|zipa> [frame layers...]
# Default frame layer: the last one (-1). Pooled vectors are always written for every layer.
. slurm/common.sh
MODEL=$1; shift
LAYERS=${*:--1}
python scripts/extract_features.py --model "$MODEL" --data data/vaani_hi_rebuild/vaani_hi_rebuild.parquet \
  --out-dir "outputs/features/${MODEL}_rebuild" --frame-layers $LAYERS
