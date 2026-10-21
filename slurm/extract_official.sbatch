#!/usr/bin/env bash
#SBATCH --job-name=extract_official
#SBATCH --time=12:00:00
# Usage: bash slurm/submit.sh slurm/extract_official.sbatch <w2v2|whisper|zipa> [frame layers...]
# Features of the official Vaani-Hi (needs data/vaani_hi_official/vaani_hi_official.parquet, docs/ada.md step 11).
. slurm/common.sh
MODEL=$1; shift
LAYERS=${*:--1}
python scripts/extract_features.py --model "$MODEL" --data data/vaani_hi_official/vaani_hi_official.parquet \
  --out-dir "outputs/features/${MODEL}_official" --frame-layers $LAYERS
