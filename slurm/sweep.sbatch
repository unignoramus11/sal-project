#!/usr/bin/env bash
#SBATCH --job-name=sweep
#SBATCH --time=08:00:00
# Usage: bash slurm/submit.sh slurm/sweep.sbatch <w2v2|whisper|zipa>
. slurm/common.sh
for P in speaker_split prism; do
  python scripts/layer_sweep.py --features "outputs/features/${1}_rebuild" --protocol $P --device cuda
done
