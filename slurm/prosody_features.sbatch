#!/usr/bin/env bash
#SBATCH --job-name=prosody_features
#SBATCH --time=06:00:00
# Usage: GPU=0 bash slurm/submit.sh slurm/prosody_features.sbatch     (CPU job, Praat contours for every utterance)
. slurm/common.sh
python scripts/prosody_features.py --workers "${SLURM_CPUS_PER_TASK:-10}"
