#!/usr/bin/env bash
#SBATCH --job-name=prosody
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/prosody.sbatch <prism|speaker_split> <all|generic> "<seeds>"
. slurm/common.sh
python scripts/run_prosody.py --protocol "$1" --prompts "$2" --seeds $3 --device cuda
