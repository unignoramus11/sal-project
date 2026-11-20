#!/usr/bin/env bash
#SBATCH --job-name=prosody
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/prosody.sbatch <prism|speaker_split> <all|generic> "<seeds>"
# Prosody BiGRU on F0, voicing and intensity contours; seeds that already have results are skipped.
. slurm/common.sh
gen=$([ "$2" = generic ] && echo _generic || true)
S=$(todo "prosody_gru_${1}${gen}" $3)
[ -n "$S" ] || { echo "all seeds done"; exit 0; }
python scripts/run_prosody.py --protocol "$1" --prompts "$2" --seeds $S --device cuda
