#!/usr/bin/env bash
#SBATCH --job-name=rp
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp.sbatch <model> <probe> <protocol> "<layers>" "<seeds>"
# e.g.   bash slurm/submit.sh slurm/rp.sbatch w2v2 attn speaker_split "-1" "42 43 44 45 46"
# Frames come from node-local scratch (slurm/frames.sh); each job extracts the layers it needs (~1-2 h).
. slurm/common.sh
. slurm/frames.sh
use_frames "$1" rebuild "$4"
python scripts/run_rp.py --features "$FEAT" --probe "$2" --protocol "$3" \
  --layers $4 --seeds $5 --device cuda --in-memory ${RP_ARGS:-}
