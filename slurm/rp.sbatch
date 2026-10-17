#!/usr/bin/env bash
#SBATCH --job-name=rp
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp.sbatch <model> <probe> <protocol> "<layers>" "<seeds>"
# e.g.   bash slurm/submit.sh slurm/rp.sbatch w2v2 attn speaker_split "-1" "42 43 44 45 46"
. slurm/common.sh
python scripts/run_rp.py --features "outputs/features/${1}_rebuild" --probe "$2" --protocol "$3" \
  --layers $4 --seeds $5 --device cuda --in-memory
