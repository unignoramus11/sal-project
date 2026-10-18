#!/usr/bin/env bash
#SBATCH --job-name=rp_check
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp_check.sbatch <model> <protocol> <generic|control> "<seeds>"
. slurm/common.sh
if [ "$3" = generic ]; then extra="--prompts generic"; else extra="--control"; fi
python scripts/run_rp.py --features "outputs/features/${1}_rebuild" --probe attn --protocol "$2" --layers -1 \
  --seeds $4 --device cuda --in-memory $extra
