#!/usr/bin/env bash
#SBATCH --job-name=rp_check
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp_check.sbatch <model> <protocol> <generic|control> "<seeds>"
. slurm/common.sh
. slurm/frames.sh
if [ "$3" = generic ]; then extra="--prompts generic"; else extra="--control"; fi
use_frames "$1" rebuild "-1"
python scripts/run_rp.py --features "$FEAT" --probe "${5:-attn}" --protocol "$2" --layers -1 \
  --seeds $4 --device cuda --in-memory $extra ${RP_ARGS:-}
