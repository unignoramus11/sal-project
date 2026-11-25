#!/usr/bin/env bash
#SBATCH --job-name=rp
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp.sbatch <model> <probe> <protocol> "<layers>" "<seeds>"
# e.g.   bash slurm/submit.sh slurm/rp.sbatch w2v2 attn speaker_split "-1" "42 43 44 45 46"
# Seeds with results are skipped; frames come from node-local scratch (slurm/frames.sh, ~1-2 h per job).
. slurm/common.sh
. slurm/frames.sh
S=$(rp_todo "$1" rebuild "$4" "$2" "$3" "$5")
[ -n "$S" ] || { echo "all seeds done"; exit 0; }
use_frames "$1" rebuild "$4"
python scripts/run_rp.py --features "$FEAT" --probe "$2" --protocol "$3" \
  --layers $4 --seeds $S --device cuda --in-memory ${RP_ARGS:-}
