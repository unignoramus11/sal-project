#!/usr/bin/env bash
#SBATCH --job-name=rp_check
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp_check.sbatch <model> <protocol> <generic|control> "<seeds>" [probe]
# Topic check (shared pictures only) or control task (each speaker given another speaker's location), last layer.
# Probe defaults to attn (PRiSM's). Seeds with results are skipped.
. slurm/common.sh
. slurm/frames.sh
PROBE=${5:-attn}
if [ "$3" = generic ]; then extra="--prompts generic"; sfx="${2}_generic"; else extra="--control"; sfx="${2}_control"; fi
S=$(rp_todo "$1" rebuild "-1" "$PROBE" "$sfx" "$4")
[ -n "$S" ] || { echo "all seeds done"; exit 0; }
use_frames "$1" rebuild "-1"
python scripts/run_rp.py --features "$FEAT" --probe "$PROBE" --protocol "$2" --layers -1 \
  --seeds $S --device cuda --in-memory $extra ${RP_ARGS:-}
