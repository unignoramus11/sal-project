#!/usr/bin/env bash
#SBATCH --job-name=tp
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/tp.sbatch <w2v2|zipa> <char|phone> <gru|shuffle> <prism|speaker_split> <all|generic> "<seeds>"
# Transcript-probe runs on the rebuild (light GPU jobs); needs outputs/transcripts/<model>_rebuild.parquet.
# Seeds that already have results are skipped.
. slurm/common.sh
gen=$([ "$5" = generic ] && echo _generic || true)
S=$(todo "tp_${1}_rebuild_${2}_${3}_${4}${gen}" $6)
[ -n "$S" ] || { echo "all seeds done"; exit 0; }
python scripts/run_tp.py --transcripts "outputs/transcripts/${1}_rebuild.parquet" --tokens "$2" --probe "$3" \
  --protocol "$4" --prompts "$5" --seeds $S --device cuda
