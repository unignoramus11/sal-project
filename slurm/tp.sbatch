#!/usr/bin/env bash
#SBATCH --job-name=tp
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/tp.sbatch <w2v2|zipa> <char|phone> <gru|shuffle> <prism|speaker_split> <all|generic> "<seeds>"
# Transcript-probe runs (light GPU jobs); needs outputs/transcripts/<model>_rebuild.parquet (slurm/transcribe.sbatch).
. slurm/common.sh
python scripts/run_tp.py --transcripts "outputs/transcripts/${1}_rebuild.parquet" --tokens "$2" --probe "$3" \
  --protocol "$4" --prompts "$5" --seeds $6 --device cuda
