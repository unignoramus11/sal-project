#!/usr/bin/env bash
#SBATCH --job-name=mdl
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/mdl.sbatch <w2v2|zipa> <char|phone> <gru|shuffle|unigram|bigram> <prism|speaker_split> "<seeds>"
# Online-coding description length (scripts/mdl.py); seeds with a results/mdl file are skipped.
. slurm/common.sh
S=""
for s in $5; do [ -f "results/mdl/mdl_${1}_rebuild_${2}_${3}_${4}_s$s.json" ] || S="$S $s"; done
[ -n "$S" ] || { echo "all seeds done"; exit 0; }
python scripts/mdl.py --transcripts "outputs/transcripts/${1}_rebuild.parquet" --tokens "$2" --probe "$3" \
  --protocol "$4" --seeds $S --device cuda
