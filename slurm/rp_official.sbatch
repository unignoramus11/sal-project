#!/usr/bin/env bash
#SBATCH --job-name=rp_official
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp_official.sbatch <model> <prism|speaker_split>
. slurm/common.sh
python scripts/run_rp.py --features "outputs/features/${1}_official" --dataset results/official/meta.parquet \
  --splits results/official/splits.parquet --probe attn --protocol "$2" --layers -1 \
  --seeds 42 43 44 45 46 --device cuda --in-memory
