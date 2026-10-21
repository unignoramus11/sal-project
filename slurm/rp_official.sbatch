#!/usr/bin/env bash
#SBATCH --job-name=rp_official
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp_official.sbatch <model> <released|clean_prism|clean_speaker>
#   released       the release as published, its own split
#   clean_prism    corrected version (deduplicated, consistent labels, fixed coordinates), released split
#   clean_speaker  corrected version, pseudo-speaker-disjoint split
. slurm/common.sh
R=results/official
case "$2" in
  released)      args=(--dataset $R/meta_released.parquet --splits $R/splits_released.parquet --protocol prism) ;;
  clean_prism)   args=(--dataset $R/meta_clean.parquet --splits $R/splits_clean.parquet --protocol prism --tag _clean) ;;
  clean_speaker) args=(--dataset $R/meta_clean.parquet --splits $R/splits_clean.parquet --protocol speaker_split --tag _clean) ;;
esac
python scripts/run_rp.py --features "outputs/features/${1}_official" "${args[@]}" --probe attn --layers -1 \
  --seeds 42 43 44 45 46 --device cuda --in-memory
