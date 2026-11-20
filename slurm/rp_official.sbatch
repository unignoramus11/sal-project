#!/usr/bin/env bash
#SBATCH --job-name=rp_official
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp_official.sbatch <model> "<versions>"
#   released       the release as published, its own split
#   clean_prism    corrected version (deduplicated, consistent labels, fixed coordinates), released split
#   clean_speaker  corrected version, pseudo-speaker-disjoint split
# Needs data/vaani_hi_official/vaani_hi_official.parquet (docs/ada.md step 11). Frames are extracted once per job.
. slurm/common.sh
. slurm/frames.sh
R=results/official
use_frames "$1" official "-1"
for V in $2; do
  case "$V" in
    released)      args=(--dataset $R/meta_released.parquet --splits $R/splits_released.parquet --protocol prism) ;;
    clean_prism)   args=(--dataset $R/meta_clean.parquet --splits $R/splits_clean.parquet --protocol prism --tag _clean) ;;
    clean_speaker) args=(--dataset $R/meta_clean.parquet --splits $R/splits_clean.parquet --protocol speaker_split --tag _clean) ;;
  esac
  python scripts/run_rp.py --features "$FEAT" "${args[@]}" --probe attn --layers -1 \
    --seeds 42 43 44 45 46 --device cuda --in-memory ${RP_ARGS:-}
done
