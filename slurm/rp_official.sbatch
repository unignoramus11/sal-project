#!/usr/bin/env bash
#SBATCH --job-name=rp_official
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/rp_official.sbatch <model> "<versions>"
#   released       the release as published, its own split
#   clean_prism    corrected version (deduplicated, consistent labels, fixed coordinates), released split
#   clean_speaker  corrected version, pseudo-speaker-disjoint split
# Needs data/vaani_hi_official/vaani_hi_official.parquet (docs/ada.md step 11). Frames are extracted once per job;
# seeds with results are skipped.
. slurm/common.sh
. slurm/frames.sh
R=results/official
SEEDS="42 43 44 45 46"
sfx() { case $1 in released) echo prism ;; clean_prism) echo prism_clean ;; clean_speaker) echo speaker_split_clean ;; esac; }
left=""
for V in $2; do [ -n "$(rp_todo "$1" official "-1" attn "$(sfx $V)" "$SEEDS")" ] && left=1; done
[ -n "$left" ] || { echo "all seeds done"; exit 0; }
use_frames "$1" official "-1"
for V in $2; do
  case "$V" in
    released)      args=(--dataset $R/meta_released.parquet --splits $R/splits_released.parquet --protocol prism) ;;
    clean_prism)   args=(--dataset $R/meta_clean.parquet --splits $R/splits_clean.parquet --protocol prism --tag _clean) ;;
    clean_speaker) args=(--dataset $R/meta_clean.parquet --splits $R/splits_clean.parquet --protocol speaker_split --tag _clean) ;;
  esac
  S=$(rp_todo "$1" official "-1" attn "$(sfx $V)" "$SEEDS")
  if [ -n "$S" ]; then
    python scripts/run_rp.py --features "$FEAT" "${args[@]}" --probe attn --layers -1 \
      --seeds $S --device cuda --in-memory ${RP_ARGS:-}
  fi
done
