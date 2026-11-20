#!/usr/bin/env bash
#SBATCH --job-name=tp_official
#SBATCH --time=24:00:00
# Usage: bash slurm/submit.sh slurm/tp_official.sbatch <w2v2|zipa> "<seeds>"
# PRiSM's transcript BiGRU on PRiSM's release (characters and phones) and the shuffled-phone BiGRU, on the release as
# published, the corrected version under the released split, and the corrected version with pseudo-speakers.
# Needs outputs/transcripts/<model>_official.parquet (sent by Mohit) and results/official/*.parquet (in the repo).
. slurm/common.sh
R=results/official
for spec in "char gru" "phone gru" "phone shuffle"; do
  set -- "$1" "$2" $spec
  for V in released clean_prism clean_speaker; do
    case $V in
      released)      a=(--dataset $R/meta_released.parquet --splits $R/splits_released.parquet --protocol prism); n=prism ;;
      clean_prism)   a=(--dataset $R/meta_clean.parquet --splits $R/splits_clean.parquet --protocol prism --tag _clean); n=prism_clean ;;
      clean_speaker) a=(--dataset $R/meta_clean.parquet --splits $R/splits_clean.parquet --protocol speaker_split --tag _clean); n=speaker_split_clean ;;
    esac
    S=$(todo "tp_${1}_official_${3}_${4}_${n}" $2)
    if [ -n "$S" ]; then
      python scripts/run_tp.py --transcripts "outputs/transcripts/${1}_official.parquet" "${a[@]}" \
        --tokens "$3" --probe "$4" --seeds $S --device cuda
    fi
  done
done
