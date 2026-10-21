#!/usr/bin/env bash
# PRiSM's RP-Attn (last layer) on the official Vaani-Hi: 3 models x {released split, speaker-disjoint split of the
# corrected version} x 5 seeds. Needs outputs/features/<model>_official (slurm/extract_official.sbatch) and the
# versioned split tables in results/official/ (docs/ada.md step 11).
cd "$(dirname "$0")/.."
for M in w2v2 whisper zipa; do
  for P in prism speaker_split; do
    bash slurm/submit.sh slurm/rp_official.sbatch "$M" "$P"
  done
done
