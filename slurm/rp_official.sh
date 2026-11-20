#!/usr/bin/env bash
# PRiSM's RP-Attn (last layer, 5 seeds) on the official Vaani-Hi: 3 models x {released, clean_prism, clean_speaker}.
# 6 jobs: the released version alone, and both corrected versions together (one frame extraction each).
cd "$(dirname "$0")/.."
for M in w2v2 whisper zipa; do
  bash slurm/submit.sh slurm/rp_official.sbatch "$M" released
  bash slurm/submit.sh slurm/rp_official.sbatch "$M" "clean_prism clean_speaker"
done
