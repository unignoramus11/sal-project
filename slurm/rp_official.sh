#!/usr/bin/env bash
# PRiSM's RP-Attn (last layer, 5 seeds) on the official Vaani-Hi: 3 models x {released, clean_prism, clean_speaker}
# = 9 jobs. Needs outputs/features/<model>_official (slurm/extract_official.sbatch) and results/official/*.parquet.
cd "$(dirname "$0")/.."
for M in w2v2 whisper zipa; do
  for V in released clean_prism clean_speaker; do
    bash slurm/submit.sh slurm/rp_official.sbatch "$M" "$V"
  done
done
