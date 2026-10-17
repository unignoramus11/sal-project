#!/usr/bin/env bash
# Mid-evaluation RP baselines: PRiSM's RP-Attn on the last layer, 3 models x 2 protocols x 5 seeds.
cd "$(dirname "$0")/.."
for M in w2v2 whisper zipa; do
  for P in prism speaker_split; do
    bash slurm/submit.sh slurm/rp.sbatch "$M" attn "$P" "-1" "42 43 44 45 46"
  done
done
