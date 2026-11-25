#!/usr/bin/env bash
# Fallback only: the transcript-probe, prosody and MDL GPU runs (40 jobs) on Ada. Since 25 Nov they run on the PC
# (tools/pc_november.sh), where every earlier BiGRU run was made. Every job skips finished runs.
set -euo pipefail
cd "$(dirname "$0")/.."
for T in all shared; do
  for M in w2v2 zipa; do
    bash slurm/submit.sh slurm/bench_gru.sbatch $M $T "0 1 2"
    bash slurm/submit.sh slurm/bench_gru.sbatch $M $T "3 4"
  done
done
for M in w2v2 zipa; do
  for spec in "char gru" "phone gru" "phone shuffle"; do
    set -- $spec
    for P in speaker_split prism; do
      bash slurm/submit.sh slurm/tp.sbatch $M $1 $2 $P all "42 43 44 45 46"
      bash slurm/submit.sh slurm/tp.sbatch $M $1 $2 $P generic "42 43 44 45 46"
    done
  done
  bash slurm/submit.sh slurm/tp_official.sbatch $M "42 43 44 45 46"
done
for P in speaker_split prism; do
  bash slurm/submit.sh slurm/prosody.sbatch $P all "42 43 44 45 46"
  for probe in gru shuffle; do bash slurm/submit.sh slurm/mdl.sbatch zipa phone $probe $P "0 1 2"; done
done
squeue -u "$USER"
