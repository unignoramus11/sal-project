#!/usr/bin/env bash
# Light GPU jobs that take load off the PC: transcription (and its reproducibility hash), prosody contours,
# then, once those finish, the shared-picture TP-GRU runs (topic check) and the prosody BiGRU runs.
# Needs the rebuilt dataset (step 2 of docs/ada.md). Submit once: bash slurm/light_jobs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
jid() { awk '{print $NF}'; }
T1=$(bash slurm/submit.sh slurm/transcribe.sbatch w2v2 | jid)
T2=$(bash slurm/submit.sh slurm/transcribe.sbatch zipa | jid)
PF=$(GPU=0 bash slurm/submit.sh slurm/prosody_features.sbatch | jid)
for M in w2v2 zipa; do
  dep=$([ $M = w2v2 ] && echo $T1 || echo $T2)
  for spec in "char gru" "phone gru" "phone shuffle"; do
    set -- $spec
    for P in speaker_split prism; do
      bash slurm/submit.sh --dependency=afterok:$dep slurm/tp.sbatch $M $1 $2 $P generic "42 43 44 45 46"
    done
  done
done
for P in speaker_split prism; do
  bash slurm/submit.sh --dependency=afterok:$PF slurm/prosody.sbatch $P all "42 43 44 45 46"
done
squeue -u "$USER"
