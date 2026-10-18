#!/usr/bin/env bash
# Representation-probe checks that mirror the transcript-probe analyses (RQ0), PRiSM's RP-Attn on the last layer:
#   topic check   trained and tested on shared-picture utterances only (--prompts generic), 5 seeds
#   control task  each speaker labelled with another speaker's location (--control), 2 seeds; under the random
#                 split a score above the control majority means the probe recognises speakers
# for each model under both splits: 3 models x 2 splits x 2 checks = 12 jobs. Needs the last-layer frames
# (step 3 of docs/ada.md). Usage: bash slurm/rp_extra.sh
set -euo pipefail
cd "$(dirname "$0")/.."
for M in w2v2 whisper zipa; do
  for P in speaker_split prism; do
    bash slurm/submit.sh slurm/rp_check.sbatch "$M" "$P" generic "42 43 44 45 46"
    bash slurm/submit.sh slurm/rp_check.sbatch "$M" "$P" control "42 43"
  done
done
