#!/usr/bin/env bash
# Final-evaluation transcript, prosody and benchmark runs, moved from the Mac and the PC so every probe runs on Ada
# (docs/ada.md step 13). Submit once: bash slurm/november.sh. Every job skips runs that already have results,
# so resubmitting after a failure only runs what is left.
#   benchmark   10 CPU jobs (bag probes, ECAPA, prosody statistics) and 8 GPU jobs (transcript BiGRUs)
#   seeds       transcript BiGRUs on the rebuild up to 5 seeds, both splits, all and shared pictures (24 jobs)
#   prosody     prosody BiGRU up to 5 seeds, both splits (2 jobs)
#   release     transcript BiGRUs on PRiSM's release, W2V2 and ZIPA, 5 seeds (2 jobs)
#   mdl         description length of the ZIPA BiGRU and shuffled BiGRU, both splits (4 jobs)
set -euo pipefail
cd "$(dirname "$0")/.."
for T in all shared; do
  for F in 0 1 2 3 4; do GPU=0 CPUS=${BENCH_CPUS:-16} bash slurm/submit.sh slurm/bench_light.sbatch $T $F; done
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
