#!/usr/bin/env bash
# Fallback only: the CPU benchmark probes (bag probes, ECAPA, prosody statistics; 10 jobs) on Ada. Since 25 Nov they
# run on the Mac (tools/bench_mac.sh), next to every earlier bag-probe run. Every job skips finished runs.
set -euo pipefail
cd "$(dirname "$0")/.."
for T in all shared; do
  for F in 0 1 2 3 4; do GPU=0 CPUS=${BENCH_CPUS:-16} bash slurm/submit.sh slurm/bench_light.sbatch $T $F; done
done
squeue -u "$USER"
