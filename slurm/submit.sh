#!/usr/bin/env bash
# Submit a job with the account/partition/GPU settings from slurm/ada.env.
#   bash slurm/submit.sh <job.sbatch> [args passed to the job...]
#   GPU=0 bash slurm/submit.sh slurm/rebuild.sbatch      # CPU-only job
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f slurm/ada.env ] || { echo "create slurm/ada.env from slurm/ada.env.example first"; exit 1; }
. slurm/ada.env
mkdir -p outputs/slurm
flags=(--cpus-per-task="${CPUS:-10}" --mem="${MEM:-60G}" --output=outputs/slurm/%x_%j.out)
[ -n "${ACCOUNT:-}" ] && flags+=(-A "$ACCOUNT")
[ -n "${PARTITION:-}" ] && flags+=(-p "$PARTITION")
[ "${GPU:-1}" = "1" ] && flags+=(--gres="${GPU_GRES:-gpu:1}")
sbatch "${flags[@]}" "$@"
