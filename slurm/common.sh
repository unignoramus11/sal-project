# Sourced at the start of every job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. tools/prism_env.sh
export HF_HOME="${HF_HOME:-$PWD/envs/hf}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "host $(hostname)  job ${SLURM_JOB_ID:-local}  $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
