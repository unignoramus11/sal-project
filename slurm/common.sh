# Sourced at the start of every job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. tools/prism_env.sh
# Keep Hugging Face caches accessible on ADA compute nodes.
export HF_HOME="$PWD/envs/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export XDG_CACHE_HOME="$HF_HOME/xdg"
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_ASSETS_CACHE" "$XDG_CACHE_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "host $(hostname)  job ${SLURM_JOB_ID:-local}  $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

# todo <run name without _s<seed>> <seeds...>: print the seeds whose results/runs/<name>_s<seed>/metrics.json is
# missing, so a resubmitted job only runs what is left.
todo() {
  local name=$1 s out=""; shift
  for s in "$@"; do [ -f "results/runs/${name}_s$s/metrics.json" ] || out="$out $s"; done
  echo $out
}
