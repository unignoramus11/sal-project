#!/usr/bin/env bash
# Create envs/prism on a Linux machine with an NVIDIA GPU (the PC, or Ada).
# It reproduces the environment the transcripts were made with: Python 3.10, torch 2.6.0+cu126, k2,
# lhotse 1.32.1, PRiSM's ESPnet fork, icefall at a pinned commit, plus this project's package.
# Usage (from the repo root):  bash tools/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PRISM_COMMIT=2e7b754
ICEFALL_COMMIT=1c388ad9be94c0ee9de76f7a6ed45b1cff677d13

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
# keep uv's cache next to the repo (home quotas on clusters are small)
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/envs/.uv-cache}"
# The CUDA wheels are large and NVIDIA's server can stall for minutes: wait longer, download fewer at once, and
# retry the whole install (finished wheels stay in the cache, so each try resumes where the last one stopped).
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}" UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
retry() { local k; for k in 1 2 3; do "$@" && return 0; echo "install failed (try $k of 3); retrying in $((30 * k)) s"; sleep $((30 * k)); done; return 1; }
mkdir -p envs external

[ -d external/prism ] || git clone -q https://github.com/changelinglab/prism.git external/prism
git -C external/prism checkout -q "$PRISM_COMMIT"

[ -d envs/prism ] || uv venv -q -p 3.10 envs/prism
# The lock is the exact package set of the working PC environment. Install it as-is (--no-deps): the
# ESPnet fork's metadata conflicts with versions that are known to work together (e.g. pypinyin).
retry uv pip install -q --python envs/prism/bin/python --no-deps \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --find-links https://k2-fsa.github.io/k2/cuda.html \
  -r tools/prism-requirements.lock

[ -d envs/icefall ] || git clone -q https://github.com/k2-fsa/icefall envs/icefall
git -C envs/icefall checkout -q "$ICEFALL_COMMIT"
retry uv pip install -q --python envs/prism/bin/python --no-deps -e envs/icefall

# this project (analysis extras that the PRiSM environment lacks)
retry uv pip install -q --python envs/prism/bin/python scipy scikit-learn duckdb panphon praat-parselmouth matplotlib
uv pip install -q --python envs/prism/bin/python --no-deps -e .
echo "declared dependency conflicts (informational; the same set runs on the PC):"
uv pip check --python envs/prism/bin/python || true

. tools/prism_env.sh
python - <<'PY'
import torch, k2, lhotse, transformers, sys
sys.path.insert(0, "external/prism")
from src.model.wav2vec2phoneme.builders import build_wav2vec2phoneme_inference  # noqa: F401
from src.model.zipa.zipformer_crctc.builders import build_zipactc_inference     # noqa: F401
from phonemizer.backend import EspeakBackend
import sal.probes.models  # noqa: F401
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("k2", k2.__dev_version__, "lhotse", lhotse.__version__, "transformers", transformers.__version__)
print("espeak", EspeakBackend.is_available())
print("environment OK")
PY
