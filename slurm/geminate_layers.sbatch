#!/usr/bin/env bash
#SBATCH --job-name=geminate_layers
#SBATCH --time=12:00:00
# Usage: bash slurm/submit.sh slurm/geminate_layers.sbatch <w2v2|whisper|zipa>
# Layer-wise geminate probe (RQ2). Needs results/gemination/layer_probe_tokens.parquet (pushed from the Mac) and
# the district audio, which the first job fetches once (~12 GB of Vaani shards streamed and filtered).
. slurm/common.sh
A=data/vaani_districts/candidates_audio.parquet
if [ ! -s $A ]; then
  python scripts/district_audio_files.py --workers 4      # needs internet on the node (else run on the login node)
fi
python scripts/token_frames.py --model "$1"
python scripts/geminate_layers.py --model "$1"
