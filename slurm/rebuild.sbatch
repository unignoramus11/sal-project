#!/usr/bin/env bash
#SBATCH --job-name=rebuild
#SBATCH --time=06:00:00
# Rebuild Vaani-Hi from Vaani (needs internet and `hf auth login` with Vaani terms accepted), fix the
# splits, and compare both checksums with the reference build made on 17 Oct 2026.
. slurm/common.sh
python scripts/build_vaani_hi.py --workers 8
python scripts/make_splits.py data/vaani_hi_rebuild/vaani_hi_rebuild.parquet
echo "dataset  $(sha256sum data/vaani_hi_rebuild/vaani_hi_rebuild.parquet | cut -d' ' -f1)"
echo "expected 07f6a8ce26e7b2b7b32bf1694d5dc27017d6e5fdcba5cd246100175f3bc32392"
echo "splits   $(sha256sum data/vaani_hi_rebuild/splits.parquet | cut -d' ' -f1)"
echo "expected 3cb5d3465db35281edeb37a01445d562189a15c44f4e438173544dd10ed51026"
