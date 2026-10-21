#!/usr/bin/env bash
# PC (CPU): ZIPA bag probes on the official Vaani-Hi after tools/pc_official_zipa.sh: {char, phone} x {unigram, bigram}
# x {released split, corrected released split, corrected pseudo-speaker split} x 5 seeds, 8 runs at a time.
set -uo pipefail
cd "$(dirname "$0")/.."
. tools/prism_env.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
until grep -q OFFICIAL_ZIPA_DONE outputs/logs/pc_official_zipa.log 2>/dev/null; do sleep 120; done
R=results/official; T=outputs/transcripts/zipa_official.parquet
jobs=/tmp/zipa_bag_jobs.txt; : > $jobs
for s in 42 43 44 45 46; do for tok in phone char; do for probe in bigram unigram; do
  [ -f results/runs/tp_zipa_official_${tok}_${probe}_prism_s$s/metrics.json ] || \
    echo "python scripts/run_tp.py --transcripts $T --dataset $R/meta_released.parquet --splits $R/splits_released.parquet --tokens $tok --probe $probe --protocol prism --seeds $s" >> $jobs
  for P in prism speaker_split; do
    [ -f results/runs/tp_zipa_official_${tok}_${probe}_${P}_clean_s$s/metrics.json ] || \
      echo "python scripts/run_tp.py --transcripts $T --dataset $R/meta_clean.parquet --splits $R/splits_clean.parquet --tokens $tok --probe $probe --protocol $P --tag _clean --seeds $s" >> $jobs
  done
done; done; done
echo "$(wc -l < $jobs) runs"
tr '\n' '\0' < $jobs | xargs -0 -P 8 -n 1 sh -c 'eval "$1" 2>&1 | grep -E "^tp_|Error"' _
echo ZIPA_BAGS_DONE
