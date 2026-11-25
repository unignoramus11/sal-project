#!/usr/bin/env bash
# The representation-probe plan, W2V2 first, in tiers ordered by what the final report needs. Submit one tier at a
# time: bash slurm/rp_plan.sh <tier>   (DRY=1 prints the jobs instead of submitting them).
# Every job skips seeds that already have results, so a tier that overlaps finished runs only submits what is left.
# Each job extracts its frames into node-local scratch first (1-2 h; slurm/frames.sh). L24 = W2V2's last layer.
#
#   1  RQ1, new speakers: does the RP use order?  W2V2 L24, speaker-disjoint split, 5 seeds:
#      attn (PRiSM's, order-free), gru (order-aware), chunkshuffle (gru on 200 ms chunks in random order),
#      attn_pe (attention with positions), transformer                                             5 jobs
#   2  The same on the cross-validated benchmark: look-alike speaker codes merged, every speaker tested once
#      (34.8% of the speaker-disjoint split's test utterances have a look-alike code in training).
#      W2V2 L24, attn / gru / chunkshuffle, 3 seeds; all track (2 jobs per probe), shared track (1 job per probe)   9 jobs
#   3  Shared speakers: W2V2 L24, PRiSM's random split, attn / gru / chunkshuffle, 5 seeds                3 jobs
#   4  Speakers and topic: control task (PRiSM's split, 2 seeds) and shared pictures only (new speakers,
#      5 seeds), attn and gru                                                                         4 jobs
#   5  Layers: learned weighted sum of W2V2's top 3 layers by validation km (step 4), speaker-disjoint split,
#      attn / gru / chunkshuffle, 5 seeds                                                              3 jobs
#   6  Other recognizers: ZIPA and Whisper last layer, attn and gru, both splits, 5 seeds               8 jobs
#   7  PRiSM's release (needs docs/ada.md step 11's download): W2V2 RP-Attn as released and corrected   2 jobs
set -euo pipefail
cd "$(dirname "$0")/.."
sub() { if [ "${DRY:-0}" = 1 ]; then echo "would submit: $*"; else bash slurm/submit.sh "$@"; fi; }
S5="42 43 44 45 46"
case "${1:?usage: bash slurm/rp_plan.sh <tier 1-7>}" in
  1) for p in attn gru chunkshuffle attn_pe transformer; do sub slurm/rp.sbatch w2v2 $p speaker_split "-1" "$S5"; done ;;
  2) for p in attn gru chunkshuffle; do
       sub slurm/rp_benchmark.sbatch w2v2 all "0 1 2" $p
       sub slurm/rp_benchmark.sbatch w2v2 all "3 4" $p
       sub slurm/rp_benchmark.sbatch w2v2 shared "0 1 2 3 4" $p
     done ;;
  3) for p in attn gru chunkshuffle; do sub slurm/rp.sbatch w2v2 $p prism "-1" "$S5"; done ;;
  4) for p in attn gru; do
       sub slurm/rp_check.sbatch w2v2 prism control "42 43" $p
       sub slurm/rp_check.sbatch w2v2 speaker_split generic "$S5" $p
     done ;;
  5) . tools/prism_env.sh
     TOP=$(python slurm/top_layers.py w2v2 3)
     echo "W2V2 top layers by validation km: $TOP"
     for p in attn gru chunkshuffle; do MEM=${MEM_WEIGHTED:-100G} sub slurm/rp.sbatch w2v2 $p speaker_split "$TOP" "$S5"; done ;;
  6) for m in zipa whisper; do for p in attn gru; do for P in speaker_split prism; do
       sub slurm/rp.sbatch $m $p $P "-1" "$S5"
     done; done; done ;;
  7) sub slurm/rp_official.sbatch w2v2 released
     sub slurm/rp_official.sbatch w2v2 "clean_prism clean_speaker" ;;
  *) echo "tiers are 1-7"; exit 1 ;;
esac
