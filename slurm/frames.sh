# Sourced by representation-probe jobs after slurm/common.sh.
#   use_frames <w2v2|whisper|zipa> <rebuild|official> "<frame layers>"
# sets FEAT to a features directory that holds frames_L<l>.npy for every requested layer.
# /home2 has a 50 GB quota and /share1 is not mounted on compute nodes, so the frames (~15 GB per layer for W2V2
# and ZIPA) are not kept between jobs: if outputs/features/<model>_<data> lacks a layer, this job extracts the
# frames into node-local scratch (/ssd_scratch, else /scratch; ~2 TB per node, wiped after ~7 days, not shared
# between nodes) and deletes them when the job ends. Only the small run results are written to /home2.
# The directory keeps the name <model>_<data>, so run names are the same as with cached features.
use_frames() {
  local model=$1 data=$2 layers=$3 home="outputs/features/${1}_${2}" missing="" l n_layers
  case $model in w2v2) n_layers=25 ;; whisper) n_layers=13 ;; zipa) n_layers=8 ;; esac
  for l in $layers; do
    [ -f "$home/frames_L$(( (l % n_layers + n_layers) % n_layers )).npy" ] || missing=1
  done
  if [ -z "$missing" ]; then FEAT=$home; echo "frames: cached in $FEAT"; return; fi
  local root="" r
  for r in ${SCRATCH_ROOTS:-/ssd_scratch /scratch}; do
    if [ -d "$r" ] && [ -w "$r" ]; then root=$r; break; fi
  done
  [ -n "$root" ] || { echo "no writable scratch (tried ${SCRATCH_ROOTS:-/ssd_scratch /scratch})"; exit 1; }
  df -h /ssd_scratch /scratch 2>/dev/null || true
  local need=$(( 20 * $(echo $layers | wc -w) ))
  local free=$(df -BG --output=avail "$root" | tail -1 | tr -dc 0-9)
  [ "$free" -ge "$need" ] || { echo "only ${free} GB free in $root, need ~${need} GB"; exit 1; }
  local dir="$root/$USER/sal_${SLURM_JOB_ID:-$$}"
  mkdir -p "$dir"
  trap 'rm -rf "'"$dir"'"' EXIT
  FEAT="$dir/${model}_${data}"
  local src=data/vaani_hi_${data}/vaani_hi_${data}.parquet
  echo "frames: extracting layers $layers of $model into $FEAT ($(date +%H:%M))"
  python scripts/extract_features.py --model "$model" --data "$src" --out-dir "$FEAT" --frame-layers $layers
  echo "frames: done ($(date +%H:%M)); $(du -sh "$FEAT" | cut -f1)"
}

# rp_todo <model> <rebuild|official> "<layers>" <probe> <protocol+tag+suffix> "<seeds>"
# prints the seeds whose run rp_<model>_<data>_L<layers>_<probe>_<protocol...>_s<seed> has no results yet, naming
# layers as scripts/run_rp.py does (sorted, resolved, joined by "-"). A job calls it before use_frames, so a
# resubmitted job whose seeds are all done exits without spending 1-2 h on extraction.
rp_todo() {
  local model=$1 data=$2 layers=$3 probe=$4 suffix=$5 seeds=$6 n_layers tag
  case $model in w2v2) n_layers=25 ;; whisper) n_layers=13 ;; zipa) n_layers=8 ;; esac
  tag=$(for l in $layers; do echo $(( (l % n_layers + n_layers) % n_layers )); done | sort -n | uniq | paste -sd- -)
  todo "rp_${model}_${data}_L${tag}_${probe}_${suffix}" $seeds
}
