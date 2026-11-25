#!/usr/bin/env bash
# PC (GPU), final evaluation: every transcript-probe and prosody GPU run that is still missing, in priority order.
# The representation probes run on Ada (docs/ada.md); the CPU benchmark probes run on the Mac (tools/bench_mac.sh).
#   1  benchmark BiGRUs, seeds 42-43: {W2V2, ZIPA} x {char GRU, phone GRU, shuffled phone GRU} x {all, shared} x 5 folds
#   2  PRiSM's release, the three BiGRUs up to 5 seeds: ZIPA (never run) and W2V2 (missing seeds)
#   3  rebuild BiGRUs, seeds 44-46, both splits
#   4  the same BiGRUs on shared pictures only (topic check), seeds 42-44, both splits
#   5  prosody BiGRU, seeds 43-46, both splits
#   6  description length (MDL) of the ZIPA phone BiGRU and shuffled BiGRU, seeds 0-2, both splits
#   7  benchmark BiGRUs seed 44; topic check seeds 45-46
# Runs WORKERS jobs at once on the GPU (default 2); a lock directory per run stops two workers taking the same run,
# and a run with results is skipped, so the script can be restarted at any time.
# Needs outputs/transcripts/{w2v2,zipa}_{rebuild,official}.parquet, outputs/features/prosody_rebuild, and
# results/{benchmark,official} plus the existing results/runs (tools/push_pc_inputs.sh copies them).
# Start it under the GPU supervisor (pc-delegate skill), which preheats the GPU, restarts the queue after a driver
# fault, and restarts it when the queue exits with 75:
#   pc supervise nov 'cd ~/developer/sal-project && bash tools/pc_november.sh'
# It appends its own output to outputs/logs/pc_november.log (no pipe on the command line: a pipe would hide the exit
# code that tells the supervisor to restart).
# The queue itself watches for two kinds of GPU trouble and then stops taking runs, lets the running ones finish and
# exits with 75 (the supervisor preheats and reruns it; finished runs are skipped, so it resumes):
#   - the power clamp: before each run, a limit under 35 W (runs would go at a fraction of the speed);
#   - a stalled run: no log output for STALL_MIN minutes (default 20), the sign of a dead CUDA context; it is killed.
# Exit 0: every run done. Exit 1: some runs failed for other reasons (their logs are listed). DRY=1 prints the queue.
# The GPU's draw, limit, clock and temperature are logged once a minute to outputs/logs/gpu_november.csv.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p outputs/logs
exec > >(tee -a outputs/logs/pc_november.log) 2>&1
. tools/prism_env.sh
export PYTHONUNBUFFERED=1 PATH=/usr/lib/wsl/lib:$PATH
STALL_MIN=${STALL_MIN:-20}
STALL_SEC=${STALL_SEC:-$(( STALL_MIN * 60 ))}   # STALL_SEC and POLL_SEC exist for testing
POLL_SEC=${POLL_SEC:-30}
B=results/benchmark
R=results/official
LOCKS=/tmp/sal_locks_november
# Only one copy runs at a time (pc supervise refuses a second session), so locks left by a stopped run are stale.
rm -rf $LOCKS
mkdir -p $LOCKS outputs/logs
FAULT=$LOCKS/.gpu_trouble
FAILED=$LOCKS/.failed
queue=$(mktemp)

# job <done-file> <log name> <command...>
job() { local f=$1 n=$2; shift 2; [ -f "$f" ] || printf '%s\t%s\t%s\n' "$f" "$n" "$*" >> "$queue"; }
tp() {  # tp <name> <args...>: a transcript-probe run whose results go to results/runs/<name>
  local n=$1; shift
  job "results/runs/$n/metrics.json" "$n" python scripts/run_tp.py "$@" --device cuda
}
bench() {
  local seeds=$1 track m spec f
  for s in $seeds; do for track in all shared; do for f in 0 1 2 3 4; do for m in w2v2 zipa; do
    for spec in "char gru" "phone gru" "phone shuffle"; do
      set -- $spec
      tp "tp_${m}_rebuild_${1}_${2}_${track}_f${f}_bench_s$s" --transcripts outputs/transcripts/${m}_rebuild.parquet \
        --dataset $B/meta.parquet --splits $B/splits.parquet --tokens $1 --probe $2 --protocol ${track}_f$f --tag _bench --seeds $s
    done
  done; done; done; done
}
grus() {  # grus <all|generic> "<seeds>"
  local s m spec P gen
  gen=$([ "$1" = generic ] && echo _generic || true)
  for s in $2; do for P in speaker_split prism; do for m in w2v2 zipa; do for spec in "char gru" "phone gru" "phone shuffle"; do
    set -- "$1" "$2" $spec
    tp "tp_${m}_rebuild_${3}_${4}_${P}${gen}_s$s" --transcripts outputs/transcripts/${m}_rebuild.parquet \
      --tokens $3 --probe $4 --protocol $P --prompts $1 --seeds $s
  done; done; done; done
}

bench "42 43"
for s in 42 43 44 45 46; do for m in zipa w2v2; do for spec in "char gru" "phone gru" "phone shuffle"; do
  set -- $spec
  tp "tp_${m}_official_${1}_${2}_prism_s$s" --transcripts outputs/transcripts/${m}_official.parquet \
    --dataset $R/meta_released.parquet --splits $R/splits_released.parquet --tokens $1 --probe $2 --protocol prism --seeds $s
  tp "tp_${m}_official_${1}_${2}_speaker_split_clean_s$s" --transcripts outputs/transcripts/${m}_official.parquet \
    --dataset $R/meta_clean.parquet --splits $R/splits_clean.parquet --tokens $1 --probe $2 --protocol speaker_split --tag _clean --seeds $s
  tp "tp_${m}_official_${1}_${2}_prism_clean_s$s" --transcripts outputs/transcripts/${m}_official.parquet \
    --dataset $R/meta_clean.parquet --splits $R/splits_clean.parquet --tokens $1 --probe $2 --protocol prism --tag _clean --seeds $s
done; done; done
grus all "44 45 46"
grus generic "42 43 44"
for s in 43 44 45 46; do for P in speaker_split prism; do
  job "results/runs/prosody_gru_${P}_s$s/metrics.json" "prosody_gru_${P}_s$s" \
    python scripts/run_prosody.py --protocol $P --seeds $s --device cuda
done; done
for s in 0 1 2; do for P in speaker_split prism; do for probe in gru shuffle; do
  job "results/mdl/mdl_zipa_rebuild_phone_${probe}_${P}_s$s.json" "mdl_zipa_rebuild_phone_${probe}_${P}_s$s" \
    python scripts/mdl.py --transcripts outputs/transcripts/zipa_rebuild.parquet --tokens phone --probe $probe \
    --protocol $P --seeds $s --device cuda
done; done; done
bench 44
grus generic "45 46"

# TEST_QUEUE=<file of done-file TAB name TAB command lines> replaces the job list (for testing the queue itself)
[ -n "${TEST_QUEUE:-}" ] && cp "$TEST_QUEUE" "$queue"
echo "$(date '+%d %b %H:%M') $(wc -l < "$queue") runs queued, ${WORKERS:-2} at a time"
if [ "${DRY:-0}" = 1 ]; then cut -f2 "$queue" | awk '{n++; if (n <= 3 || n % 50 == 0) print "  " n, $0}'; rm -f "$queue"; exit 0; fi
( while true; do
    nvidia-smi --query-gpu=timestamp,power.draw,enforced.power.limit,clocks.sm,temperature.gpu,utilization.gpu \
      --format=csv,noheader >> outputs/logs/gpu_november.csv 2>/dev/null
    sleep 60
  done ) &
sampler=$!
limit() { nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader,nounits 2>/dev/null | cut -d. -f1; }
stop_run() {   # kill one run: its bash -c and everything below it
  local pid=$1 kids
  kids=$(pgrep -P "$pid" || true)
  kill -TERM $kids "$pid" 2>/dev/null; sleep 10; kill -KILL $kids "$pid" 2>/dev/null
}
worker() {
  local f n cmd pid lim stalled rc
  while IFS=$'\t' read -r f n cmd; do
    [ -f "$FAULT" ] && break
    [ -f "$f" ] && continue
    mkdir "$LOCKS/$n" 2>/dev/null || continue
    lim=$(limit)
    if [ -n "$lim" ] && [ "$lim" -lt 35 ]; then
      echo "$(date '+%d %b %H:%M') GPU clamped at $lim W: stopping for a preheat"; touch "$FAULT"; rmdir "$LOCKS/$n"; break
    fi
    echo "$(date '+%d %b %H:%M') start $n"
    bash -c "$cmd" > "outputs/logs/$n.log" 2>&1 &
    pid=$!
    stalled=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$POLL_SEC"
      if [ $(( $(date +%s) - $(stat -c %Y "outputs/logs/$n.log") )) -gt "$STALL_SEC" ]; then
        echo "$(date '+%d %b %H:%M') STALLED $n (no output for $STALL_MIN min): killing it, stopping for a preheat"
        stalled=1; touch "$FAULT"; stop_run "$pid"
      fi
    done
    wait "$pid"; rc=$?
    if [ $stalled = 1 ]; then
      rmdir "$LOCKS/$n"
    elif [ $rc = 0 ] && [ -f "$f" ]; then
      echo "$(date '+%d %b %H:%M') done  $n"
    else
      echo "$(date '+%d %b %H:%M') FAILED $n (exit $rc, outputs/logs/$n.log)"; echo "$n" >> "$FAILED"; rmdir "$LOCKS/$n"
    fi
  done < "$queue"
}
workers=()
for _ in $(seq "${WORKERS:-2}"); do worker & workers+=($!); sleep 5; done
wait "${workers[@]}"
kill $sampler 2>/dev/null
rm -f "$queue"
if [ -f "$FAULT" ]; then
  echo "$(date '+%d %b %H:%M') GPU trouble: exiting with 75 so the supervisor preheats and restarts the queue"; exit 75
fi
if [ -s "$FAILED" ]; then
  echo "$(date '+%d %b %H:%M') $(wc -l < "$FAILED") runs failed:"; sed 's/^/  /' "$FAILED"; exit 1
fi
echo "$(date '+%d %b %H:%M') PC_NOVEMBER_DONE"
