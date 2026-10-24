#!/bin/bash
# Chain driver for the full VDE run on Perlmutter, run detached on a login node:
#   (setsid nohup bash vde_chain.sh [JOBID] > chain.log 2>&1 < /dev/null &)
#
# Each 4 h interactive GPU allocation runs segments of hybrid on 4 ranks / 4 A100.
# A segment ends at the Parthenon wall limit (-t), which writes a final restart dump;
# the next segment restarts from the newest restart dump. The first segment starts
# from the relaxed equilibrium X_vde.petscbin. Between checks the live plots and
# movies are refreshed (vde_live.py). The chain ends when vde_live.py reports the end
# of the VDE (exit 3), or when RUN/STOP exists.
#
# JOBID: an allocation to use first (else one is requested).
# Environment: RUN_BASE (run directory base), DECK, ID (problem_id), for other runs.
set -u
A=$PSCRATCH/agent
B=${RUN_BASE:-$A/results/vde_full}   # holds inputs -> the owner's inputs (read only)
W=$B/r1/r2                     # run directory
EXE=$A/build-full-gpu/bin/hybrid
TOOLS=$A/vde_full_tools
PY=${PY:-/global/common/software/nersc/pe/conda-envs/26.8.1/python-3.13/nersc-python/bin/python3}
DECK=${DECK:-vde_full.input}
ID=${ID:-vde_full}            # problem_id of the deck; also the allocation name
MARGIN_MIN=30                  # allocation left after the -t limit: the cycle in progress must finish
LIVE_EVERY=${LIVE_EVERY:-900}                 # s between live refreshes
QUOTA_PCT=90                   # pscratch use at which the phdf files are cleared
OVR="Simulation/push_timing=1 ${EXTRA_OVR:-}"   # EXTRA_OVR: deck overrides for every segment, e.g. Hybrid/gamma_min_floor=1.001
MODS="ml use /global/cfs/cdirs/m3016/hymera/lmod; ml load hymera-cuda; ml craype-x86-milan"

log() { echo "[$(date '+%F %T')] $*"; }

live() {
  local part
  part=$(cat $W/live/part 2>/dev/null || echo "")
  HDF5_USE_FILE_LOCKING=FALSE $PY $TOOLS/tests/hybrid/vde_live.py $W \
    --geometry $B/inputs/AxisSymmetricGeometry.dat --check-end --part "$part" > $W/live/live.log 2>&1
  return $?
}

# When pscratch fills up: close the movies made so far as a part, delete the numbered
# phdf files except the newest, and continue the movies as the next part.
space_check() {
  local pct
  pct=$(myquota 2>/dev/null | awk '$1 == "pscratch" {gsub("%", "", $4); print int($4)}')
  [ -n "$pct" ] && [ "$pct" -ge $QUOTA_PCT ] || return 0
  local k; k=$(( $(cat $W/live/nparts 2>/dev/null || echo 0) + 1 ))
  log "pscratch at $pct%: closing movie part $k, deleting phdf files"
  echo "_part$k" > $W/live/part
  live
  mkdir -p $W/live/part$k; mv $W/live/*_part$k.webm $W/live/part$k/ 2>/dev/null
  mv $W/live/frames $W/live/part$k/frames
  ls $W/$ID.out2.[0-9]*.phdf | head -n -1 | xargs -r rm -f
  echo $k > $W/live/nparts
  echo "_part$((k + 1))" > $W/live/part
}

# Newest numbered restart dump that opens: prints "<file> <simulation time>".
# out8.final is skipped: a segment ends by the -t signal, and the signal-time outputs
# do not advance the output schedule, so a restart from it writes its first out2
# output again at once, before the MHD state is loaded into the kinetic field grid.
# That output (fields, currents) is stale; the numbered dumps restart cleanly
# (bit-identical to an uninterrupted run, tests/hybrid/compare_restart.py --exact).
# The cost is the last MHD step of each segment.
latest_restart() {
  HDF5_USE_FILE_LOCKING=FALSE $PY - "$W" "$ID" <<'EOF'
import glob, sys, h5py
best = None
for fn in glob.glob(f"{sys.argv[1]}/{sys.argv[2]}.out8.[0-9]*.rhdf"):
    rd = fn[:-len(".rhdf")] + ".rdat"
    try:
        with h5py.File(fn, "r") as f:
            t = float(f["Info"].attrs["Time"])
        open(rd, "rb").close()
    except Exception:
        continue
    if best is None or t > best[0]:
        best = (t, fn)
print(f"{best[1]} {best[0]!r}" if best else "")
EOF
}

minutes_left() {
  squeue -h -j $1 -o %L | awk -F'[-:]' '{ if (NF==4) m=$1*1440+$2*60+$3; else if (NF==3) m=$1*60+$2; else m=$1; print m }'
}

# Prints only the job ID of a running allocation named $ID, or nothing. Reuses one
# that is already running or pending, so at most one allocation exists at a time.
alloc() {
  local j
  j=$(squeue -h -u $USER -n $ID -o %i | sort -n | head -1)
  if [ -z "$j" ]; then
    log "requesting a 4 h interactive GPU node" >&2
    salloc --no-shell --nodes 1 --qos interactive --time 04:00:00 --constraint gpu \
      --account m3016 -J $ID > $B/salloc.log 2>&1
    j=$(grep -oE 'Granted job allocation [0-9]+|job [0-9]+ queued' $B/salloc.log | grep -oE '[0-9]+' | head -1)
  fi
  [ -n "$j" ] || return 0
  while [ "$(squeue -h -j $j -o %T 2>/dev/null)" = "PENDING" ]; do sleep 60; done
  [ "$(squeue -h -j $j -o %T 2>/dev/null)" = "RUNNING" ] && echo $j
}

mkdir -p $W/live
[ -L $B/inputs ] || ln -s $PSCRATCH/git/hymera/inputs $B/inputs
JOB=${1:-}
SEG=$(ls $W/segments 2>/dev/null | wc -l)
mkdir -p $W/segments
while :; do
  [ -e $B/STOP ] && { log "STOP file found"; break; }
  if [ -z "$JOB" ] || [ -z "$(squeue -h -j $JOB -t R -o %i 2>/dev/null)" ]; then
    JOB=$(alloc)
    [[ "$JOB" =~ ^[0-9]+$ ]] || { log "no allocation ($(tail -1 $B/salloc.log)), retry in 10 min"; JOB=""; sleep 600; continue; }
    log "allocation $JOB"
  fi
  # A driver restarted while a segment runs: let that hybrid step finish first.
  # (squeue -s does not list the steps on this system; sacct does.)
  while [ -n "$(sacct -j $JOB --format=JobID,State -n -P 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\|RUNNING')" ]; do
    log "a step is already running on job $JOB; waiting for it to end"
    sleep 300
  done
  [ -e $B/STOP ] && { log "STOP file found"; break; }
  [ -z "$(squeue -h -j $JOB -t R -o %i 2>/dev/null)" ] && { JOB=""; continue; }
  left=$(minutes_left $JOB)
  lim=$(( left - MARGIN_MIN ))
  if [ $lim -lt 30 ]; then
    log "job $JOB has $left min left, too few for a segment; waiting for it to end"
    while [ -n "$(squeue -h -j $JOB -o %i 2>/dev/null)" ]; do sleep 60; done
    JOB=""; continue
  fi
  SEG=$((SEG + 1))
  read -r R RT <<< "$(latest_restart)"
  cd $W
  if [ -z "$R" ]; then
    START="-i $DECK MHD_Config/ic_binary_path=X_vde.petscbin"
  else
    START="-r $(basename $R)"
  fi
  T=$(printf "%02d:%02d:00" $((lim / 60)) $((lim % 60)))
  log "segment $SEG on job $JOB: $START, wall limit $T"
  echo "job=$JOB start=$START limit=$T t0=$(date +%s)" > $W/segments/$(printf %03d $SEG).txt
  srun --jobid=$JOB --overlap -n 4 -c 32 --cpu-bind=cores --gpus-per-task=1 --gpu-bind=single:1 \
    bash -lc "$MODS; export OMP_NUM_THREADS=1; $EXE $START -t $T $OVR" \
    > $W/log_$(printf %03d $SEG).txt 2>&1 &
  RUNPID=$!
  END=0
  while kill -0 $RUNPID 2>/dev/null; do
    sleep $LIVE_EVERY
    live; rc=$?
    space_check
    if [ $rc -eq 3 ]; then
      log "end of the VDE reached: $(tr '\n' ' ' < $W/live/status.txt)"
      END=1
      # srun forwards SIGTERM; Parthenon writes its final outputs at the end of the cycle
      kill -TERM $RUNPID 2>/dev/null
      break
    fi
    [ -e $B/STOP ] && break
  done
  wait $RUNPID; RC=$?
  echo "rc=$RC t1=$(date +%s)" >> $W/segments/$(printf %03d $SEG).txt
  log "segment $SEG ended rc=$RC"
  live; [ $? -eq 3 ] && END=1
  [ $END -eq 1 ] && break
  # The newest dump is usually out8.final in every segment: compare times, not names.
  read -r R2 RT2 <<< "$(latest_restart)"
  if [ -z "$R2" ] || { [ -n "$R" ] && [ "$RT2" = "$RT" ]; }; then
    log "segment $SEG made no new restart dump; stopping the chain (see log_$(printf %03d $SEG).txt)"
    break
  fi
done
log "chain finished"
