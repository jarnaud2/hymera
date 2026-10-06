#!/usr/bin/env bash
# Kinetic regression check: run one kinetic executable on a reduced deck and
# compare a fingerprint of everything it wrote with a recorded hash.
#
#   run_kinetic.sh <exe> <deck> --baseline NAME [options] [Block/param=value ...]
#
#   --baseline NAME   compare with tests/regression/baselines/NAME.sha256
#   --update          record a new baseline instead of comparing
#   --keep DIR        run in DIR and keep it (default: a temporary directory)
#   --threads N       OMP_NUM_THREADS for the run (default 1)
#
# Any remaining arguments are passed to the executable as Parthenon input
# overrides, e.g. parthenon/time/nlim=4.
#
# The executables read fields.h5, X_4.petscbin and ../../inputs/... relative to
# the working directory, so the run happens in DIR/r1/r2 with those staged and
# DIR/inputs linked to the repository's inputs/.
#
# The fingerprint (fingerprint_kinetic.py) hashes every number in every output
# file, not the raw bytes, so wall-clock attributes and output layout do not
# affect it. A hash mismatch says only that some number changed; run --keep on
# both revisions and diff the two fingerprint.txt files to see which.
set -euo pipefail

usage="usage: run_kinetic.sh <exe> <deck> --baseline NAME [--update] [--keep DIR] [--threads N] [overrides]"
EXE="${1:?$usage}"
DECK="${2:?$usage}"
shift 2
MODE=check
KEEP=""
NAME=""
THREADS=1
OVERRIDES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --update) MODE=update; shift ;;
    --keep) KEEP="$2"; shift 2 ;;
    --baseline) NAME="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --*) echo "unknown option: $1" >&2; exit 2 ;;
    *) OVERRIDES+=("$1"); shift ;;
  esac
done
[ -n "$NAME" ] || { echo "$usage" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
BASELINE="$HERE/../baselines/${NAME}.sha256"
PYTHON="${HARNESS_PYTHON:-$HOME/.venv-hymera/bin/python}"
"$PYTHON" -c 'import h5py' 2>/dev/null ||
  { echo "need h5py in $PYTHON; run tests/regression/setup_env.sh" >&2; exit 1; }

EXE="$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")"
DECK="$(cd "$(dirname "$DECK")" && pwd)/$(basename "$DECK")"

if [ -n "$KEEP" ]; then
  rm -rf "$KEEP"
  mkdir -p "$KEEP"
  WORK="$(cd "$KEEP" && pwd)"
else
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT
fi
RUN="$WORK/r1/r2"
mkdir -p "$RUN"
ln -s "$REPO/inputs" "$WORK/inputs"
cp "$HERE/fixtures/fields.h5" "$HERE/fixtures/X_4.petscbin" "$HERE/fixtures/X_4.petscbin.info" "$RUN/"

LAUNCH=()
if command -v mpirun >/dev/null; then LAUNCH=(mpirun -n 1); fi
if ! (cd "$RUN" && OMP_NUM_THREADS="$THREADS" "${LAUNCH[@]}" "$EXE" -i "$DECK" "${OVERRIDES[@]}" > log.txt 2>&1); then
  echo "FAIL: $NAME run exited non-zero; last lines of $RUN/log.txt:" >&2
  grep -v '^\[' "$RUN/log.txt" | tail -20 >&2
  exit 1
fi

"$PYTHON" "$HERE/fingerprint_kinetic.py" "$RUN" > "$WORK/fingerprint.txt"
HASH="$(sha256sum < "$WORK/fingerprint.txt" | awk '{print $1}')"
LINES="$(wc -l < "$WORK/fingerprint.txt")"

if [ "$MODE" = update ]; then
  mkdir -p "$(dirname "$BASELINE")"
  printf '%s  %s lines\n' "$HASH" "$LINES" > "$BASELINE"
  echo "recorded baseline $NAME: $HASH ($LINES lines)"
  exit 0
fi

[ -f "$BASELINE" ] || { echo "no baseline at $BASELINE; run with --update" >&2; exit 1; }
WANT="$(awk '{print $1}' "$BASELINE")"
if [ "$HASH" = "$WANT" ]; then
  echo "PASS: $NAME bit-identical to baseline ($LINES lines)"
  exit 0
fi
echo "FAIL: $NAME output changed"
echo "  baseline: $WANT"
echo "  current:  $HASH ($LINES lines)"
echo "  rerun with --keep DIR on both revisions and diff DIR/fingerprint.txt"
exit 1
