#!/usr/bin/env bash
# run_figure2.sh -- ONE command for the whole of Figure 2: Monte-Carlo build -> four panels
# -> the assembled combined figure, in both cohort versions.
#
#   ./figure2/run_figure2.sh             # full run   (~25 min on 32 cores, then ~2 min of plotting)
#   ./figure2/run_figure2.sh --limit 300 # smoke run  (~2 min end to end; numbers are NOT publishable)
#   DEVICE=cuda ./figure2/run_figure2.sh # MC pass on GPU instead of CPU multiprocess
#   WORKERS=8 ./figure2/run_figure2.sh   # cap the MC worker pool
#
# Everything lands in results/figure2/ -- the combined figure is figure2_combined_matched.{png,pdf}
# (and figure2_combined_allcohort.{png,pdf} for the second cohort).
#
# This is the off-cluster equivalent of slurm/figure2.sbatch, which stays the entry point for
# batch submission. Both drive the same two Python programs with the same arguments, so they
# produce the same figures; keep them in step if you change either.
#
# Extra arguments are forwarded to ALL THREE steps, so only pass flags that every step accepts:
#   --limit N     cap the number of patients (smoke runs)
#   --split NAME  which split to score (default: test)
#   --n-mc N      Monte-Carlo trajectories per patient (default: 100)
# Use the DEVICE / WORKERS / CKPT env vars for the rest -- those are core-only flags.
set -euo pipefail
cd "$(dirname "$0")/.."          # everything resolves from the package root, Delphi-2M/

DEVICE="${DEVICE:-cpu}"
CKPT="${CKPT:-out-delphi2m-dedup-mask-s42/ckpt.pt}"
DATA="${DATA:-data/nacc-dedup-s42}"          # preflighted below; also passed to both steps as --dataset
PY="${PY:-python}"
# leave 2 cores for the parent process; each MC worker is pinned single-threaded below
if [ -z "${WORKERS:-}" ]; then
  NCPU=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4 )
  WORKERS=$(( NCPU > 3 ? NCPU - 2 : 1 ))
fi
export OMP_NUM_THREADS=1                     # an MC worker that spawns threads oversubscribes the box
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== Figure 2 | device=$DEVICE workers=$WORKERS ckpt=$CKPT | extra args: ${*:-none}"

# ------------------------------------------------------------------ preflight
# Both checks below exist because their failure mode is a *quietly wrong figure*, not a crash:
# a missing link makes the panels score nothing, and the keep-all build makes them score a model
# on data it was never trained on. Fail loudly instead.
[ -f "$CKPT" ] || {
  echo "FATAL: checkpoint not found: $CKPT" >&2
  echo "  This repository ships CODE ONLY -- no checkpoint is committed. Train one first:" >&2
  echo "    python training/train.py config/train_delphi2m_mask_dedup.py --device=cuda" >&2
  echo "  (or point CKPT=... at a checkpoint you already have)" >&2
  exit 1
}
[ -f "$DATA/test.bin" ] || {
  echo "FATAL: $DATA/test.bin not found." >&2
  echo "  Run 'python data_prep/make_dataset.py --csv /path/to/investigator_nacc72.csv --seed 42' first." >&2
  exit 1
}
# Fingerprint the split so a mismatched dataset cannot be scored silently. Only the default
# dedup build has a known event count; a deliberately different dataset (e.g. a cohort subset
# passed via DATA=) is reported but not enforced.
EXPECT_EVENTS="${EXPECT_EVENTS:-}"
# 611,673 since the sum-scores were split into items (CDRSUM -> 6 CDR boxes, FAQ total -> 9
# domains, NPI-Q total -> 12 symptoms, + GDS; vocab 111 -> 228). It was 228,057 before that,
# and 310,241 at the intermediate CDR-only step. Bump this whenever the tokenizer changes --
# the number is the whole point of the guard, a stale one just fails every run.
if [ -z "$EXPECT_EVENTS" ] && [ "$DATA" = "data/nacc-dedup-s42" ]; then EXPECT_EVENTS=611673; fi
"$PY" - "$DATA" "${EXPECT_EVENTS:-0}" <<'PY'
import numpy as np, sys
d = np.fromfile(f"{sys.argv[1]}/test.bin", dtype=np.uint32).reshape(-1, 3)
n_ev, n_pt = len(d), len(np.unique(d[:, 0]))
expect = int(sys.argv[2])
print(f"  test split: {n_ev:,} events / {n_pt:,} patients")
if not expect:
    print("  (non-default dataset -- fingerprint reported, not enforced)")
elif abs(n_ev - expect) > 2_000:
    sys.exit(f"FATAL: expected ~{expect:,} events, got {n_ev:,}/{n_pt:,}. The checkpoint and the "
             "data do not match -- e.g. this is the KEEP-ALL build (~329k), not keep-transitions. "
             "Re-point the dataset link, or set EXPECT_EVENTS= if the mismatch is intentional.")
else:
    print("  dataset fingerprint ok (matches the checkpoint)")
PY
"$PY" -c "import importlib.util as u, sys; sys.exit(0) if u.find_spec('umap') else \
  print('  WARNING: umap-learn missing -> panels a/b/c render, panel d will be reported as failed.\n\
           enable with: pip install umap-learn')"

# ------------------------------------------------------------------ run
# 1) the expensive part: MC trajectories + embeddings, cached under results/figure2/_cache/
#    keyed by (checkpoint, split, n_mc). Re-runs are near-instant; delete the cache to rebuild.
echo "--- [1/3] Monte-Carlo + embedding build"
DATASET="$(basename "$DATA")"                # both programs take the dir NAME under data/, not the path
"$PY" -u figure2/figure2_core.py --build --ckpt "$CKPT" --dataset "$DATASET" --device "$DEVICE" --workers "$WORKERS" "$@"

# 2+3) two cohorts out of the SAME cache. "matched" restricts scoring to the patients train.py
#      actually fitted on (>=4 visits, OR >=2 visits + a NACCUDSD transition) -- that is the paper
#      figure. "all" scores the whole evaluable test split, ~43% of which is outside the training
#      regime; kept as a generalisation check under *_allcohort names.
echo "--- [2/3] panels a-d + combined figure   (cohort: matched -- the paper figure)"
"$PY" -u figure2/figure2_panels.py --cohort matched --ckpt "$CKPT" --dataset "$DATASET" "$@"
echo "--- [3/3] panels a-d + combined figure   (cohort: all -- generalisation check)"
"$PY" -u figure2/figure2_panels.py --cohort all --ckpt "$CKPT" --dataset "$DATASET" "$@"

echo
echo "=== DONE -> $(pwd)/results/figure2/"
echo "    combined figure : figure2_combined_matched.png / .pdf   <- the paper figure"
echo "                      figure2_combined_allcohort.png / .pdf"
echo "    panels          : fig2a_matched..fig2d_matched.{png,pdf}  (+ *_data.csv per panel)"
echo "    metrics         : metrics_matched.json / metrics_allcohort.json + SUMMARY_*.md"
