#!/usr/bin/env bash
# run_figure2.sh -- ONE command for the whole of Figure 2: Monte-Carlo build -> four panels
# -> the assembled combined figure, in both cohort versions.
#
#   ./run_figure2.sh                  # full run  (~5 min MC on 8 cores, then ~30 s of plotting)
#   ./run_figure2.sh --limit 40       # smoke run (~30 s end to end; numbers are NOT publishable)
#   DEVICE=mps ./run_figure2.sh       # MC pass on the GPU instead of CPU multiprocess
#   WORKERS=4 ./run_figure2.sh        # cap the MC worker pool
#   SPLIT=train ./run_figure2.sh      # score the training split instead (see the warning below)
#   DATA=data/rosmap_nodedup CKPT=../delphi/Delphi-ROSMAP-nodedup/ckpt.pt FIG2_TAG=nodedup \
#     ./run_figure2.sh                # 另一份分词 + 它自己的 ckpt，输出隔到 results/figure2/nodedup/
#
# Everything lands in results/figure2/ (or results/figure2/$FIG2_TAG/). The headline file is
# figure2_combined_matched.png.
#
# Ported from delphi-fine-tuning's figure2/run_figure2.sh. What changed: our paths, our split
# names, and the two preflight checks -- upstream fingerprints its dataset against a build
# report.json that ../../tokenization does not write, so the check below verifies what we can
# actually verify (the .bin parses, and the label table matches the checkpoint's vocab size).
set -euo pipefail
cd "$(dirname "$0")"

DEVICE="${DEVICE:-cpu}"
CKPT="${CKPT:-../delphi/Delphi-ROSMAP/ckpt.pt}"
DATA="${DATA:-data/rosmap}"
SPLIT="${SPLIT:-val}"
N_MC="${N_MC:-100}"
PY="${PY:-python3}"
if [ -z "${WORKERS:-}" ]; then
  NCPU=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4 )
  WORKERS=$(( NCPU > 3 ? NCPU - 2 : 1 ))
fi
export OMP_NUM_THREADS=1                     # an MC worker that spawns threads oversubscribes
export KMP_DUPLICATE_LIB_OK=TRUE
# Bind the vocab module's LIVE table to the dataset being scored, not to data/rosmap. figure2_core
# always re-resolves against the checkpoint's own labels.csv, so this only matters for the module
# constants that a probe or a future call site might read -- but two tokenizations now coexist and
# defaulting to the wrong one is silent.
export ROSMAP_LABELS="${ROSMAP_LABELS:-$(pwd)/$DATA/labels.csv}"

echo "=== Figure 2 | device=$DEVICE workers=$WORKERS split=$SPLIT ckpt=$CKPT | extra: ${*:-none}"

# ------------------------------------------------------------------ preflight
[ -f "$CKPT" ] || {
  echo "FATAL: checkpoint not found: $CKPT" >&2
  echo "  Train one first:  cd ../delphi && python train.py config/train_delphi_rosmap.py" >&2
  exit 1
}
[ -f "$DATA/$SPLIT.bin" ] || {
  echo "FATAL: $DATA/$SPLIT.bin not found." >&2
  echo "  Rebuild it:  cd ../tokenization && python build.py" >&2
  echo "  then copy data_rosmap/{train,val}.bin + labels.csv + meta.json into ../delphi/data/rosmap/" >&2
  exit 1
}
if [ "$SPLIT" = "train" ]; then
  echo "  WARNING: scoring the TRAINING split. Every number will be optimistic by memorisation," >&2
  echo "           not just by model selection. Useful as an upper bound, nothing else." >&2
fi

# The failure mode this guards against is a QUIETLY WRONG FIGURE, not a crash: a labels.csv that
# no longer matches the checkpoint loads fine and mislabels every clinical row.
"$PY" - "$DATA" "$SPLIT" "$CKPT" <<'PY'
import sys, numpy as np, pandas as pd, torch
data_dir, split, ckpt = sys.argv[1:4]
d = np.fromfile(f"{data_dir}/{split}.bin", dtype=np.uint32).reshape(-1, 3)
lab = pd.read_csv(f"{data_dir}/labels.csv")["event_name"].tolist()
n_ck = int(torch.load(ckpt, map_location="cpu", weights_only=False)["model_args"]["vocab_size"])
print(f"  {split} split: {len(d):,} events / {len(np.unique(d[:, 0])):,} patients")
if n_ck != len(lab):
    sys.exit(f"FATAL: checkpoint vocab_size {n_ck} != {len(lab)} rows in labels.csv. The "
             f"checkpoint and the data are different tokenizations.")
hi = int(d[:, 2].max()) + 1            # disk -> model
if hi >= len(lab):
    sys.exit(f"FATAL: {split}.bin contains model id {hi} but labels.csv has {len(lab)} rows.")
print(f"  vocabulary ok ({len(lab)} tokens, checkpoint agrees)")
PY
"$PY" -c "import importlib.util as u, sys; sys.exit(0) if u.find_spec('umap') else \
  print('  NOTE: umap-learn missing -> panel d falls back to a deterministic 2-component PCA\n\
        and says so in its title, axis labels, CSV columns and metrics.json. The headline\n\
        number there is 10-NN purity in the original 96-d space, which does not use the\n\
        projection at all. Enable with: pip install umap-learn')"

# ------------------------------------------------------------------ run
# 1) the expensive part: MC trajectories + embeddings, cached under results/figure2/_cache/
#    keyed by (checkpoint md5, dataset, split, n_mc). Re-runs are near-instant.
echo "--- [1/3] Monte-Carlo + embedding build"
"$PY" -u figure2/figure2_core.py --build --ckpt "$CKPT" --dataset "$(basename "$DATA")" \
  --split "$SPLIT" --device "$DEVICE" --workers "$WORKERS" --n-mc "$N_MC" "$@"

# BUILD-ONLY FLAGS MUST NOT REACH THE PLOT STEP: argparse in figure2_panels.py exits 2 on an
# unknown argument, so forwarding --force rebuilt the cache and then died before drawing.
PLOT_ARGS=()
for a in "$@"; do case "$a" in --force) ;; *) PLOT_ARGS+=("$a") ;; esac; done
# Expanded as ${PLOT_ARGS[@]+"${PLOT_ARGS[@]}"} below, NOT as "${PLOT_ARGS[@]}". macOS ships
# bash 3.2, where `set -u` treats an EMPTY array expansion as an unbound variable -- so the
# plain form works for `./run_figure2.sh --limit 40` and dies on the bare no-argument run,
# which is the one anybody reproducing the figure actually types.

# 2+3) two cohorts out of the SAME cache. On this project they contain the same people -- our
#      train.py has no cohort filter -- and metrics_matched.json records `n_dropped: 0` to prove
#      it. Both are produced anyway so the figure never stops saying which population it means.
echo "--- [2/3] panels a-d + combined figure   (cohort: matched)"
"$PY" -u figure2/figure2_panels.py --cohort matched --ckpt "$CKPT" \
  --dataset "$(basename "$DATA")" --split "$SPLIT" --n-mc "$N_MC" ${PLOT_ARGS[@]+"${PLOT_ARGS[@]}"}
echo "--- [3/3] panels a-d + combined figure   (cohort: all)"
"$PY" -u figure2/figure2_panels.py --cohort all --ckpt "$CKPT" \
  --dataset "$(basename "$DATA")" --split "$SPLIT" --n-mc "$N_MC" ${PLOT_ARGS[@]+"${PLOT_ARGS[@]}"}

echo
echo "=== DONE -> $(pwd)/results/figure2/${FIG2_TAG:-}"
echo "    combined figure : figure2_combined_matched.png / .pdf"
echo "                      figure2_combined_allcohort.png / .pdf"
echo "    panels          : fig2a_matched..fig2d_matched.{png,pdf}  (+ *_data.csv per panel)"
echo "    metrics         : metrics_matched.json / metrics_allcohort.json + SUMMARY_*.md"
echo
echo "    READ ./README.md BEFORE QUOTING ANY OF IT -- the calibration rows of panel a are"
echo "    dominated by a rollout artefact that is measured there, and the split is not held out."
