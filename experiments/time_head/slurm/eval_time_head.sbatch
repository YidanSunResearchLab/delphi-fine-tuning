#!/bin/bash
# Time-head experiment, phase 2 -- score each trained arm on the Figure-2 metrics (test split).
#
#   sbatch --array=1-6%3 experiments/time_head/slurm/eval_time_head.sbatch
#
# Array index indexes manifest.tsv rows, i.e. ALL SIX (arm, seed) pairs -- not one seed per
# arm as the capacity sweep's eval did. Two reasons: the capacity sweep already evaluated
# every control at all three seeds, so pairing needs all three here too; and the headline
# metric for this experiment is TIMING (a/transition_time/mae, b/timing_error/*), which is a
# Monte-Carlo quantity with real seed spread. 6 x ~40 min of CPU.
#
# WHY EACH TASK GETS ITS OWN COPY OF THE CODE TREE: figure2_core.OUT_DIR is derived from
# __file__ with no CLI override, so concurrent tasks in one tree overwrite each other's
# metrics.json and figures. Each task copies the ~1 MB code tree so HERE -- and therefore
# OUT_DIR -- is unique for free. data/ is symlinked, never copied: no PHI moves.
#SBATCH --job-name=th-eval
#SBATCH --partition=general-cpu
#SBATCH --account=compute2-mdan
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/theval_%A_%a.out
#SBATCH --error=logs/theval_%A_%a.err
#SBATCH --exclude=c2-node-011

set -euo pipefail

BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0820_capacity}"
EXP="$BASE/experiments/time_head"
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ad-projection
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1                       # each MC worker must stay single-threaded
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-32} - 2 ))

ROW=$(awk -v i="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==i {print; exit}' "$EXP/manifest.tsv")
[ -n "$ROW" ] || { echo "FATAL: no manifest row for array id $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
read -r IDX ARM CTL SEED NL NH NE NP <<< "$ROW"

RUN="${ARM}_s${SEED}"
CKPT="$BASE/th_runs/$RUN/ckpt.pt"
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint at $CKPT -- did phase 1 finish for $RUN?" >&2; exit 1; }

# The control's eval must already exist, or there is nothing to pair against. Not fatal --
# the metrics are still worth having -- but say so loudly rather than discovering it in analyze.
CTLEVAL="$BASE/results/eval/${CTL}_s${SEED}/metrics_matched.json"
[ -f "$CTLEVAL" ] || echo "WARNING: control eval missing ($CTLEVAL). Run the capacity sweep's eval_sweep.sbatch for ${CTL}_s${SEED}, or analyze.py will have no pair for this run."

TREE="$BASE/th_evaltrees/$RUN"
echo "=== eval $RUN  (time head)"
echo "    ckpt  $CKPT"
echo "    ctrl  $CTLEVAL"
echo "    tree  $TREE  (private results/ dir -- see header)"
rm -rf "$TREE"; mkdir -p "$TREE"
cp -r "$BASE/Delphi-2M" "$TREE/Delphi-2M"
rm -f "$TREE/Delphi-2M/data"
ln -s "$BASE/Delphi-2M/data" "$TREE/Delphi-2M/data"   # symlink-to-symlink, resolves fine
cp "$CKPT" "$TREE/Delphi-2M/ckpt_under_test.pt"

cd "$TREE/Delphi-2M"
echo "    host  $(hostname) | workers $WORKERS"

# The whole point of this eval is the timing panels, and those come out of generate(). A
# checkpoint that silently trained the CONTROL -- because phase 1 ran against a stale config
# or a tree that predates the key -- would score without erroring, so verify the checkpoint
# actually carries this arm's overrides before spending 40 CPU-minutes.
CFG="$EXP/configs/th_${ARM}.py" python - <<'PY'
import os, re, sys, torch
sys.path.insert(0, ".")
c = torch.load("ckpt_under_test.pt", map_location="cpu", weights_only=False)
args = c["model_args"]
want = dict(re.findall(r"^(time_head|dt_target) *= *(.+?)\s*$",
                       open(os.environ["CFG"]).read(), re.M))
for k, v in want.items():
    got, exp = args.get(k, "<absent>"), eval(v)
    if got != exp:
        sys.exit(f"FATAL: checkpoint has {k}={got!r}, this arm's config says {exp!r}. "
                 f"Phase 1 did not train what this eval thinks it did.")
if want.get("time_head") == "True":
    keys = [k for k in c["model"] if "time_head" in k]
    if len(keys) != 2:
        sys.exit(f"FATAL: expected 2 time_head tensors in the state dict, found {keys}")
print(f"  checkpoint matches the arm: {want or 'delivered keys'}, iter {c['iter_num']}")
PY

START=$(date +%s)
python -u figure2/figure2_core.py --build --ckpt ckpt_under_test.pt \
       --dataset nacc-dedup-s42 --device cpu --workers "$WORKERS" "$@"
python -u figure2/figure2_panels.py --cohort matched --ckpt ckpt_under_test.pt \
       --dataset nacc-dedup-s42 "$@"
END=$(date +%s)

DEST="$BASE/results/th_eval/$RUN"
mkdir -p "$DEST"
cp -f results/figure2/metrics_matched.json "$DEST/" 2>/dev/null || true
cp -f results/figure2/SUMMARY_matched.md   "$DEST/" 2>/dev/null || true
cp -f results/figure2/*_matched*.csv       "$DEST/" 2>/dev/null || true
cp -f results/figure2/figure2_combined_matched.png "$DEST/" 2>/dev/null || true
echo "{\"run\":\"$RUN\",\"arm\":\"$ARM\",\"control\":\"${CTL}_s${SEED}\",\"seed\":$SEED,\"wall_seconds\":$((END-START))}" \
  > "$DEST/eval_meta.json"

echo "DONE  $RUN  in $(( (END-START)/60 )) min  ->  $DEST"
ls -la "$DEST"
