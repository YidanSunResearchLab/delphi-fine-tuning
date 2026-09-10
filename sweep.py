"""
sweep.py -- the capacity / regularisation sweep, scored by k-fold cross-validation.

    python sweep.py --device cuda                       # the full grid
    python sweep.py --device cuda --folds 5 --out sweep-radc
    python sweep.py --device cuda --dry-run             # print the grid and exit

WHY A SWEEP AND NOT A CHOSEN NUMBER. One pass over the training split scores ~44,800 positions.
Delphi-2M's published shape (12L/12H/120d, 2.2M parameters) was fitted to a corpus roughly 90x
larger; ported here it would sit about 50x into the over-parameterized regime. But this
vocabulary is 50 tokens against Delphi-2M's ~1,300, which makes the per-step problem far
easier and buys back some slack. Those two facts point in opposite directions and neither
settles the architecture, so it is measured.

WHY K-FOLD AND NOT THE VALIDATION SPLIT. The delivered validation split is ~400 subjects.
At this scale its sampling noise is comparable to the differences between adjacent cells of
the ladder, so a sweep scored on it would be selecting largely on noise. Folds are cut from
the `stratum` column so each keeps the study x follow-up x outcome balance, and the pooled
train+val subjects are the only ones touched -- THE TEST SPLIT IS NEVER OPENED HERE.

WHAT IS REPORTED. Mean and spread of the best validation loss across folds, plus its two
components. The winner is the lowest mean, but read the spread before believing a gap: cells
whose intervals overlap are not distinguishable at this sample size and the smaller one should
win on principle.
"""
import os
import sys
import json
import time
import argparse
import subprocess
import itertools

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))

# (n_layer, n_embd, n_head). n_head is chosen so head_dim stays in the 8-24 band the previous
# arm's geometry sweep found mattered; the model asserts n_embd % n_head == 0.
SHAPES = [
    (2, 32, 4),      # ~29k
    (2, 64, 4),      # ~108k
    (3, 48, 4),      # ~90k   <- the config default
    (4, 64, 4),      # ~207k
    (6, 96, 4),      # ~682k
]
DROPOUTS = [0.1, 0.2, 0.3]


def run_one(cfg, shape, dropout, fold, folds, device, out_root, max_iters, extra):
    n_layer, n_embd, n_head = shape
    tag = f"L{n_layer}E{n_embd}H{n_head}_d{dropout}_f{fold}"
    out_dir = os.path.join(out_root, tag)
    cmd = [sys.executable, os.path.join(_ROOT, "train.py"), cfg,
           "--device", device,
           "--n_layer", str(n_layer), "--n_embd", str(n_embd), "--n_head", str(n_head),
           "--dropout", str(dropout),
           "--cv_folds", str(folds), "--cv_fold", str(fold),
           "--max_iters", str(max_iters), "--lr_decay_iters", str(max_iters),
           "--out_dir", out_dir,
           "--log_interval", "1000",
           "--wandb", "disabled"] + extra
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    hist = os.path.join(out_dir, "history.json")
    if proc.returncode != 0 or not os.path.exists(hist):
        tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-1500:]
        print(f"  {tag:<28s} FAILED rc={proc.returncode} ({dt:.0f}s)\n{tail}", flush=True)
        return None
    h = json.load(open(hist))
    best = min(h["history"], key=lambda r: r["selected"]) if h["history"] else None
    if best is None:
        print(f"  {tag:<28s} no evaluations recorded", flush=True)
        return None
    rec = dict(n_layer=n_layer, n_embd=n_embd, n_head=n_head, dropout=dropout, fold=fold,
               best_val=h["best_val_loss"], best_iter=h["best_iter"],
               val_ce=best["val_ce"], val_dt=best["val_dt"], val_total=best["val"],
               seconds=round(dt, 1))
    print(f"  {tag:<28s} val {rec['best_val']:.4f} "
          f"(ce {rec['val_ce']:.4f} dt {rec['val_dt']:.4f}) @ iter {rec['best_iter']:<5d} "
          f"{dt:.0f}s", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/radc_base.py")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=6000)
    ap.add_argument("--out", default="out-sweep-radc")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shapes-only", action="store_true",
                    help="stage 1 only: sweep shapes at dropout 0.2")
    ap.add_argument("--only-dropout", type=float, default=None,
                    help="run just this dropout column of the grid, so the three columns can "
                         "be submitted as three concurrent jobs writing into one --out")
    a, extra = ap.parse_known_args()

    out_root = os.path.join(_ROOT, a.out)
    os.makedirs(out_root, exist_ok=True)

    grid = ([(s, 0.2) for s in SHAPES] if a.shapes_only
            else list(itertools.product(SHAPES, DROPOUTS)))
    if a.only_dropout is not None:
        grid = [(s, d) for s, d in grid if abs(d - a.only_dropout) < 1e-9]
        if not grid:
            sys.exit(f"[sweep] --only-dropout {a.only_dropout} matches no cell of {DROPOUTS}")
    print(f"[sweep] {len(grid)} cells x {a.folds} folds = {len(grid) * a.folds} runs, "
          f"{a.max_iters} iters each, device={a.device}")
    for s, d in grid:
        print(f"    L{s[0]} E{s[1]} H{s[2]}  dropout {d}")
    if a.dry_run:
        return

    rows = []
    t_start = time.time()
    for shape, dropout in grid:
        for fold in range(a.folds):
            r = run_one(a.config, shape, dropout, fold, a.folds, a.device, out_root,
                        a.max_iters, extra)
            if r:
                rows.append(r)
                tag_f = ("_d%g" % a.only_dropout) if a.only_dropout is not None else ""
                with open(os.path.join(out_root, f"sweep_results{tag_f}.json"), "w") as fh:
                    json.dump(rows, fh, indent=1)

    if not rows:
        sys.exit("[sweep] every run failed")

    # Pick up the other columns if they have finished, so the last job out collates the whole
    # grid rather than its own third of it.
    import glob as _glob
    seen = {(r["n_layer"], r["n_embd"], r["n_head"], r["dropout"], r["fold"]) for r in rows}
    for f in _glob.glob(os.path.join(out_root, "sweep_results*.json")):
        try:
            for r in json.load(open(f)):
                k = (r["n_layer"], r["n_embd"], r["n_head"], r["dropout"], r["fold"])
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
        except Exception:
            pass

    # ---- collate
    print(f"\n[sweep] {len(rows)} runs in {(time.time() - t_start) / 60:.1f} min\n")
    key = lambda r: (r["n_layer"], r["n_embd"], r["n_head"], r["dropout"])
    cells = {}
    for r in rows:
        cells.setdefault(key(r), []).append(r)

    print(f"  {'shape':<16s} {'drop':>5s} {'n':>3s} {'mean val':>9s} {'sd':>7s} "
          f"{'mean ce':>8s} {'mean dt':>8s} {'med iter':>9s}")
    summary = []
    for k, rs in cells.items():
        v = np.array([r["best_val"] for r in rs])
        summary.append(dict(
            n_layer=k[0], n_embd=k[1], n_head=k[2], dropout=k[3], n_folds=len(rs),
            mean_val=float(v.mean()), sd_val=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            mean_ce=float(np.mean([r["val_ce"] for r in rs])),
            mean_dt=float(np.mean([r["val_dt"] for r in rs])),
            median_best_iter=float(np.median([r["best_iter"] for r in rs])),
        ))
    summary.sort(key=lambda d: d["mean_val"])
    for d in summary:
        print(f"  L{d['n_layer']} E{d['n_embd']:<3d} H{d['n_head']:<9d} {d['dropout']:>5.1f} "
              f"{d['n_folds']:>3d} {d['mean_val']:>9.4f} {d['sd_val']:>7.4f} "
              f"{d['mean_ce']:>8.4f} {d['mean_dt']:>8.4f} {d['median_best_iter']:>9.0f}")

    best = summary[0]
    # Anything whose mean sits inside the winner's +/- 1 sd is not distinguishable at this
    # sample size. Among those, prefer the smallest model -- at ~45k scored positions the
    # burden of proof is on the larger one.
    thresh = best["mean_val"] + best["sd_val"]
    tied = [d for d in summary if d["mean_val"] <= thresh]
    params = lambda d: 12 * d["n_layer"] * d["n_embd"] ** 2
    pick = min(tied, key=params)
    print(f"\n  lowest mean : L{best['n_layer']}/E{best['n_embd']}/H{best['n_head']} "
          f"dropout {best['dropout']}  ({best['mean_val']:.4f} +/- {best['sd_val']:.4f})")
    if pick is not best:
        print(f"  {len(tied)} cells sit within 1 sd of it; taking the SMALLEST of those:")
    print(f"  SELECTED    : L{pick['n_layer']}/E{pick['n_embd']}/H{pick['n_head']} "
          f"dropout {pick['dropout']}  ({pick['mean_val']:.4f} +/- {pick['sd_val']:.4f}, "
          f"~{params(pick) / 1000:.0f}k block params, median best iter "
          f"{pick['median_best_iter']:.0f})")

    with open(os.path.join(out_root, "sweep_summary.json"), "w") as fh:
        json.dump(dict(summary=summary, selected=pick, lowest_mean=best, runs=rows), fh, indent=2)
    print(f"\n[sweep] -> {out_root}/sweep_summary.json")


if __name__ == "__main__":
    main()
