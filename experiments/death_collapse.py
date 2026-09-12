"""
death_collapse.py -- which knob makes a rollout stop dying?

    python experiments/death_collapse.py --device mps                 # the grid
    python experiments/death_collapse.py --device mps --seeds 42,43   # two seeds
    python experiments/death_collapse.py --dry-run                    # print and exit
    python experiments/death_collapse.py --report out-death-collapse  # re-read a finished run

THE FINDING THIS EXISTS TO EXPLAIN. Two checkpoints in this repository, same data and same
vocabulary (identical data_sig and vocab_sig), differing in exactly four config values:

                        n_layer/n_embd   max_iters   dropout   best_val   never-dies
    out-radc-testckpt        3 / 48           800       0.2      9.154         2.8%
    out-radc-final           4 / 64          6000       0.1      8.670        89.0%

out-radc-final is BETTER on validation loss -- on both components, val_ce 2.140 against 2.552
and val_dt 6.536 against 6.602 -- and its rollouts have stopped dying. Its simulated all-cause
death CIF at age 85 is 0.0083 against an observed 0.246 (eval/results_test.json), a factor of
30; at age 90 it is 0.0110 against 0.534, a factor of 49.

Three variables moved at once, so the cause is not identified. This file identifies it.

THE DESIGN. A 2 x 2 factorial over the two structural knobs, at several seeds:

    shape    3L/48d (90k)   vs   4L/64d (207k)
    dropout  0.2            vs   0.1

max_iters is deliberately NOT a cell. Every run goes the full length with the probe firing at
every evaluation, so the iteration axis is recovered from the probe trajectory of each run at
no extra cost -- 60 points per run instead of one run per point. Making it a cell would also
have confounded it with the learning-rate schedule, since lr_decay_iters tracks max_iters and
"iteration 800 of a 6000-iteration cosine" is not the same optimiser state as "iteration 800
of an 800-iteration cosine". That confound is exactly why the original pair cannot be read.

To keep the comparison against the original pair honest, the grid also runs REPLICA cells that
reproduce the two delivered configurations exactly, schedule included.

TWO DELIBERATE DEVIATIONS FROM configs/radc_base.py, both recorded because they change what
the numbers mean:

  patience = 0. Early stopping would cut the trajectories at different iterations per cell and
  the iteration axis would no longer be shared. Cells are compared at matched iteration counts
  instead, and best-val is still tracked for reference.

  rollout_probe = True everywhere. Verified not to perturb training: a probe-on and a probe-off
  run of the same config produce bit-identical validation losses (16.4659 / 16.0628 / 14.9589
  at iters 100/200/300). tests/test_pipeline.py:test_rollout_probe asserts the RNG isolation
  that this rests on.

WHAT TO READ IN THE OUTPUT. `frac_no_death` is the symptom; `p_death_next` is the same thing
without Monte-Carlo noise and is the better trace. The question is not which cell ends lowest
but WHEN each trajectory turns over, and whether the turn is driven by iterations (all cells
collapse, just at different rates), by capacity (only the wide cells collapse) or by
regularisation (only the d0.1 cells collapse).
"""
import os
import sys
import json
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (tag, n_layer, n_embd, dropout, max_iters). The first four are the factorial; the last two
# reproduce the delivered checkpoints exactly, schedule included.
FACTORIAL = [
    ("S_small_d0.2", 3, 48, 0.2, None),
    ("S_small_d0.1", 3, 48, 0.1, None),
    ("S_big_d0.2",   4, 64, 0.2, None),
    ("S_big_d0.1",   4, 64, 0.1, None),
]
REPLICAS = [
    ("R_testckpt", 3, 48, 0.2, 800),      # exactly out-radc-testckpt
    ("R_final",    4, 64, 0.1, 6000),     # exactly out-radc-final
]
PROBE_KEYS = ("probe/p_death_next", "probe/frac_no_death", "probe/median_end_age",
              "probe/death_cif_85", "probe/death_cif_ratio_85",
              "probe/death_cif_90", "probe/death_cif_ratio_90")


def run_one(cell, seed, device, out_root, max_iters, extra):
    tag, n_layer, n_embd, dropout, cell_iters = cell
    iters = cell_iters or max_iters
    name = f"{tag}_s{seed}"
    out_dir = os.path.join(out_root, name)
    cmd = [sys.executable, os.path.join(_ROOT, "train.py"),
           os.path.join(_ROOT, "configs", "radc_base.py"),
           "--device", device,
           "--n_layer", str(n_layer), "--n_embd", str(n_embd), "--dropout", str(dropout),
           "--max_iters", str(iters), "--lr_decay_iters", str(iters),
           "--seed", str(seed),
           "--rollout_probe", "True",
           "--patience", "0",                 # see the deviations note in the module docstring
           "--eval_interval", "100",
           "--log_interval", "100000",
           "--out_dir", out_dir,
           "--wandb", "disabled"] + extra
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    hist = os.path.join(out_dir, "history.json")
    if proc.returncode != 0 or not os.path.exists(hist):
        tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
        print(f"  {name:<22s} FAILED rc={proc.returncode} ({dt:.0f}s)\n{tail}", flush=True)
        return None
    h = json.load(open(hist))
    last = h["history"][-1] if h["history"] else {}
    print(f"  {name:<22s} {dt:5.0f}s  best_val {h['best_val_loss']:.4f} @ {h['best_iter']}"
          f"  final never-dies {100 * last.get('probe/frac_no_death', float('nan')):5.1f}%"
          f"  p_death {last.get('probe/p_death_next', float('nan')):.4f}", flush=True)
    return dict(tag=tag, seed=seed, n_layer=n_layer, n_embd=n_embd, dropout=dropout,
                max_iters=iters, out_dir=out_dir, best_val=h["best_val_loss"],
                best_iter=h["best_iter"], history=h["history"], seconds=round(dt, 1))


def collapse_iter(hist, key="probe/frac_no_death", thresh=0.5):
    """First iteration at which more than `thresh` of draws never die. None if it never does."""
    for r in hist:
        if r.get(key) is not None and r[key] > thresh:
            return r["iter"]
    return None


def at_iter(hist, it, key):
    """The recorded value at the evaluation nearest to (and not past) `it`."""
    seen = [r for r in hist if r["iter"] <= it and r.get(key) is not None]
    return seen[-1][key] if seen else float("nan")


def report(results, out_root):
    by_tag = {}
    for r in results:
        by_tag.setdefault(r["tag"], []).append(r)

    lines = []
    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("\n" + "=" * 100)
    out("NEVER-DIES RATE (fraction of age-80 draws that reach the cap without emitting Death)")
    out("=" * 100)
    marks = [400, 800, 1600, 3000, 6000]
    out(f"{'cell':<16s} {'n':>2s} " + " ".join(f"{('it' + str(m)):>9s}" for m in marks)
        + f" {'collapse@':>10s} {'best_val':>9s}")
    out("-" * 100)
    for tag in [c[0] for c in FACTORIAL + REPLICAS]:
        rs = by_tag.get(tag, [])
        if not rs:
            continue
        cells = []
        for m in marks:
            v = [at_iter(r["history"], m, "probe/frac_no_death") for r in rs]
            v = [x for x in v if np.isfinite(x)]
            cells.append(f"{100 * np.mean(v):8.1f}%" if v else "        -")
        ci = [collapse_iter(r["history"]) for r in rs]
        ci_s = "never" if all(c is None for c in ci) else \
               "/".join("never" if c is None else str(c) for c in ci)
        bv = np.mean([r["best_val"] for r in rs])
        out(f"{tag:<16s} {len(rs):>2d} " + " ".join(cells) + f" {ci_s:>10s} {bv:9.4f}")

    out("\n" + "=" * 100)
    out("P(Death | next token), renormalised -- the same signal without Monte-Carlo noise")
    out("=" * 100)
    out(f"{'cell':<16s} {'n':>2s} " + " ".join(f"{('it' + str(m)):>9s}" for m in marks))
    out("-" * 100)
    for tag in [c[0] for c in FACTORIAL + REPLICAS]:
        rs = by_tag.get(tag, [])
        if not rs:
            continue
        cells = []
        for m in marks:
            v = [at_iter(r["history"], m, "probe/p_death_next") for r in rs]
            v = [x for x in v if np.isfinite(x)]
            cells.append(f"{np.mean(v):9.4f}" if v else "        -")
        out(f"{tag:<16s} {len(rs):>2d} " + " ".join(cells))

    out("\n" + "=" * 100)
    out("MAIN EFFECTS at iteration 6000 (factorial cells only, averaged over seeds)")
    out("=" * 100)
    fac = [r for r in results if r["tag"].startswith("S_")]
    if fac:
        for label, keyf in (("shape 3L/48d", lambda r: r["n_embd"] == 48),
                            ("shape 4L/64d", lambda r: r["n_embd"] == 64),
                            ("dropout 0.2", lambda r: r["dropout"] == 0.2),
                            ("dropout 0.1", lambda r: r["dropout"] == 0.1)):
            sel = [r for r in fac if keyf(r)]
            if not sel:
                continue
            nd = np.mean([at_iter(r["history"], 6000, "probe/frac_no_death") for r in sel])
            pd_ = np.mean([at_iter(r["history"], 6000, "probe/p_death_next") for r in sel])
            bv = np.mean([r["best_val"] for r in sel])
            out(f"  {label:<16s} n={len(sel):<2d} never-dies {100 * nd:6.1f}%   "
                f"p_death {pd_:.4f}   best_val {bv:.4f}")

    out("\nHow to read the collapse@ column: the first iteration at which more than half the")
    out("draws stop dying. 'never' means the cell held for the whole run.")

    path = os.path.join(out_root, "report.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(out_root, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[death_collapse] wrote {path} and {out_root}/results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--max-iters", type=int, default=6000)
    ap.add_argument("--out", default=os.path.join(_ROOT, "out-death-collapse"))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--skip-replicas", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", metavar="DIR",
                    help="skip training, re-read results.json from DIR and re-print the report")
    a, extra = ap.parse_known_args()

    if a.report:
        report(json.load(open(os.path.join(a.report, "results.json"))), a.report)
        return

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    cells = FACTORIAL + ([] if a.skip_replicas else REPLICAS)
    # A replica pins its own schedule, so running it at more than one seed still measures the
    # seed spread of the delivered configuration -- keep them on the first seed only unless
    # the factorial is also single-seed.
    jobs = [(c, s) for c in cells for s in (seeds if c[4] is None else seeds[:1])]

    print(f"[death_collapse] {len(jobs)} runs, {a.jobs} at a time, device {a.device}")
    for c, s in jobs:
        print(f"    {c[0]}_s{s}: {c[1]}L/{c[2]}d dropout {c[3]} "
              f"iters {c[4] or a.max_iters}")
    if a.dry_run:
        return

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, c, s, a.device, a.out, a.max_iters, extra) for c, s in jobs]
        results = [f.result() for f in futs]
    results = [r for r in results if r]
    print(f"\n[death_collapse] {len(results)}/{len(jobs)} runs in {(time.time() - t0) / 60:.1f} min")
    if results:
        report(results, a.out)


if __name__ == "__main__":
    main()
