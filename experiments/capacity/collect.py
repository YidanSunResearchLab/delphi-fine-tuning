"""
collect.py -- parse every runs/<run>/train.log into two tidy CSVs. No cluster needed.

    python experiments/capacity/collect.py --runs <dir> --out <dir>

Writes to results/:
  curves.csv    one row per (run, eval point): iter, train_loss, val_loss
  summary.csv   one row per run: best_iter, best_val, train-val gap at the best point,
                final_val, throughput, wall time

train.py does not write a metrics file -- it prints. Rather than patch the delivered
training script (which would invalidate tests/golden.py's premise that the framework is
untouched), this parses stdout. The two formats it depends on are:

    step {iter}: train loss {float}, val loss {float}
    iter {n}: loss {float}, gradnorm {float}, max|logit| {float}, time {float}ms

If train.py's print format ever changes, this file is the single place to fix it -- and
`--strict` makes a parse that finds no eval points a hard error rather than a silent
empty CSV.
"""
import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

RE_STEP = re.compile(r"^step\s+(\d+):\s+train loss\s+([-\d.naif]+),\s+val loss\s+([-\d.naif]+)")
RE_ITER = re.compile(r"^iter\s+(\d+):\s+loss\s+([-\d.naif]+),\s+gradnorm\s+([-\d.naifn]+),"
                     r"\s+max\|logit\|\s+([-\d.naif]+),\s+time\s+([\d.]+)ms")
RE_PARAM = re.compile(r"number of parameters:\s+([\d.]+)M")
RE_COHORT = re.compile(r"cohort filter.*?train\s+(\d+)\s*->\s*(\d+)\s*\|\s*val\s+(\d+)\s*->\s*(\d+)")
RE_NANGUARD = re.compile(r"\[nan-guard\]")


def f(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def parse_log(path):
    steps, iters = [], []
    reported_M = None
    cohort = None
    nan_guard = False
    with open(path, errors="replace") as fh:
        for line in fh:
            m = RE_STEP.match(line)
            if m:
                steps.append((int(m.group(1)), f(m.group(2)), f(m.group(3))))
                continue
            m = RE_ITER.match(line)
            if m:
                iters.append((int(m.group(1)), f(m.group(2)), f(m.group(3)), f(m.group(4)),
                              f(m.group(5))))
                continue
            m = RE_PARAM.search(line)
            if m:
                reported_M = f(m.group(1))
            m = RE_COHORT.search(line)
            if m:
                cohort = tuple(int(g) for g in m.groups())
            if RE_NANGUARD.search(line):
                nan_guard = True
    return dict(steps=steps, iters=iters, reported_M=reported_M, cohort=cohort,
                nan_guard=nan_guard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any run has no eval points")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    logs = sorted(glob.glob(os.path.join(a.runs, "*", "train.log")))
    if not logs:
        sys.exit(f"[collect] no runs/*/train.log under {a.runs}")

    crows, srows, problems = [], [], []
    for lp in logs:
        rdir = os.path.dirname(lp)
        run = os.path.basename(rdir)
        meta = {}
        mp = os.path.join(rdir, "run_meta.json")
        if os.path.exists(mp):
            try:
                meta = json.load(open(mp))
            except json.JSONDecodeError:
                problems.append(f"{run}: run_meta.json is malformed")
        p = parse_log(lp)

        if not p["steps"]:
            problems.append(f"{run}: no `step N:` eval lines -- run died early or never eval'd")
            continue

        base = dict(run=run, tag=meta.get("tag", run.rsplit("_s", 1)[0]),
                    seed=meta.get("seed", int(run.rsplit("_s", 1)[-1]) if "_s" in run else -1),
                    n_params=meta.get("n_params"), n_layer=meta.get("n_layer"),
                    n_head=meta.get("n_head"), n_embd=meta.get("n_embd"),
                    head_dim=meta.get("head_dim"))

        for it, tr, va in p["steps"]:
            crows.append(dict(**base, iter=it, train_loss=tr, val_loss=va))

        df = pd.DataFrame(p["steps"], columns=["iter", "train", "val"])
        fin = df[df["val"].notna() & (df["val"] < 1e8)]
        if fin.empty:
            problems.append(f"{run}: every val loss is nan/inf")
            continue
        b = fin.loc[fin["val"].idxmin()]
        ms = pd.DataFrame(p["iters"], columns=["it", "loss", "gn", "mx", "ms"])["ms"]
        # drop the first 20 iters: they carry CUDA context init and are not steady state
        ms_steady = float(ms.iloc[20:].median()) if len(ms) > 25 else float("nan")

        srows.append(dict(
            **base,
            reported_M=p["reported_M"],
            cohort_train=p["cohort"][1] if p["cohort"] else None,
            cohort_val=p["cohort"][3] if p["cohort"] else None,
            n_evals=len(fin),
            max_iter=int(df["iter"].max()),
            best_iter=int(b["iter"]), best_val=float(b["val"]),
            train_at_best=float(b["train"]), gap_at_best=float(b["val"] - b["train"]),
            final_val=float(fin.iloc[-1]["val"]),
            rise_after_best=float(fin.iloc[-1]["val"] - b["val"]),
            best_frac_of_schedule=float(b["iter"]) / max(float(df["iter"].max()), 1),
            ms_per_iter=ms_steady,
            wall_min=(meta.get("wall_seconds", 0) or 0) / 60.0,
            gpu=meta.get("gpu"), nan_guard=p["nan_guard"],
        ))

    cur = pd.DataFrame(crows).sort_values(["tag", "seed", "iter"])
    smy = pd.DataFrame(srows).sort_values(["n_params", "tag", "seed"], ascending=[False, True, True])
    cur.to_csv(os.path.join(a.out, "curves.csv"), index=False)
    smy.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print(f"[collect] {len(logs)} logs -> {len(smy)} runs, {len(cur)} eval points")
    print(f"          {a.out}/curves.csv  {a.out}/summary.csv")
    if problems:
        print("\n[collect] PROBLEMS:")
        for x in problems:
            print("  -", x)
    if not smy.empty:
        cols = ["run", "n_params", "best_iter", "best_val", "gap_at_best", "rise_after_best",
                "ms_per_iter", "wall_min"]
        print("\n" + smy[cols].to_string(index=False))
    if a.strict and problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
