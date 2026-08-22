"""
decompose.py -- split each arm's validation loss into its two terms.

    python experiments/capacity/decompose.py --runs <dir> --out results/decomposition.csv

WHY THIS EXISTS: train.py prints only `val loss` = loss_ce + loss_dt. The two terms answer
completely different questions --

    loss_ce   given that an event happens, WHICH event is it (categorical cross-entropy over
              the 89 non-ignored tokens)
    loss_dt   WHEN does it happen (negative log-likelihood of the competing-exponential
              time-to-event model)

-- and they are not on a common scale. If loss_dt sits near a floor set by the data's timing
entropy while loss_ce varies, then a "0.3% change in total val loss" can be hiding a large
change in the part of the model anyone actually cares about, or vice versa. Reporting only
the sum makes a capacity comparison uninterpretable.

The per-term values exist inside train.py's estimate_loss() but are only ever sent to wandb,
which the sweep runs with disabled. Rather than patch the training script, this reproduces
estimate_loss() exactly -- same get_batch arguments, same validation_loss_mode=True, same
eval_iters, same batch_size, same generator seeding -- against each run's saved best
checkpoint.

NOTE the checkpoint IS the best-val one: train.py writes ckpt.pt only when val loss improves
(always_save_checkpoint defaults to False), so this measures each shape at its own early-stop
point, which is the checkpoint that would actually be used.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Delphi-2M")
sys.path.insert(0, PKG)
from delphi.model import Delphi, DelphiConfig      # noqa: E402
from delphi.utils import get_p2i, get_batch        # noqa: E402


def filter_cohort(data, p2i, min_visits=4, short_min_visits=2, udsd_disk=(105, 106, 107, 108)):
    """Byte-for-byte the rule in train.py:106 -- the val set must be the SAME population the
    run validated against, or the numbers are not the ones the checkpoint was selected on."""
    ages = np.asarray(data[:, 1]); toks = np.asarray(data[:, 2])
    udsd = np.asarray(udsd_disk)
    keep = np.zeros(len(p2i), dtype=bool)
    for k in range(len(p2i)):
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        nv = np.unique(ages[s:s + n][ages[s:s + n] > 0]).size
        if nv >= min_visits:
            keep[k] = True
        elif nv >= short_min_visits:
            keep[k] = int(np.isin(toks[s:s + n], udsd).sum()) >= 2
    return p2i[keep]


@torch.no_grad()
def val_terms(model, data, p2i, eval_iters, batch_size, block_size, seed, device="cpu"):
    """estimate_loss()'s val branch, with the two terms kept apart instead of summed."""
    model.eval()
    torch.manual_seed(seed)          # same RNG entry point train.py's eval sees
    ce, dt = [], []
    for _ in range(eval_iters):
        ix = torch.randint(len(p2i), (batch_size,))
        X, A, Y, B = get_batch(ix, data, p2i, block_size=block_size, device=device,
                               select="left", no_event_token_rate=5, cut_batch=True)
        _, loss, _ = model(X, A, Y, B, validation_loss_mode=True)
        ce.append(float(loss["loss_ce"])); dt.append(float(loss["loss_dt"]))
    return float(np.mean(ce)), float(np.mean(dt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--data-root", default=os.path.join(PKG, "data"))
    ap.add_argument("--dataset", default="nacc-dedup-s42")
    ap.add_argument("--eval-iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "decomposition.csv"))
    a = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4)))
    val = np.memmap(os.path.join(a.data_root, a.dataset, "val.bin"),
                    dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i_all = get_p2i(val)
    p2i = filter_cohort(val, p2i_all)
    print(f"val cohort: {len(p2i_all)} -> {len(p2i)} patients (train.py's 4/2 filter)")

    rows = []
    for mp in sorted(glob.glob(os.path.join(a.runs, "*", "run_meta.json"))):
        rdir = os.path.dirname(mp)
        meta = json.load(open(mp))
        ck = os.path.join(rdir, "ckpt.pt")
        if not os.path.exists(ck):
            print(f"  {meta['run']}: no ckpt.pt, skipped"); continue
        c = torch.load(ck, map_location="cpu", weights_only=False)
        args = {k: v for k, v in c["model_args"].items() if k != "_ckpt_sig"}
        m = Delphi(DelphiConfig(**args))
        m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in c["model"].items()})
        ce, dt = val_terms(m, val, p2i, a.eval_iters, a.batch_size,
                           args["block_size"], seed=meta["seed"])
        rows.append(dict(run=meta["run"], tag=meta["tag"], seed=meta["seed"],
                         n_params=meta["n_params"], ckpt_iter=c["iter_num"],
                         ckpt_best_val=c["best_val_loss"],
                         loss_ce=ce, loss_dt=dt, total=ce + dt))
        print(f"  {meta['run']:<16} iter {c['iter_num']:>4}  ce {ce:.4f}  dt {dt:.4f}  "
              f"sum {ce+dt:.4f}  (ckpt records {c['best_val_loss']:.4f})")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)

    g = df.groupby("tag")
    agg = pd.DataFrame({"n_params": g["n_params"].first(),
                        "ce": g["loss_ce"].mean(), "ce_sd": g["loss_ce"].std(),
                        "dt": g["loss_dt"].mean(), "dt_sd": g["loss_dt"].std(),
                        "total": g["total"].mean()}).sort_values("n_params", ascending=False)
    print("\n" + "=" * 74)
    print(agg.to_string())
    base = agg.index[0]
    print(f"\nrelative to {base}:")
    for t in agg.index:
        dce = agg.loc[t, "ce"] - agg.loc[base, "ce"]
        ddt = agg.loc[t, "dt"] - agg.loc[base, "dt"]
        print(f"  {t:<12} ce {dce:+.4f} ({100*dce/agg.loc[base,'ce']:+.2f}%)   "
              f"dt {ddt:+.4f} ({100*ddt/agg.loc[base,'dt']:+.2f}%)")
    print(f"\n-> loss_dt is {100*agg['dt'].mean()/agg['total'].mean():.1f}% of the total; "
          f"its spread across shapes is {agg['dt'].max()-agg['dt'].min():.4f}, "
          f"loss_ce's is {agg['ce'].max()-agg['ce'].min():.4f}")
    print(f"[decompose] -> {a.out}")


if __name__ == "__main__":
    main()
