"""
time_head_probe.py -- what does the timing objective actually see, and does the model track it?

THERE IS NO TIME HEAD. Delphi has one output projection (`lm_head`, vocab-wide, weight-tied to
the token embedding). Both objectives are read off that single logits vector:

    loss_ce = cross_entropy(logits, next_token)                  -- WHICH event
    lambda  = logsumexp(logits), floored via t_min               -- WHEN, as one scalar
    loss_dt = lambda*t - log(lambda)                                (exponential NLL)

So "when" is ONE number per position, and it is a deterministic function of the same logits
that decide "which". The capacity sweep found loss_dt inert -- 73% of validation loss, moved
by 0.10% of itself over a 6.9x parameter range. This probe asks why.

GETTING THE TARGET RIGHT IS THE WHOLE DIFFICULTY. Two things make a naive comparison wrong,
and both were hit while writing this file:

  1. validation_loss_mode drops ignore_tokens (0..21) AND the No-event token from the loss.
     A naive (X>0)&(Y>0) mask folds in get_batch's synthetic no-event grid, which sits at
     EXACTLY no_event_token_rate-year spacing -- so the "observed" gaps show a fake spike at
     5.0 y that loss_dt never sees.
  2. With mask_ties=True the target time is NOT targets_age - age. model.py gathers dt from
     the last UNTIED token, so same-visit tokens are timed from the previous distinct visit,
     not from their same-visit predecessor. Skipping the gather makes >75% of targets look
     like the 1-day clamp floor.

Both are handled below by copying model.py's own mask + gather. That copy is then VERIFIED:
the probe recomputes loss_dt from its own (lambda, t) and asserts it matches the value the
model returns. If the replication ever drifts from model.py, the assert fires instead of the
probe quietly reporting a wrong distribution.

    python experiments/capacity/time_head_probe.py --runs <dir> --data-root <dir>
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
from delphi.model import Delphi, DelphiConfig          # noqa: E402
from delphi.utils import get_p2i, get_batch            # noqa: E402
sys.path.insert(0, HERE)
from decompose import filter_cohort                    # noqa: E402

DAY = 365.25


def target_dt(idx, age, targets_age, mask_ties):
    """model.py's dt, verbatim: the attention mask, then the last-untied-token gather."""
    dev = idx.device
    T = idx.size(1)
    tril = torch.tril(torch.ones(T, T, device=dev))[None, None, :, :] > 0
    am = (idx > 0).view(idx.size(0), 1, 1, T) * (idx > 0).view(idx.size(0), 1, T, 1)
    am = am * tril
    if mask_ties:
        am = am * (age.view(idx.size(0), 1, 1, T) != targets_age.view(idx.size(0), 1, T, 1))
        am = am + (am.sum(-1, keepdim=True) == 0) * (torch.diag(torch.ones(T, device=dev)) > 0)
    am = am + (idx == 0).view(idx.size(0), 1, 1, T) * (torch.diag(torch.ones(T, device=dev)) > 0)
    am = am * tril
    dt = torch.clamp(targets_age - age, min=1.0)
    if mask_ties:
        pos = torch.arange(0, T, device=dev, dtype=torch.float32).view(1, 1, 1, -1)
        dt = torch.gather(dt, -1, (am * pos).max(-1).indices.squeeze((1, 2)))
    return dt


@torch.no_grad()
def probe(model, data, p2i, cfg, n_batches=40, batch_size=128, seed=42):
    torch.manual_seed(seed)
    t_min, ignore = cfg["t_min"], list(cfg.get("ignore_tokens", [0]))
    pred, obs = [], []
    max_err = 0.0
    for _ in range(n_batches):
        ix = torch.randint(len(p2i), (batch_size,))
        X, A, Y, B = get_batch(ix, data, p2i, block_size=cfg["block_size"], device="cpu",
                               select="left", no_event_token_rate=5, cut_batch=True)
        logits, loss, _ = model(X, A, Y, B, validation_loss_mode=True)
        lse = torch.clamp(torch.logsumexp(logits, -1), min=-60.0, max=60.0)
        lse = -torch.log(torch.exp(-lse) + t_min)          # log of the FLOORED intensity
        dt = target_dt(X, A, B, model.config.mask_ties)

        keep = (Y != -1)
        for k in ignore + [1]:
            keep = keep & (Y != k)
        # --- verify the replication against the model's own number -------------------
        ldt = -torch.log(dt + t_min)
        mine = -(lse - torch.exp(lse - ldt))
        mine = mine.reshape(-1)[keep.reshape(-1)].mean()
        max_err = max(max_err, abs(float(mine) - float(loss["loss_dt"])))
        # -----------------------------------------------------------------------------
        pred.append((torch.exp(-lse) + t_min)[keep].numpy())      # model's E[dt], days
        obs.append((dt + t_min)[keep].numpy())                    # the target t, days
    assert max_err < 1e-3, f"replication of model.py's dt drifted: max |Δloss_dt| = {max_err:.2e}"
    return np.concatenate(pred), np.concatenate(obs), max_err


def stats(x, label):
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(what=label, n=len(x), p5_y=q[0] / DAY, p25_y=q[1] / DAY, median_y=q[2] / DAY,
                p75_y=q[3] / DAY, p95_y=q[4] / DAY, iqr_y=(q[3] - q[1]) / DAY,
                mean_y=x.mean() / DAY, cv=x.std() / x.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--data-root", default=os.path.join(PKG, "data"))
    ap.add_argument("--dataset", default="nacc-dedup-s42")
    ap.add_argument("--seed-only", type=int, default=42)
    ap.add_argument("--n-batches", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "time_head_probe.csv"))
    a = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 4))
    val = np.memmap(os.path.join(a.data_root, a.dataset, "val.bin"),
                    dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = filter_cohort(val, get_p2i(val))

    rows, obs_done, corr = [], False, []
    for mp in sorted(glob.glob(os.path.join(a.runs, "*", "run_meta.json"))):
        meta = json.load(open(mp))
        rdir = os.path.dirname(mp)
        # run_meta's "run" omits the _long15000 suffix -- filter on the directory name
        if meta["seed"] != a.seed_only or "long" in os.path.basename(rdir):
            continue
        ck = os.path.join(rdir, "ckpt.pt")
        if not os.path.exists(ck):
            continue
        c = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = {k: v for k, v in c["model_args"].items() if k != "_ckpt_sig"}
        m = Delphi(DelphiConfig(**cfg))
        m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in c["model"].items()})
        m.eval()
        pred, obs, err = probe(m, val, p2i, cfg, a.n_batches)
        r = stats(pred, meta["tag"]); r["n_params"] = meta["n_params"]; r["repl_err"] = err
        rows.append(r)
        corr.append((meta["tag"],
                     float(np.corrcoef(np.log(pred), np.log(obs))[0, 1]),
                     float(pd.Series(pred).corr(pd.Series(obs), method="spearman"))))
        if not obs_done:
            o = stats(obs, "TARGET t (what loss_dt sees)"); o["n_params"] = 0; o["repl_err"] = 0
            rows.append(o); obs_done = True

    df = pd.DataFrame(rows).sort_values("n_params", ascending=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print("\nAll times in YEARS, over exactly the positions loss_dt scores")
    print(df[["what", "n_params", "n", "p5_y", "p25_y", "median_y", "p75_y", "p95_y",
              "iqr_y", "mean_y", "cv"]].round(3).to_string(index=False))
    print(f"\nreplication check: max |computed loss_dt - model loss_dt| = "
          f"{df.repl_err.max():.2e}  (assert < 1e-3)")
    print("\ndoes the predicted intensity TRACK the target?")
    print(f"  {'shape':<12}{'pearson(log)':>14}{'spearman':>11}")
    for t, p, s in corr:
        print(f"  {t:<12}{p:>14.4f}{s:>11.4f}")
    print(f"[probe] -> {a.out}")


if __name__ == "__main__":
    main()
