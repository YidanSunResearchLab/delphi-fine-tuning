"""
grad_balance.py -- which objective actually drives training, and how much of its gradient
does the t_min floor let through? No training, one forward + two backwards per batch.

    python experiments/time_head/grad_balance.py --ckpt <path> [--ckpt <path> ...]
    python experiments/time_head/grad_balance.py --runs <dir> --tags L8E120H6_s42,...

WHY THIS EXISTS
---------------
`loss = loss_ce + loss_dt`, an unweighted sum, and `loss_dt` is ~73% of the reported value.
That 73% is NOT evidence that the timing objective dominates training:

  * loss_ce is a discrete NLL -- unit-free.
  * loss_dt is a continuous-density NLL -- it carries the unit of dt. Re-expressing dt in
    years instead of days subtracts log(365.25) = 5.90 from it, taking its share from 73% to
    35%, with the SAME weights and a bit-identical gradient (verified: max |Δgrad| = 7e-09).

So the share is partly an accounting artifact. What decides how much each term steers the
optimiser is the size of its GRADIENT, and that is unit-invariant. This measures it.

It also measures the thing that can silently destroy loss_dt's gradient. model.py floors the
intensity:

    E[Δt] = exp(-lse) + t_min ,   t_min = 365.25/12 ≈ 30.44 d

The derivative of the floored log-intensity w.r.t. the raw one is

    d/d(lse) [ -log(exp(-lse) + t_min) ]  =  raw / (raw + t_min) ,   raw = exp(-lse)

-- a damping factor in (0, 1). When the model's raw expected gap is small next to t_min the
floor SATURATES and the timing objective stops transmitting gradient at all, no matter how
wrong it is. On a synthetic fixture this was measured at 1.6e-04 with 100% of positions
saturated, which is exactly why that measurement had to be thrown away and this one run on
the real split.

WHAT IS MEASURED, AND IN WHICH MODE
-----------------------------------
The question is about the gradient that actually drove TRAINING, so the primary numbers
replicate train.py's own batch call (train split, padding='random',
lifestyle_augmentations=True, no validation_loss_mode) rather than estimate_loss()'s. The
val-mode losses are reported alongside for continuity with decompose.py, but no gradient is
taken there -- that mode writes -inf into the ignored columns and is a reporting path.

The t_min scan is POST-HOC sensitivity: these weights were trained at t_min=30.44, so it says
"how much does the floor throttle this model", not "what would training at another floor
give". Still the cheapest way to find out whether the floor is worth an arm of its own.

Outputs aggregate scalars only -- no per-patient anything -- so the stdout is safe to copy off
a PHI-holding machine.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Delphi-2M")
sys.path.insert(0, PKG)
from delphi.model import Delphi, DelphiConfig                  # noqa: E402
from delphi.utils import get_p2i, get_batch                    # noqa: E402

DAY = 365.25


def filter_cohort(data, p2i, min_visits=4, short_min_visits=2, udsd_disk=(105, 106, 107, 108)):
    """train.py's 4/2 rule, verbatim -- the population these checkpoints were fitted on."""
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


def load(ckpt_path):
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = {k: v for k, v in c["model_args"].items() if k != "_ckpt_sig"}
    valid = {f for f in DelphiConfig.__dataclass_fields__}
    m = Delphi(DelphiConfig(**{k: v for k, v in args.items() if k in valid}))
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in c["model"].items()})
    m.eval()                      # dropout is 0 in every config; eval() only removes doubt
    return m, args, c.get("iter_num", -1)


def gnorm(loss, params, retain=True):
    g = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    tot = sum(float((x ** 2).sum()) for x in g if x is not None)
    per = {n: math.sqrt(float((x ** 2).sum())) for (n, _), x in zip(params_named, g)
           if x is not None}
    return math.sqrt(tot), per


def scored_mask(Y, ignore):
    keep = (Y != -1)
    for k in ignore:
        keep = keep & (Y != k)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--runs", default=None, help="a runs/ dir; adds every <run>/ckpt.pt")
    ap.add_argument("--tags", default=None, help="comma list restricting --runs")
    ap.add_argument("--data-root", default=os.path.join(PKG, "data"))
    ap.add_argument("--dataset", default="nacc-dedup-s42")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t-min-scan", default="1,7,30.4375,91.3125,365.25")
    ap.add_argument("--out", default=None, help="optional CSV of the aggregate numbers")
    a = ap.parse_args()

    ckpts = list(a.ckpt)
    if a.runs:
        want = set(a.tags.split(",")) if a.tags else None
        for p in sorted(glob.glob(os.path.join(a.runs, "*", "ckpt.pt"))):
            run = os.path.basename(os.path.dirname(p))
            if want is None or run in want:
                ckpts.append(p)
    if not ckpts:
        sys.exit("[grad_balance] no checkpoints given (--ckpt or --runs)")

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    ddir = os.path.join(a.data_root, a.dataset)
    train = np.memmap(os.path.join(ddir, "train.bin"), dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = filter_cohort(train, get_p2i(train))
    print(f"train split: {len(train):,} events, cohort {len(p2i):,} patients "
          f"(train.py's 4/2 filter)\n")

    rows = []
    for cp in ckpts:
        name = os.path.basename(os.path.dirname(cp)) or os.path.basename(cp)
        m, args, it = load(cp)
        t_min0 = m.config.t_min
        ignore = list(m.config.ignore_tokens)
        global params_named
        params_named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        params = [p for _, p in params_named]

        print("=" * 78)
        print(f"{name}   iter {it}   {args['n_layer']}L/{args['n_head']}H/{args['n_embd']}d"
              f"   t_min {t_min0:.4f} d   time_head={args.get('time_head', False)}")
        print("=" * 78)

        acc = dict(ce=[], dt=[], gce=[], gdt=[], damp=[], raw=[], sat=[])
        emb_share = []
        torch.manual_seed(a.seed)
        for _ in range(a.batches):
            ix = torch.randint(len(p2i), (a.batch_size,))
            # train.py's OWN batch call -- this is the gradient that drove training
            X, A, Y, B = get_batch(ix, train, p2i, block_size=args["block_size"], device="cpu",
                                   padding="random", lifestyle_augmentations=True,
                                   select="left", no_event_token_rate=5, cut_batch=True)
            logits, loss, _ = m(X, A, Y, B)
            ce, dt = loss["loss_ce"], loss["loss_dt"]
            n_ce, per_ce = gnorm(ce, params)
            n_dt, per_dt = gnorm(dt, params, retain=False)
            acc["ce"].append(float(ce)); acc["dt"].append(float(dt))
            acc["gce"].append(n_ce); acc["gdt"].append(n_dt)
            emb_share.append((per_dt.get("transformer.wte.weight", 0.0), n_dt))

            with torch.no_grad():
                lse = torch.clamp(torch.logsumexp(logits.detach(), -1), -60.0, 60.0)
                raw = torch.exp(-lse)                      # the model's RAW E[dt], days
                k = scored_mask(Y, ignore)
                raw = raw[k]
                damp = raw / (raw + t_min0)
                acc["raw"].append(raw.numpy()); acc["damp"].append(damp.numpy())
                acc["sat"].append(float((damp < 0.01).float().mean()))

        ce_m, dt_m = float(np.mean(acc["ce"])), float(np.mean(acc["dt"]))
        gce, gdt = np.array(acc["gce"]), np.array(acc["gdt"])
        raw = np.concatenate(acc["raw"]); damp = np.concatenate(acc["damp"])
        ratio = gdt / np.maximum(gce, 1e-30)

        print(f"\n  loss values (training mode, {a.batches} batches x {a.batch_size})")
        print(f"    loss_ce {ce_m:8.4f}   loss_dt {dt_m:8.4f}   "
              f"loss_dt is {100*dt_m/(ce_m+dt_m):.1f}% of the sum")
        print(f"    the same loss_dt with dt in YEARS: {dt_m - math.log(DAY):8.4f}   "
              f"-> {100*(dt_m-math.log(DAY))/(ce_m+dt_m-math.log(DAY)):.1f}% "
              f"(same weights, same gradient)")

        print(f"\n  GRADIENT norms over all {sum(p.numel() for p in params):,} parameters")
        print(f"    ||g_ce|| {gce.mean():10.4f} +/- {gce.std():.4f}")
        print(f"    ||g_dt|| {gdt.mean():10.4f} +/- {gdt.std():.4f}")
        print(f"    ||g_dt|| / ||g_ce|| = {ratio.mean():.4f}  "
              f"(median {np.median(ratio):.4f}, min {ratio.min():.4f}, max {ratio.max():.4f})")
        print(f"    -> the timing term supplies {100*gdt.mean()/(gdt.mean()+gce.mean()):.1f}% "
              f"of the gradient it takes to steer this model")
        es = np.array([s / max(t, 1e-30) for s, t in emb_share])
        print(f"    of ||g_dt||, {100*es.mean():.1f}% lands on the tied token embedding "
              f"(= the event head's own tensor)")

        print(f"\n  the t_min floor, on the positions the loss scores")
        print(f"    raw exp(-lse):  median {np.median(raw)/DAY:7.3f} y   "
              f"p5 {np.percentile(raw,5)/DAY:.3f} y   p95 {np.percentile(raw,95)/DAY:.3f} y")
        print(f"    damping raw/(raw+t_min): median {np.median(damp):.4f}   "
              f"mean {damp.mean():.4f}   p5 {np.percentile(damp,5):.4f}")
        print(f"    positions with damping < 0.01 (floor saturated): "
              f"{100*(damp<0.01).mean():.2f}%")

        print(f"\n  post-hoc t_min sensitivity (NOT retrained -- these weights saw "
              f"t_min={t_min0:.2f})")
        print(f"    {'t_min':>12s}{'loss_dt':>10s}{'||g_dt||':>11s}{'ratio vs ce':>13s}"
              f"{'mean damping':>14s}")
        scan = []
        for tm in [float(x) for x in a.t_min_scan.split(",")]:
            m.config.t_min = tm
            torch.manual_seed(a.seed)          # same batches for every t_min
            l_dt, n_dt_s, dmp = [], [], []
            for _ in range(max(4, a.batches // 4)):
                ix = torch.randint(len(p2i), (a.batch_size,))
                X, A, Y, B = get_batch(ix, train, p2i, block_size=args["block_size"],
                                       device="cpu", padding="random",
                                       lifestyle_augmentations=True, select="left",
                                       no_event_token_rate=5, cut_batch=True)
                lg, ls, _ = m(X, A, Y, B)
                l_dt.append(float(ls["loss_dt"]))
                n_dt_s.append(gnorm(ls["loss_dt"], params, retain=False)[0])
                with torch.no_grad():
                    r = torch.exp(-torch.clamp(torch.logsumexp(lg.detach(), -1), -60, 60))
                    r = r[scored_mask(Y, ignore)]
                    dmp.append(float((r / (r + tm)).mean()))
            unit = ("1 d" if tm == 1 else "1 wk" if abs(tm - 7) < .1 else
                    "1 mo" if abs(tm - DAY / 12) < .1 else
                    "3 mo" if abs(tm - DAY / 4) < .1 else "1 y" if abs(tm - DAY) < .1 else "")
            print(f"    {tm:9.3f} {unit:<4s}{np.mean(l_dt):10.4f}{np.mean(n_dt_s):11.4f}"
                  f"{np.mean(n_dt_s)/gce.mean():13.4f}{np.mean(dmp):14.4f}")
            scan.append((tm, np.mean(l_dt), np.mean(n_dt_s), np.mean(dmp)))
        m.config.t_min = t_min0

        rows.append(dict(run=name, iter=it, n_layer=args["n_layer"], n_head=args["n_head"],
                         n_embd=args["n_embd"], t_min=t_min0, loss_ce=ce_m, loss_dt=dt_m,
                         g_ce=float(gce.mean()), g_dt=float(gdt.mean()),
                         g_ratio=float(ratio.mean()),
                         g_dt_share=float(gdt.mean() / (gdt.mean() + gce.mean())),
                         g_dt_on_embedding=float(es.mean()),
                         raw_median_y=float(np.median(raw) / DAY),
                         damp_median=float(np.median(damp)),
                         frac_saturated=float((damp < 0.01).mean()),
                         scan=json.dumps([[round(x, 6) for x in s] for s in scan])))
        print()

    if a.out:
        import pandas as pd
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"[grad_balance] -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
