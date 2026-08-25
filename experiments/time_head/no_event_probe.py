"""
no_event_probe.py -- what the synthetic No-event token does to the timing objective.

    python experiments/time_head/no_event_probe.py --runs <dir> --tags L8E120H6_s42

Four measurements, no training. Each answers one step of the argument that the model's timing
calibration may be stored somewhere inference throws away.

BACKGROUND -- what get_batch actually inserts (utils.py:104-113)
---------------------------------------------------------------
Not "a filler when nothing happened": it drops ~20 No-event tokens (model token 1) onto the
0-100 y axis UNCONDITIONALLY, then deletes the ones after the subject's last real event. It
can therefore land right beside a real visit and split a real gap in two. And the two callers
disagree:

    train.py's training loop        padding='random'   20 uniformly random ages in [0, 100) y
    estimate_loss / decompose /     padding='regular'  a deterministic 5-y grid, identical
    time_head_probe (the default)                      for every subject

`ignore_tokens = range(22)` contains token 1, so any position whose TARGET is a No-event token
is dropped from BOTH losses. Two consequences pull in opposite directions:

  (1) SHORT BIAS. A real gap interrupted by an inserted token loses the supervision at its
      real start; what survives is the truncated remainder. Long gaps are interrupted more
      often, so the supervised dt distribution is biased short -> lambda too high -> the model
      predicts too EARLY.

  (2) A DISCARDED KNOB. Training applies no mask, so lambda = logsumexp over ALL 111 logits,
      No-event included -- and No-event is never a target, so loss_ce never constrains it,
      only pushes it down as one of 110 non-target classes. loss_dt can therefore park the
      total-rate calibration on a logit that costs it nothing. Generation and
      validation_loss_mode both mask tokens 0-21 out, so that part of the rate is deleted:
      the clock at inference runs 1/P(content) times slower than the clock that was trained.
      -> the model predicts too LATE.

Which dominates decides whether the +4 y bias in the Figure-2 timing metrics is caused here.
Two numbers in the repo disagree by 3x about the size of (2): ad_engine.py's comment says
No-event absorbs ~75% of the probability mass (=> inference 4x slower), while backing it out
of time_head_probe's numbers (loss_dt is minimised at E[dt] = the target MEAN, 1.95 y; the
probe measures 2.28 y) suggests P(content) ~ 0.85 (=> 1.17x). This measures it directly.

WHAT IS REPORTED
  [1] where the rate mass sits: P(No-event), P(tokens 0-21), P(content), lambda_full/lambda_content
  [2] the supervised dt distribution vs the TRUE real-visit-to-real-visit gap distribution
      (the latter read straight off the .bin, with no get_batch and no synthetic insertion)
  [3] padding='random' (training) vs 'regular' (validation), same checkpoint, same seeds
  [4] a sweep of no_event_token_rate: how all of the above moves

Aggregate scalars only -- safe to copy off a PHI-holding machine.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(os.path.dirname(HERE), "capacity")
PKG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Delphi-2M")
sys.path.insert(0, PKG)
sys.path.insert(0, CAP)
from delphi.model import Delphi, DelphiConfig                  # noqa: E402
from delphi.utils import get_p2i, get_batch                    # noqa: E402
from time_head_probe import target_dt                          # noqa: E402  (verified vs model.py)

DAY = 365.25


def filter_cohort(data, p2i, min_visits=4, short_min_visits=2, udsd_disk=(105, 106, 107, 108)):
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


def true_visit_gaps(data, p2i):
    """Real-visit-to-real-visit gaps in years, straight off the .bin.

    No get_batch, so no synthetic tokens: this is the gap distribution the model is actually
    asked to reproduce at generation time. Ages of 0 are the static/background tokens (sex,
    APOE, ...), not visits, so they are excluded.
    """
    out = []
    ages = np.asarray(data[:, 1])
    for k in range(len(p2i)):
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        a = np.unique(ages[s:s + n])
        a = a[a > 0]
        if a.size >= 2:
            out.append(np.diff(a))
    return np.concatenate(out) / DAY if out else np.zeros(0)


def q(x, name, extra="", prec=2):
    """Quantiles on one line. `prec` matters: P(content) can be a fraction of a percent, and
    the first draft printed it as '0.00' and hid the whole point."""
    if len(x) == 0:
        return f"  {name:34s} (empty)"
    p = np.percentile(x, [5, 25, 50, 75, 95])
    w = prec + 5
    return (f"  {name:34s} n={len(x):>8,}  p5 {p[0]:{w}.{prec}f}  p25 {p[1]:{w}.{prec}f}  "
            f"med {p[2]:{w}.{prec}f}  p75 {p[3]:{w}.{prec}f}  p95 {p[4]:{w}.{prec}f}  "
            f"mean {x.mean():{w}.{prec}f}" + (f"  {extra}" if extra else ""))


def load(cp):
    c = torch.load(cp, map_location="cpu", weights_only=False)
    args = {k: v for k, v in c["model_args"].items() if k != "_ckpt_sig"}
    valid = set(DelphiConfig.__dataclass_fields__)
    m = Delphi(DelphiConfig(**{k: v for k, v in args.items() if k in valid}))
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in c["model"].items()})
    m.eval()
    return m, args


@torch.no_grad()
def sweep(m, args, data, p2i, batches, bs, seed, padding, rate):
    """One (padding, rate) setting. Returns the per-position aggregates."""
    ignore = list(m.config.ignore_tokens)
    NOEV = 1
    keep_dt, mass_no, mass_ign, lam_ratio = [], [], [], []
    n_tok = n_noev = n_pad = 0
    n_pos = n_scored = n_dropped_noev = 0
    ce, dt_loss = [], []
    torch.manual_seed(seed)
    for _ in range(batches):
        ix = torch.randint(len(p2i), (bs,))
        X, A, Y, B = get_batch(ix, data, p2i, block_size=args["block_size"], device="cpu",
                               padding=padding, lifestyle_augmentations=(padding == "random"),
                               select="left", no_event_token_rate=rate, cut_batch=True)
        logits, loss, _ = m(X, A, Y, B)
        ce.append(float(loss["loss_ce"])); dt_loss.append(float(loss["loss_dt"]))

        # ---- composition of what the model is fed
        n_tok += int(X.numel()); n_noev += int((X == NOEV).sum()); n_pad += int((X == 0).sum())

        # ---- which positions the loss scores, and which it loses to a No-event target
        scored = (Y != -1)
        for k in ignore:
            scored = scored & (Y != k)
        real_pos = (X != 0)
        n_pos += int(real_pos.sum()); n_scored += int(scored.sum())
        n_dropped_noev += int((real_pos & (Y == NOEV)).sum())

        # ---- the supervised dt, exactly as model.py computes it
        d = target_dt(X, A, B, m.config.mask_ties)
        keep_dt.append((d[scored] / DAY).numpy())

        # ---- where the rate mass sits (training mode -> logits are UNMASKED)
        p = torch.softmax(logits, -1)
        mass_no.append(p[..., NOEV][scored].numpy())
        mass_ign.append(p[..., ignore].sum(-1)[scored].numpy())
        lam_full = torch.logsumexp(logits, -1)
        masked = logits.clone(); masked[..., ignore] = -torch.inf
        lam_cont = torch.logsumexp(masked, -1)
        lam_ratio.append(torch.exp(lam_full - lam_cont)[scored].numpy())   # = 1 / P(content)

    cat = lambda L: np.concatenate(L) if L else np.zeros(0)
    return dict(dt=cat(keep_dt), p_noev=cat(mass_no), p_ign=cat(mass_ign),
                lam_ratio=cat(lam_ratio), loss_ce=float(np.mean(ce)),
                loss_dt=float(np.mean(dt_loss)),
                frac_noev_input=n_noev / max(n_tok, 1), frac_pad_input=n_pad / max(n_tok, 1),
                frac_scored=n_scored / max(n_pos, 1),
                frac_dropped_noev=n_dropped_noev / max(n_pos, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--runs", default=None)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--data-root", default=os.path.join(PKG, "data"))
    ap.add_argument("--dataset", default="nacc-dedup-s42")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rates", default="0,2,5,10")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ckpts = list(a.ckpt)
    if a.runs:
        want = set(a.tags.split(",")) if a.tags else None
        for p in sorted(glob.glob(os.path.join(a.runs, "*", "ckpt.pt"))):
            run = os.path.basename(os.path.dirname(p))
            if want is None or run in want:
                ckpts.append(p)
    if not ckpts:
        sys.exit("[no_event_probe] no checkpoints (--ckpt or --runs)")

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    train = np.memmap(os.path.join(a.data_root, a.dataset, "train.bin"),
                      dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = filter_cohort(train, get_p2i(train))
    print(f"train split: {len(train):,} events, cohort {len(p2i):,} patients")

    # ---------------------------------------------------------------- the ground truth
    gaps = true_visit_gaps(train, p2i)
    print("\n" + "=" * 100)
    print("[0] THE TRUE TARGET -- real-visit-to-real-visit gaps, read off the .bin "
          "(no get_batch, no synthetic tokens)")
    print("=" * 100)
    print(q(gaps, "true visit gaps (years)", f"cv {gaps.std()/gaps.mean():.3f}", prec=3))
    print(f"  {'':34s} an exponential has cv = 1 exactly, and ANY mixture of exponentials has "
          f"cv >= 1")

    rows = []
    for cp in ckpts:
        name = os.path.basename(os.path.dirname(cp)) or os.path.basename(cp)
        m, args = load(cp)
        rate0 = 5

        print("\n" + "=" * 100)
        print(f"{name}   {args['n_layer']}L/{args['n_head']}H/{args['n_embd']}d   "
              f"t_min {m.config.t_min:.2f} d   ignore_tokens 0-{max(m.config.ignore_tokens)}")
        print("=" * 100)

        # ---------------------------------------------- [1] + [3] training vs validation
        res = {}
        for pad in ("random", "regular"):
            res[pad] = sweep(m, args, train, p2i, a.batches, a.batch_size, a.seed, pad, rate0)

        print(f"\n[1] WHERE THE RATE MASS SITS  (training mode, logits UNMASKED, "
              f"padding='random' = what train.py uses)")
        r = res["random"]
        print(q(r["p_noev"] * 100, "P(No-event) %", prec=3))
        print(q(r["p_ign"] * 100, "P(tokens 0-21, all ignored) %", prec=3))
        print(q((1 - r["p_ign"]) * 100, "P(content 22-110) %", prec=3))
        print(q(r["lam_ratio"], "lambda_full / lambda_content", prec=3))
        print(f"\n  -> generation and validation mask tokens 0-21, so their clock is "
              f"{np.median(r['lam_ratio']):.3f}x SLOWER (median) than the clock that was trained.")
        print(f"     ad_engine's comment implies ~4x; backing it out of time_head_probe implies "
              f"~1.17x. Measured: {np.median(r['lam_ratio']):.3f}x")

        print(f"\n[2] THE SUPERVISED dt vs THE TRUE GAP DISTRIBUTION")
        print(q(gaps, "true visit gaps (years)", prec=3))
        print(q(r["dt"], "dt loss_dt actually sees (years)", prec=3))
        sb = np.median(r["dt"]) / np.median(gaps) if len(gaps) else float("nan")
        print(f"\n  -> the supervised gap is {sb:.3f}x the true gap at the median "
              f"({'SHORT' if sb < 1 else 'LONG'}-biased)")
        print(f"     {100*r['frac_dropped_noev']:.1f}% of real input positions lose their "
              f"supervision because the next token in the merged stream is a No-event token")
        print(f"     {100*r['frac_noev_input']:.1f}% of the non-padding tokens the model is fed "
              f"are synthetic No-event; {100*r['frac_scored']:.1f}% of positions are scored")

        print(f"\n[3] padding='random' (train.py) vs 'regular' (estimate_loss, decompose, probe)")
        print(f"  {'':22s}{'loss_ce':>10s}{'loss_dt':>10s}{'med dt (y)':>12s}"
              f"{'% dropped to No-ev':>20s}{'lam ratio':>11s}{'% input No-ev':>15s}")
        for pad in ("random", "regular"):
            x = res[pad]
            print(f"  {pad:<22s}{x['loss_ce']:>10.4f}{x['loss_dt']:>10.4f}"
                  f"{np.median(x['dt']):>12.3f}{100*x['frac_dropped_noev']:>20.1f}"
                  f"{np.median(x['lam_ratio']):>11.3f}{100*x['frac_noev_input']:>15.1f}")

        print(f"\n[4] SWEEP no_event_token_rate  (padding='random', one per N years; 0 = off)")
        print(f"  {'rate':>6s}{'loss_ce':>10s}{'loss_dt':>10s}{'med dt (y)':>12s}"
              f"{'% dropped':>11s}{'lam ratio':>11s}{'P(No-ev) %':>12s}{'% input No-ev':>15s}")
        for rt in [float(x) for x in a.rates.split(",")]:
            x = sweep(m, args, train, p2i, max(6, a.batches // 3), a.batch_size, a.seed,
                      "random", rt)
            print(f"  {rt:>6g}{x['loss_ce']:>10.4f}{x['loss_dt']:>10.4f}"
                  f"{np.median(x['dt']):>12.3f}{100*x['frac_dropped_noev']:>11.1f}"
                  f"{np.median(x['lam_ratio']):>11.3f}"
                  f"{100*np.median(x['p_noev']):>12.3f}{100*x['frac_noev_input']:>15.1f}")
            rows.append(dict(run=name, setting=f"rate={rt:g}", **{
                k: v for k, v in x.items() if not isinstance(v, np.ndarray)},
                med_dt_y=float(np.median(x["dt"])),
                med_lam_ratio=float(np.median(x["lam_ratio"])),
                med_p_noev=float(np.median(x["p_noev"]))))

        for pad in ("random", "regular"):
            x = res[pad]
            rows.append(dict(run=name, setting=f"padding={pad}", **{
                k: v for k, v in x.items() if not isinstance(v, np.ndarray)},
                med_dt_y=float(np.median(x["dt"])),
                med_lam_ratio=float(np.median(x["lam_ratio"])),
                med_p_noev=float(np.median(x["p_noev"]))))

    print(f"\n[ref] true visit gaps: median {np.median(gaps):.3f} y, mean {gaps.mean():.3f} y, "
          f"cv {gaps.std()/gaps.mean():.3f}")
    if a.out:
        import pandas as pd
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        df = pd.DataFrame(rows)
        df["true_gap_med_y"] = float(np.median(gaps))
        df["true_gap_mean_y"] = float(gaps.mean())
        df["true_gap_cv"] = float(gaps.std() / gaps.mean())
        df.to_csv(a.out, index=False)
        print(f"[no_event_probe] -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
