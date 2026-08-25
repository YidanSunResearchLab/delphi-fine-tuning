"""
dt_ablation.py -- attribute the inflation of loss_dt's target to its three causes.

    python experiments/time_head/dt_ablation.py --data-root <dir> [--dataset ...]

NO MODEL IS LOADED. The quantity under test -- the dt that loss_dt is trained on -- is a
function of get_batch and the mask_ties gather alone, not of any weight. So this is exact,
fast, and cannot be confounded by which checkpoint you point it at.

THE FINDING BEING EXPLAINED (no_event_probe.py, real split, 20x128 batches)
--------------------------------------------------------------------------
    true real-visit-to-real-visit gaps    median 1.112 y   mean 1.520 y
    dt that loss_dt actually sees         median 1.309 y   mean 2.648 y      <- 1.74x the mean
    ... and with the No-event insertion turned OFF  median 2.716 y           <- 2.44x

So loss_dt is fitting a target systematically longer than the real clinical gaps, which by
lambda = 1/E[t] makes every predicted time too long -- the direction of the +2.6 y bias in the
Figure-2 timing metrics. Three candidate causes, and the first draft of this analysis
attributed it to the wrong one:

  A  lifestyle_augmentations (utils.py:96)   tokens 4-12 (BMI, smoking, alcohol, education)
     have their age shifted by a uniform -20..+40 YEARS. They start at age 0 while real visits
     sit at 65-90, so a shifted lifestyle token followed by a real visit creates a supervised
     gap of decades. ON in train.py, OFF in estimate_loss.

  B  the mask_ties gather (model.py:390)     a position does not use its own dt but the dt of
     the last position it may attend to. For the k-1 non-final tokens of a k-token visit that
     is the gap BEFORE their own visit, not the gap after it -- the wrong direction, and a
     quantity already computable from their own input. Predicted effect: wrong target, but NOT
     a longer one, since backward and forward gaps share a marginal distribution.

  C  the No-event insertion (utils.py:104)   ~20 synthetic tokens per subject; they break ties
     and shorten gaps, so they PARTLY CANCEL A and B. Measured: turning them off makes the
     inflation worse (1.17x -> 2.44x at the median).

This runs the 2x2x2 and prints the marginal distribution of the supervised dt in each cell, so
each cause gets its own number instead of a story.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(os.path.dirname(HERE), "capacity")
PKG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Delphi-2M")
sys.path.insert(0, PKG)
sys.path.insert(0, CAP)
from delphi.model import next_visit_dt                         # noqa: E402  (the FIX, imported not copied)
from delphi.utils import get_p2i, get_batch                    # noqa: E402
from time_head_probe import target_dt                          # noqa: E402  (verified vs model.py)

DAY = 365.25
IGNORE = list(range(22))          # config/train_delphi2m_mask_dedup.py
LIFESTYLE = (4, 12)               # MODEL space; utils.py shifts DISK 3..11 == model 4..12


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
    out, ages = [], np.asarray(data[:, 1])
    for k in range(len(p2i)):
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        a = np.unique(ages[s:s + n]); a = a[a > 0]
        if a.size >= 2:
            out.append(np.diff(a))
    return np.concatenate(out) / DAY if out else np.zeros(0)


def row(x, label, ref_med, ref_mean):
    if len(x) == 0:
        return f"  {label:32s}      (no scored positions)"
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return (f"  {label:32s} n={len(x):>8,} {p[0]:7.3f}{p[1]:7.3f}{p[2]:8.3f}{p[3]:8.3f}"
            f"{p[4]:8.2f}{x.mean():8.3f}   {p[2]/ref_med:6.2f}x {x.mean()/ref_mean:6.2f}x")


HDR = (f"  {'cell':32s} {'':10s}{'p5':>7s}{'p25':>7s}{'med':>8s}{'p75':>8s}{'p95':>8s}"
       f"{'mean':>8s}   {'med/true':>7s} {'mean/true':>8s}")


def cell(data, p2i, batches, bs, seed, block_size, lifestyle, ties, rate, padding="random"):
    """The supervised dt (years) for one ablation cell, plus the tie decomposition."""
    dts, dts_redir, dts_own, dts_life = [], [], [], []
    by_class = {k: [] for k in ("sex", "lifestyle", "noevent", "clinical")}
    n_pos = n_scored = 0
    torch.manual_seed(seed)
    for _ in range(batches):
        ix = torch.randint(len(p2i), (bs,))
        X, A, Y, B = get_batch(ix, data, p2i, block_size=block_size, device="cpu",
                               padding=padding, lifestyle_augmentations=lifestyle,
                               select="left", no_event_token_rate=rate, cut_batch=True)
        scored = (Y != -1)
        for k in IGNORE:
            scored = scored & (Y != k)
        n_pos += int((X != 0).sum()); n_scored += int(scored.sum())

        d_gathered = target_dt(X, A, B, ties)               # what the loss uses
        d_own = torch.clamp(B - A, min=1.0)                 # what the position's own dt is
        redirected = (d_gathered != d_own) & scored         # the gather moved it
        dts.append((d_gathered[scored] / DAY).numpy())
        dts_redir.append((d_gathered[redirected] / DAY).numpy())
        dts_own.append((d_gathered[scored & ~redirected] / DAY).numpy())
        life = scored & (X >= LIFESTYLE[0]) & (X <= LIFESTYLE[1])
        dts_life.append((d_gathered[life] / DAY).numpy())
        # WHERE the supervised dt comes from, by the kind of token sitting at the position.
        # The smoke run showed p95 = 68 YEARS with the insertion off: the age-0 static tokens
        # (sex, and the lifestyle ones) are followed by the first real visit at age ~70, so
        # their supervised gap is a whole lifetime. Splitting by input class says how much of
        # the inflation is that, rather than anything clinical.
        for k, m in (("sex", (X >= 2) & (X <= 3)),
                     ("lifestyle", (X >= LIFESTYLE[0]) & (X <= LIFESTYLE[1])),
                     ("noevent", X == 1),
                     ("clinical", X >= 22)):
            by_class[k].append((d_gathered[scored & m] / DAY).numpy())
    cat = lambda L: np.concatenate(L) if L else np.zeros(0)
    return dict(dt=cat(dts), dt_redirected=cat(dts_redir), dt_own=cat(dts_own),
                dt_from_lifestyle=cat(dts_life),
                by_class={k: cat(v) for k, v in by_class.items()},
                frac_scored=n_scored / max(n_pos, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(PKG, "data"))
    ap.add_argument("--dataset", default="nacc-dedup-s42")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    d = np.memmap(os.path.join(a.data_root, a.dataset, f"{a.split}.bin"),
                  dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = filter_cohort(d, get_p2i(d))
    gaps = true_visit_gaps(d, p2i)
    RM, RA = float(np.median(gaps)), float(gaps.mean())
    print(f"{a.split} split: {len(d):,} events, cohort {len(p2i):,} patients")
    print(f"\nREFERENCE -- true real-visit-to-real-visit gaps, straight off the .bin:")
    print(f"  median {RM:.3f} y   mean {RA:.3f} y   cv {gaps.std()/gaps.mean():.3f}   "
          f"n={len(gaps):,}")
    print(f"  (every cell below is compared against these two numbers)")

    print("\n" + "=" * 118)
    print("THE 2x2x2 -- marginal distribution of the dt loss_dt is trained on, in YEARS")
    print("=" * 118)
    print(HDR)
    rows = []
    for life in (False, True):
        for ties in (False, True):
            for rate in (0, 5):
                c = cell(d, p2i, a.batches, a.batch_size, a.seed, a.block_size, life, ties, rate)
                lab = (f"life={'on ' if life else 'off'} ties={'on ' if ties else 'off'} "
                       f"rate={rate}")
                print(row(c["dt"], lab, RM, RA))
                rows.append(dict(lifestyle=life, mask_ties=ties, no_event_rate=rate,
                                 n=len(c["dt"]),
                                 med=float(np.median(c["dt"])) if len(c["dt"]) else np.nan,
                                 mean=float(c["dt"].mean()) if len(c["dt"]) else np.nan,
                                 frac_scored=c["frac_scored"],
                                 n_redirected=len(c["dt_redirected"]),
                                 med_redirected=float(np.median(c["dt_redirected"]))
                                 if len(c["dt_redirected"]) else np.nan,
                                 n_from_lifestyle=len(c["dt_from_lifestyle"]),
                                 med_from_lifestyle=float(np.median(c["dt_from_lifestyle"]))
                                 if len(c["dt_from_lifestyle"]) else np.nan,
                                 true_med=RM, true_mean=RA))
            print()

    # ---------------------------------------------------------------- B, in detail
    print("=" * 118)
    print("THE GATHER, DECOMPOSED -- delivered setting (life=on, ties=on, rate=5)")
    print("=" * 118)
    c = cell(d, p2i, a.batches, a.batch_size, a.seed, a.block_size, True, True, 5)
    n = max(len(c["dt"]), 1)
    print(HDR)
    print(row(c["dt"], "all scored positions", RM, RA))
    print(row(c["dt_own"], "kept its OWN dt (forward gap)", RM, RA))
    print(row(c["dt_redirected"], "REDIRECTED by the gather", RM, RA))
    print(f"\n  -> {100*len(c['dt_redirected'])/n:.1f}% of scored positions are trained on some "
          f"OTHER position's dt.")
    print(f"     For a tied token that is the gap BEFORE its own visit -- backward-looking, and "
          f"already\n     computable from its own input (it can see both ages).")

    # ---------------------------------------------------------------- A, in detail
    print("\n" + "=" * 118)
    print("THE LIFESTYLE SHIFT, DECOMPOSED -- positions whose INPUT is a shifted token "
          "(model 4-12)")
    print("=" * 118)
    print(HDR)
    for life in (False, True):
        cc = cell(d, p2i, a.batches, a.batch_size, a.seed, a.block_size, life, True, 5)
        print(row(cc["dt_from_lifestyle"], f"input is lifestyle, life={'on' if life else 'off'}",
                  RM, RA))
        print(row(cc["dt"], f"  all scored,          life={'on' if life else 'off'}", RM, RA))

    print("\n" + "=" * 118)
    print("WHERE THE SUPERVISED dt COMES FROM -- by the kind of token AT the position "
          "(life=on, ties=on)")
    print("=" * 118)
    for rate in (0, 5):
        cc = cell(d, p2i, a.batches, a.batch_size, a.seed, a.block_size, True, True, rate)
        tot = max(len(cc["dt"]), 1)
        print(f"\n  no_event_token_rate = {rate}"
              f"{'   <- the delivered setting' if rate == 5 else '   <- insertion OFF'}")
        print(HDR)
        print(row(cc["dt"], "ALL scored positions", RM, RA))
        for k in ("clinical", "noevent", "sex", "lifestyle"):
            v = cc["by_class"][k]
            print(row(v, f"  input is {k:<10s} ({100*len(v)/tot:4.1f}% of scored)", RM, RA))

    # ---------------------------------------------------------------- THE FIX
    print("\n" + "=" * 118)
    print("THE FIX -- dt_target='next_visit' / 'next_event' vs the delivered 'gather'")
    print("=" * 118)
    print("  Acceptance test: a correct forward target should reproduce the TRUE visit-gap")
    print(f"  distribution (median {RM:.3f} y, mean {RA:.3f} y) -- i.e. 1.00x in both columns --")
    print("  and should be INSENSITIVE to lifestyle_augmentations and no_event_token_rate,")
    print("  because it no longer depends on where the attention mask's argmax happens to land.")
    fix_rows = []
    for life in (True, False):
        for rate in (5, 0):
            print(f"\n  life={'on ' if life else 'off'} rate={rate}")
            print(HDR)
            torch.manual_seed(a.seed)
            acc = {"gather": [], "next_visit": [], "next_event": []}
            for _ in range(a.batches):
                ix = torch.randint(len(p2i), (a.batch_size,))
                X, A, Y, B = get_batch(ix, d, p2i, block_size=a.block_size, device="cpu",
                                       padding="random", lifestyle_augmentations=life,
                                       select="left", no_event_token_rate=rate, cut_batch=True)
                scored = (Y != -1)
                for k in IGNORE:
                    scored = scored & (Y != k)
                acc["gather"].append((target_dt(X, A, B, True)[scored] / DAY).numpy())
                for kind, ig in (("next_visit", None), ("next_event", IGNORE)):
                    dtv, ok = next_visit_dt(X, A, ignore=ig)
                    acc[kind].append((dtv[scored & ok] / DAY).numpy())
            for kind in ("gather", "next_visit", "next_event"):
                v = np.concatenate(acc[kind])
                print(row(v, f"  {kind}", RM, RA))
                fix_rows.append(dict(lifestyle=life, no_event_rate=rate, dt_target=kind,
                                     n=len(v), med=float(np.median(v)), mean=float(v.mean()),
                                     med_over_true=float(np.median(v) / RM),
                                     mean_over_true=float(v.mean() / RA),
                                     true_med=RM, true_mean=RA))

    if a.out:
        import pandas as pd
        pd.DataFrame(fix_rows).to_csv(a.out.replace(".csv", "_fix.csv"), index=False)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"\n[dt_ablation] -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
