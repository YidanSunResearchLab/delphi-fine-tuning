"""
gen_steps_probe.py -- does generate() spend the right NUMBER of steps, or does it serialise
one visit into several?

    python experiments/time_head/gen_steps_probe.py --runs <dir> --tags L8E120H6_s42

THE HYPOTHESIS UNDER TEST
-------------------------
The Figure-2 timing metrics show the model predicting transitions ~2.6 y too late, with a
fitted floor of ~5.3 y (it essentially never predicts an event inside 5 years). Fixing the
training target did not move that (DT arms: -0.015 +/- 0.367 y at the recommended shape), and
the per-step clock is roughly calibrated (time_head_probe: predicted E[dt] median 1.63 y vs a
true visit gap of 1.11 y). So the remaining explanation is the STEP COUNT:

    the real data records ~4 tokens at the SAME age per visit (CDRSUM, MOCA, FAQ, NACCUDSD),
    but generate() samples one token per step and adds a strictly positive dt every time:

        t_next = clamp(-exp(-logits) * rand.log(), min=0, ...).min(1)
        age_next = age[..., [-1]] + t_next

    Emitting 4 tokens at one age needs 3 consecutive draws of dt ~ 0, which has probability
    ~0. So one real visit becomes ~4 simulated time steps, and reaching a target state costs
    ~4x the time it should.

If that is right, the numbers below show: generated tokens-per-visit ~ 1 against ~4 real, and
generated steps-to-Dementia ~ 4x the real visits-to-Dementia, while the per-step gap is about
right. If the per-step gap is instead way off, the story is wrong and the sampler is not the
place to fix it.

WHAT IS REPORTED (all aggregate, no per-patient anything)
  [1] tokens per distinct age: REAL vs GENERATED
  [2] the gap between consecutive DISTINCT ages: REAL vs GENERATED (the per-step clock)
  [3] the elapsed time and the step count to first reach NACCUDSD=Dementia, real vs generated,
      over the same patients
  [4] the fraction of sampled dt small enough to count as a tie (< 30 d), i.e. how often the
      sampler even tries to co-locate two tokens
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
from delphi.model import Delphi, DelphiConfig                   # noqa: E402
from delphi.utils import get_p2i, patient_stream                # noqa: E402

DAY = 365.25
DEMENTIA, DEATH = 109, 110          # MODEL space
TIE_DAYS = 30.0                     # two tokens this close count as the same visit


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


def visits(ages, toks, tol=TIE_DAYS):
    """Group a single stream into visits: consecutive ages within `tol` days are one visit.
    Returns (visit_ages, tokens_per_visit). Ages of 0 (static background) are excluded."""
    m = ages > 0
    a, t = ages[m], toks[m]
    if a.size == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    o = np.argsort(a, kind="stable")
    a, t = a[o], t[o]
    va, cnt = [a[0]], [1]
    for x in a[1:]:
        if x - va[-1] <= tol:
            cnt[-1] += 1
        else:
            va.append(x); cnt.append(1)
    return np.asarray(va), np.asarray(cnt)


def q(x, label, unit=""):
    if len(x) == 0:
        return f"  {label:38s} (empty)"
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return (f"  {label:38s} n={len(x):>7,}  p5 {p[0]:7.2f}  p25 {p[1]:7.2f}  med {p[2]:7.2f}  "
            f"p75 {p[3]:7.2f}  p95 {p[4]:7.2f}  mean {x.mean():7.2f} {unit}")


def load(cp):
    c = torch.load(cp, map_location="cpu", weights_only=False)
    args = {k: v for k, v in c["model_args"].items() if k != "_ckpt_sig"}
    valid = set(DelphiConfig.__dataclass_fields__)
    m = Delphi(DelphiConfig(**{k: v for k, v in args.items() if k in valid}))
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in c["model"].items()})
    m.eval()
    return m, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--runs", default=None)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--data-root", default=os.path.join(PKG, "data"))
    ap.add_argument("--dataset", default="nacc-dedup-s42")
    ap.add_argument("--split", default="test")
    ap.add_argument("--patients", type=int, default=250)
    ap.add_argument("--samples", type=int, default=16, help="MC trajectories per patient")
    ap.add_argument("--prefix-visits", type=int, default=2, help="real visits used as the seed")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--visit-batched", action="store_true",
                    help="sample a whole VISIT per step: draw k from the empirical "
                         "tokens-per-visit distribution of NON-FIRST real visits and emit k "
                         "tokens at one age. The fix under test.")
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
        sys.exit("[gen_steps] no checkpoints (--ckpt or --runs)")

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    d = np.memmap(os.path.join(a.data_root, a.dataset, f"{a.split}.bin"),
                  dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = filter_cohort(d, get_p2i(d))
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(p2i), size=min(a.patients, len(p2i)), replace=False)
    print(f"{a.split} split: cohort {len(p2i):,} patients, probing {len(pick)}")

    # ------------------------------------------------------------------ REAL
    real_tpv, real_gaps = [], []
    real_reach = []                                   # (years, n_visits) to Dementia
    streams = []
    for k in pick:
        ages, toks = patient_stream(d, int(p2i[k, 0]), int(p2i[k, 1]))
        streams.append((ages, toks))
        va, cnt = visits(ages, toks)
        real_tpv.append(cnt)
        if va.size >= 2:
            real_gaps.append(np.diff(va) / DAY)
        hit = np.where((toks == DEMENTIA) & (ages > 0))[0]
        if hit.size and va.size:
            t0 = va[min(a.prefix_visits, va.size) - 1]
            th = ages[hit[0]]
            if th > t0:
                nv = int(((va > t0) & (va <= th)).sum())
                real_reach.append(((th - t0) / DAY, nv))
    # k for the fix is drawn from NON-FIRST visits: the first visit of a stream carries every
    # baseline scale at once (that is the p95 = 16 tail), while generation always continues
    # from a seed prefix, so the visits it invents are later ones.
    later_tpv = np.concatenate([c[1:] for c in real_tpv if len(c) > 1]) if real_tpv else np.zeros(0)
    real_tpv = np.concatenate(real_tpv) if real_tpv else np.zeros(0)
    real_gaps = np.concatenate(real_gaps) if real_gaps else np.zeros(0)
    real_reach = np.asarray(real_reach) if real_reach else np.zeros((0, 2))

    rows = []
    for cp in ckpts:
        name = os.path.basename(os.path.dirname(cp)) or os.path.basename(cp)
        m, args = load(cp)
        print("\n" + "=" * 112)
        print(f"{name}   {args['n_layer']}L/{args['n_head']}H/{args['n_embd']}d   "
              f"time_head={args.get('time_head', False)}  "
              f"dt_target={args.get('dt_target', 'gather')}  "
              f"visit_batched={a.visit_batched}"
              + (f"  (k ~ non-first real visits: median "
                 f"{np.median(later_tpv):.0f}, mean {later_tpv.mean():.2f})"
                 if a.visit_batched and len(later_tpv) else ""))
        print("=" * 112)

        gen_tpv, gen_gaps, gen_reach, all_dt = [], [], [], []
        torch.manual_seed(a.seed)
        for ages, toks in streams:
            va, _ = visits(ages, toks)
            if va.size < a.prefix_visits:
                continue
            cut = va[a.prefix_visits - 1] + TIE_DAYS
            keep = (ages <= cut)
            if keep.sum() < 1:
                continue
            pi = torch.tensor(toks[keep][None].repeat(a.samples, 0), dtype=torch.long)
            pa = torch.tensor(ages[keep][None].repeat(a.samples, 0), dtype=torch.float32)
            n0 = pi.size(1)
            with torch.no_grad():
                gi, ga, _ = m.generate(pi, pa, max_new_tokens=a.max_new, max_age=100 * DAY,
                                       termination_tokens=[DEATH],
                                       visit_sizes=later_tpv if a.visit_batched else None)
            t0 = float(pa[0, -1])
            for r in range(gi.size(0)):
                nt, na_ = gi[r, n0:].numpy(), ga[r, n0:].numpy()
                live = (nt > 0) & (na_ > 0)
                nt, na_ = nt[live], na_[live]
                if nt.size == 0:
                    continue
                all_dt.append(np.diff(np.concatenate([[t0], na_])))
                _, cnt = visits(na_, nt)
                gen_tpv.append(cnt)
                vv, _ = visits(na_, nt)
                if vv.size >= 2:
                    gen_gaps.append(np.diff(vv) / DAY)
                h = np.where(nt == DEMENTIA)[0]
                if h.size:
                    th = na_[h[0]]
                    nv = int((vv <= th).sum())
                    gen_reach.append(((th - t0) / DAY, h[0] + 1, nv))
        gen_tpv = np.concatenate(gen_tpv) if gen_tpv else np.zeros(0)
        gen_gaps = np.concatenate(gen_gaps) if gen_gaps else np.zeros(0)
        all_dt = np.concatenate(all_dt) if all_dt else np.zeros(0)
        gen_reach = np.asarray(gen_reach) if gen_reach else np.zeros((0, 3))

        print("\n[1] tokens per visit  (a visit = ages within 30 d of each other)")
        print(q(real_tpv, "REAL", "tokens"))
        print(q(gen_tpv, "GENERATED", "tokens"))
        print(f"\n[2] gap between consecutive DISTINCT visits -- the per-step clock")
        print(q(real_gaps, "REAL", "years"))
        print(q(gen_gaps, "GENERATED", "years"))
        print(f"\n[4] sampled dt per generate() step")
        print(q(all_dt / DAY, "every sampled step", "years"))
        print(f"  {'':38s} fraction of steps with dt < {TIE_DAYS:.0f} d (a tie): "
              f"{100 * (all_dt < TIE_DAYS).mean():.3f}%")

        print(f"\n[3] reaching NACCUDSD=Dementia from visit {a.prefix_visits}")
        if len(real_reach):
            print(q(real_reach[:, 0], "REAL   elapsed", "years"))
            print(q(real_reach[:, 1].astype(float), "REAL   visits crossed", "visits"))
        if len(gen_reach):
            print(q(gen_reach[:, 0], "GEN    elapsed", "years"))
            print(q(gen_reach[:, 1].astype(float), "GEN    tokens emitted (= steps)", "steps"))
            print(q(gen_reach[:, 2].astype(float), "GEN    visits crossed", "visits"))
            if len(real_reach):
                rt, gt = np.median(real_reach[:, 0]), np.median(gen_reach[:, 0])
                rv, gs = np.median(real_reach[:, 1]), np.median(gen_reach[:, 1])
                print(f"\n  -> elapsed  GEN / REAL = {gt / max(rt, 1e-9):.2f}x   "
                      f"({gt:.2f} y vs {rt:.2f} y, medians)")
                print(f"     steps    GEN / REAL visits = {gs / max(rv, 1e-9):.2f}x   "
                      f"({gs:.0f} steps vs {rv:.0f} visits)")
                print(f"     tokens/visit  REAL {np.median(real_tpv):.1f} vs "
                      f"GEN {np.median(gen_tpv):.1f}")
        rows.append(dict(run=name, time_head=args.get("time_head", False),
                         dt_target=args.get("dt_target", "gather"),
                         real_tpv_med=float(np.median(real_tpv)) if len(real_tpv) else np.nan,
                         gen_tpv_med=float(np.median(gen_tpv)) if len(gen_tpv) else np.nan,
                         real_gap_med=float(np.median(real_gaps)) if len(real_gaps) else np.nan,
                         gen_gap_med=float(np.median(gen_gaps)) if len(gen_gaps) else np.nan,
                         frac_tie=float((all_dt < TIE_DAYS).mean()) if len(all_dt) else np.nan,
                         real_reach_y=float(np.median(real_reach[:, 0])) if len(real_reach) else np.nan,
                         gen_reach_y=float(np.median(gen_reach[:, 0])) if len(gen_reach) else np.nan,
                         real_reach_visits=float(np.median(real_reach[:, 1])) if len(real_reach) else np.nan,
                         gen_reach_steps=float(np.median(gen_reach[:, 1])) if len(gen_reach) else np.nan))

    if a.out:
        import pandas as pd
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"\n[gen_steps] -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
