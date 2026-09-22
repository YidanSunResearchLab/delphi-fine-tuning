"""
diagnostics.py -- the measurements the ported Figure 2 has to be read against.

Every number quoted in README.md's "What the port measured about itself" section is produced
here. Run it after any change to radc_delphi/engine.py: three of the four checks below are
properties of the ROLLOUT, not of the model, and a change to the sampler moves them.

    python diagnostics.py                 # ~3 min on 4 threads
    python diagnostics.py --n 20          # quicker, noisier

The four checks, and what a bad answer would mean:

  1. WAITING TIMES. Observed vs simulated Delta-t. This tokenization puts every token of one
     visit on the same day (84.7% of observed Delta-t are exactly 0). An autoregressive rollout
     samples Delta-t from a continuous exponential and CANNOT emit an exact tie, so it spreads
     one visit across months. This is the port's dominant finding -- see README.
  2. TOKEN RATE. Clinical tokens per follow-up year, observed vs simulated. Check 1 predicts a
     deficit here; this measures it.
  3. COMPOSITION vs RATE. P(next token is an MMSE bin) from the model at real prefixes,
     against what actually came next -- reported BOTH with and without the No-event marker in
     the denominator, because only the second is a fair comparison and the difference reverses
     the conclusion. The observed stream contains no markers (they are injected by get_batch at
     training time only), so a denominator that includes them is not the same quantity.

     MEASURED FAIRLY, THE COMPOSITION IS FINE: model 0.073 against an observed 0.059, i.e. it
     slightly OVER-calls MMSE. An earlier version of this check reported "the model under-calls
     MMSE by 3x" off the unfair denominator and that was wrong. Essentially all of panel a2's
     under-prediction is the RATE (check 3b), not the token choice.

  3b. THE TIME MODEL. Engine.rate_profile: the sampler's mean waiting time per token, against
     the observed token-to-token and visit-to-visit gaps. The model lands on the VISIT gap
     (~335 d vs an observed median 365 d) because mask_ties trains dt from the last untied
     token -- but generation emits one token per wait, and a real visit carries 4.2 clinical
     tokens. This is the root cause of the systematic under-prediction.
  4. BLOCKED MASS. The share of next-token rate the post-mortem block removes at baseline
     prefixes -- the cost of vocab.Resolved.SAMPLE_BLOCKED, which the docstring there promises
     to measure rather than assume.

  5. THE MMSE TRANSITION MATRIX. Empirical P(next MMSE bin | current bin) from train.bin,
     against the model's own one-step prediction at real val prefixes, same denominator. This
     is the check that located panel a2's ~110x under-prediction of recovery, and it put it in
     the MODEL rather than in the rollout -- see README section 3.1. Two things to read off it:
     the diagonal (structurally impossible under run-length dedup; the model still spends
     16-23% there, which engine.simulate now masks) and the "-> normal" column (13.0% observed
     against 1.1% predicted, which nothing in the sampler can fix).

  6. STAGE-CONDITIONAL CHANGE RATE. P(the MMSE bin changes at all within h years), observed vs
     model, split by baseline stage. This is the sharpest form of the defect and the one to
     quote: the observed rate varies 7x across baseline stages (a Borderline subject sits in a
     3-point window on a noisy measurement and leaves it with probability 1.000 within 5 y),
     while the model's rate is nearly CONSTANT at 0.02-0.03 per year regardless of stage. It
     has not learned that Borderline is unstable, so it under-emits worst exactly where the
     events are.

     It also answers an accounting objection that has to be ruled out before any of this can be
     believed: is the observed side counting tokens the model could never emit -- i.e. is it
     read off a NON-deduplicated stream? No. `verify_dedup()` below checks that val.bin has
     zero consecutive same-bin pairs, recomputes the observed rate by hand off the dedup'd
     stream and reproduces the cached frame exactly, and counts MMSE emissions on both sides
     over the same window with the same rule.

Check 3 is the one that decides whether a disappointing panel is the model's fault or ours.
Check 5 is the one that decides WHICH defect a disappointing panel is showing.
Check 6 is the one to put in a slide.
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from radc_delphi import engine as EN  # noqa: E402
from figure2 import radc_states as S  # noqa: E402

D = EN.DAYS_PER_YEAR
CKPT = "../delphi/Delphi-ROSMAP/ckpt.pt"
DATA = "data/rosmap"


def _profile(x, label):
    x = np.asarray(x, float)
    return (f"  {label:11s} n={len(x):7d}   ==0 d {np.mean(x < 1):6.1%}   <30 d "
            f"{np.mean(x < 30):6.1%}   median {np.median(x):7.1f} d   mean {x.mean():7.1f} d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="subjects sampled for the rollout checks")
    ap.add_argument("--n-mc", type=int, default=50)
    ap.add_argument("--split", default="val")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    eng = EN.load(CKPT, data_dir=DATA, device="cpu")
    S.configure(eng.labels)
    data, p2i, _ = eng.load_split(a.split)
    MM = list(eng.res.SCALES["MMSE"])
    NORM = eng.res.ID["MMSE_normal"]
    skip = list(eng.ignore_tokens) + [eng.res.NO_EVENT]

    rng = np.random.default_rng(0)
    ks = rng.choice(len(p2i), min(a.n, len(p2i)), replace=False)

    # ---------------------------------------------------------------- 1 + 2 + blocked mass
    dt_obs = [np.diff(eng.stream(data, p2i, k)[0]) for k in range(len(p2i))]
    dt_obs = np.concatenate(dt_obs)

    dt_sim, obs_rate, sim_rate, bm = [], [], [], []
    rec_obs, rec_sim = [], []
    for k in ks:
        ages, toks, _ = eng.stream(data, p2i, k)
        content = ~np.isin(toks, skip)
        va = np.unique(ages[(ages > 0) & content])
        if len(va) < 2:
            continue
        base = va[0]
        m = ages <= base
        fu = (ages.max() - base) / D
        if fu <= 0:
            continue
        bm.append(eng.blocked_mass(toks[m], ages[m]))
        obs_rate.append(((ages > base) & content).sum() / fu)
        T, A = eng.simulate(toks[m], ages[m], n_mc=a.n_mc, seed=int(k),
                            until_age_years=max(105.0, base / D + 16))
        real = A > EN.PAD_AGE + 1
        npre = int(m.sum())
        new = real.copy()
        new[:, :npre] = False
        win = new & (A <= base + fu * D) & ~np.isin(T, skip)
        sim_rate.append(win.sum(1).mean() / fu)
        for r in range(A.shape[0]):
            aa = A[r][real[r]]
            if len(aa) > npre:
                dt_sim.append(np.diff(aa[npre - 1:]))
        # recovery to Normal, among subjects not already Normal at baseline
        bmask = np.isin(toks, MM) & (ages <= base)
        if bmask.any() and int(S.TOK2SLOT[int(toks[bmask][np.argmax(ages[bmask])])]) > 0:
            hit = ((T == NORM) & (A > base) & (A <= base + 5 * D)).any(1)
            died = ((T == eng.res.DEATH) & (A > base) & (A <= base + 5 * D)).any(1)
            rec_sim.append((hit & ~died).mean())
            rec_obs.append(float((np.isin(toks, [NORM]) & (ages > base)
                                  & (ages <= base + 5 * D)).any()))
    dt_sim = np.concatenate(dt_sim)

    print("\n1. WAITING TIMES (Delta-t between consecutive tokens)")
    print(_profile(dt_obs, "observed"))
    print(_profile(dt_sim, "simulated"))
    print("   -> the annual grid puts a whole visit on one day; a continuous-time sampler "
          "cannot emit a tie,\n      so one observed visit becomes several months of "
          "simulated time. See README.")

    print("\n2. CLINICAL TOKEN RATE (tokens per follow-up year, statics and No-event excluded)")
    o, s = float(np.mean(obs_rate)), float(np.mean(sim_rate))
    print(f"   observed {o:.2f}   simulated {s:.2f}   ratio {s / o:.2f}")
    if rec_obs:
        print(f"   5-y recovery to MMSE_normal (not Normal at baseline, n={len(rec_obs)}): "
              f"observed {np.mean(rec_obs):.3f}   predicted {np.mean(rec_sim):.3f}")

    # ---------------------------------------------------------------- 3
    pm, pmf, pe, prof = [], [], [], []
    for k in ks[:min(30, len(ks))]:
        ages, toks, _ = eng.stream(data, p2i, k)
        for j in range(5, len(toks) - 1):
            p = eng.next_token_probs(toks[:j + 1], ages[:j + 1])
            pm.append(p[MM].sum())
            # fair denominator: the observed stream contains no No-event markers
            pmf.append(p[MM].sum() / max(1.0 - p[eng.res.NO_EVENT], 1e-12))
            pe.append(float(toks[j + 1] in MM))
            prof.append(eng.rate_profile(toks[:j + 1], ages[:j + 1]))
    mm, me = float(np.mean(pm)), float(np.mean(pe))
    mmf = float(np.mean(pmf))
    print("\n3. COMPOSITION -- P(next token is an MMSE bin)")
    print(f"   model, No-event IN  the denominator {mm:.4f}   <- NOT comparable to observed")
    print(f"   model, No-event OUT of it          {mmf:.4f}   <- the fair one")
    print(f"   observed (the stream has no markers) {me:.4f}   ratio {mmf / me:.2f}")
    print("   -> composition is fine; the model slightly OVER-calls MMSE. So the deficit in "
          "check 2\n      is the RATE, not the token choice. See 3b.")

    # ---------------------------------------------------------------- 3b
    w = np.array([x["mean_wait_days"] for x in prof])
    ne = np.array([x["noevent_share"] for x in prof])
    cy = np.array([x["clinical_per_yr"] for x in prof])
    vg = []
    for k in ks:
        ages, _t, _ = eng.stream(data, p2i, k)
        vg.append(np.diff(np.unique(ages[ages > 0])))
    vg = np.concatenate(vg)
    print("\n3b. THE TIME MODEL -- how long the sampler waits per emitted token")
    print(f"   model implied mean wait  median {np.median(w):7.1f} d")
    print(f"   observed token-to-token  median {np.median(dt_obs):7.1f} d   "
          f"mean {dt_obs.mean():.1f} d")
    print(f"   observed VISIT-to-VISIT  median {np.median(vg):7.1f} d   mean {vg.mean():.1f} d")
    print(f"   -> the model learned the VISIT gap, but emits ONE token per wait.")
    print(f"   No-event share of the rate  median {np.median(ne):.3f}")
    print(f"   => clinical tokens/year: model {np.mean(cy):.2f}   observed {o:.2f}   "
          f"ratio {np.mean(cy) / o:.2f}")

    # ---------------------------------------------------------------- 4
    print("\n4. POST-MORTEM BLOCK (share of next-token rate removed at baseline prefixes)")
    print(f"   median {np.median(bm):.4f}   mean {np.mean(bm):.4f}   max {np.max(bm):.4f}")

    # ---------------------------------------------------------------- 5
    transition_matrix(eng, a.split)

    # ---------------------------------------------------------------- 6
    verify_dedup(eng, a.split)
    change_rate_by_stage(eng, a.split, n_mc=a.n_mc)


def transition_matrix(eng, split):
    """Empirical vs model-predicted P(next MMSE bin | current bin), on the same denominator.

    The empirical side is read off train.bin -- what the model was fitted on -- and the model
    side is scored at val prefixes, only at positions where the next real token IS an MMSE
    token, so the two are the same conditional quantity and are directly comparable.
    """
    from radc_delphi import batching as Bt, vocab as V
    MM = list(eng.res.SCALES["MMSE"])
    nm = [eng.res.NAMES[t].split("_")[1] for t in MM]

    tr = np.fromfile("data/rosmap/train.bin", dtype=np.uint32).reshape(-1, 3)
    tp2i = Bt.get_p2i(tr)
    E = np.zeros((3, 3))
    first = np.zeros(3)
    for k in range(len(tp2i)):
        s0, n0 = int(tp2i[k, 0]), int(tp2i[k, 1])
        _ag, tk = Bt.patient_stream(tr, s0, n0)
        seq = [MM.index(int(t)) for t in tk if int(t) in MM]
        if seq:
            first[seq[0]] += 1
        for i in range(1, len(seq)):
            E[seq[i - 1], seq[i]] += 1

    data, p2i, _ = eng.load_split(split)
    P = np.zeros((3, 3))
    cnt = np.zeros(3)
    for k in range(len(p2i)):
        ages, toks, _ = eng.stream(data, p2i, k)
        cur = -1
        for j in range(len(toks) - 1):
            t = int(toks[j])
            if t in MM:
                cur = MM.index(t)
            if cur < 0 or int(toks[j + 1]) not in MM:
                continue
            p = eng.next_token_probs(toks[:j + 1], ages[:j + 1])
            P[cur] += p[MM] / p[MM].sum()
            cnt[cur] += 1

    print("\n5. MMSE TRANSITION MATRIX -- P(next MMSE bin | current bin)")
    print(f"   {'current':13s}" + "".join(f"{x:>12s}" for x in nm) + "      n")
    for i in range(3):
        e = E[i] / max(E[i].sum(), 1)
        print(f"   {nm[i]:9s} obs  " + "".join(f"{e[j]:12.3f}" for j in range(3))
              + f"  {int(E[i].sum()):6d}")
        if cnt[i]:
            print(f"   {'':9s} pred " + "".join(f"{P[i][j] / cnt[i]:12.3f}" for j in range(3))
                  + f"  {int(cnt[i]):6d}")
    w = cnt / max(cnt.sum(), 1)
    obs_n = float((E[:, 2] / np.maximum(E.sum(1), 1) * w).sum())
    pred_n = float((P[:, 2] / np.maximum(cnt, 1) * w).sum())
    print(f"\n   P(next MMSE == normal):  observed {obs_n:.3f}   model {pred_n:.3f}   "
          f"-> {pred_n / obs_n:.2f}x")
    tot = E.sum(0) + first
    print(f"   and why: {int(first[2])} of the {int(tot[2])} MMSE_normal tokens in train.bin "
          f"({first[2] / tot[2]:.0%}) are a subject's\n      FIRST MMSE observation. Genuine "
          f"recoveries number {int(E[:, 2].sum())} in the whole corpus.")




def verify_dedup(eng, split):
    """Rule out the accounting objection: is `observed` read off a non-deduplicated stream?

    Three checks, because "the observed bar is 100x the predicted bar" is exactly the shape a
    denominator bug makes, and it has to be excluded before the defect can be blamed on the
    model.
    """
    import pandas as pd
    import glob
    from figure2 import figure2_core as F2
    MM = list(eng.res.SCALES["MMSE"])
    NORM = eng.res.ID["MMSE_normal"]
    data, p2i, _ = eng.load_split(split)

    rep = same_age = 0
    for k in range(len(p2i)):
        ages, toks, _ = eng.stream(data, p2i, k)
        seq = [(float(a), int(t)) for a, t in zip(ages, toks) if int(t) in MM]
        for i in range(1, len(seq)):
            rep += seq[i][1] == seq[i - 1][1]
            same_age += seq[i][0] == seq[i - 1][0]
    print(f"\n6a. DEDUP IN {split}.bin: {rep} consecutive same-bin pairs, "
          f"{same_age} same-age MMSE pairs (both must be 0)")

    cached = glob.glob(f"results/figure2/_cache/frame_*_{split}_n*.pkl")
    if not cached:
        print("    (no cache -- skipping the frame cross-check; run run_figure2.sh first)")
        return
    df = pd.read_pickle(sorted(cached)[0])
    at = df[df.baseline_state != 0]
    manual, frame, nobs = [], [], []
    for _, row in at.iterrows():
        k = int(row.pid)
        ages, toks, _ = eng.stream(data, p2i, k)
        base = F2._baseline(ages, toks)[0]
        w = (ages > base) & (ages <= base + 5 * D)
        manual.append(float(np.isin(toks, [NORM])[w].any()))
        frame.append(float(row["obs_t_Normal"] <= 5))
        nobs.append(int(np.isin(toks, MM)[w].sum()))
    print(f"    hand-recomputed off the dedup'd stream: {np.mean(manual):.3f}   "
          f"cached frame: {np.mean(frame):.3f}   identical: {np.allclose(manual, frame)}")
    print(f"    MMSE tokens emitted in the same 5-y window, same rule, n={len(at)}: "
          f"observed {np.mean(nobs):.3f}/subject")


def change_rate_by_stage(eng, split, n_mc=50, horizons=(1, 2, 5)):
    """P(the MMSE bin changes at all within h years), observed vs model, by baseline stage.

    Subjects whose follow-up ends before h are DROPPED, not counted as "no change" -- scoring
    them as non-events is the same censoring mistake labels_at_h exists to avoid, and it would
    flatter the model by depressing the observed rate.
    """
    import pandas as pd
    import glob
    from figure2 import radc_states as S2, figure2_core as F2
    MM = list(eng.res.SCALES["MMSE"])
    data, p2i, _ = eng.load_split(split)
    cached = glob.glob(f"results/figure2/_cache/frame_*_{split}_n*.pkl")
    if not cached:
        return
    df = pd.read_pickle(sorted(cached)[0])
    print("\n6b. P(MMSE bin changes within h y), by baseline stage")
    print(f"   {'baseline':12s}{'n':>5s}" + "".join(f"{f'{h}y obs':>9s}{f'{h}y pred':>10s}"
                                                    for h in horizons))
    for st in range(S2.NSTAGE):
        sub = df[df.baseline_state == st]
        if len(sub) < 10:
            continue
        O = {h: [] for h in horizons}
        P = {h: [] for h in horizons}
        for _, row in sub.iterrows():
            k = int(row.pid)
            ages, toks, _ = eng.stream(data, p2i, k)
            base = F2._baseline(ages, toks)[0]
            m = ages <= base
            T, A = eng.simulate(toks[m], ages[m], n_mc=n_mc, seed=k,
                                until_age_years=max(105.0, base / D + 16))
            for h in horizons:
                if row.followup + 1e-6 < h:
                    continue
                O[h].append(float((np.isin(toks, MM) & (ages > base)
                                   & (ages <= base + h * D)).any()))
                P[h].append(float((np.isin(T, MM) & (A > base) & (A <= base + h * D)
                                   & (A > EN.PAD_AGE + 1)).any(1).mean()))
        line = f"   {S2.STAGE_NAMES[st]:12s}{len(sub):5d}"
        for h in horizons:
            line += f"{np.mean(O[h]):9.3f}{np.mean(P[h]):10.3f}"
        print(line)
    print("   -> the observed rate swings 7x across stages; the model's barely moves. It has "
          "not learned\n      that Borderline (a 3-point window on a noisy scale) is unstable.")


if __name__ == "__main__":
    main()
