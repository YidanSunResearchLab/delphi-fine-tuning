"""
tie_leak_probe.py -- what does a token learn by seeing its OWN visit's other tokens?

    python experiments/time_head/tie_leak_probe.py --data-root <dir> [--ckpt <path>]

WHY THIS DECIDES SOMETHING
--------------------------
generate() emits ~1 token per visit where the real data has ~4 (measured: 1.03 vs 4.11), so
one real visit becomes ~4 simulated time steps and the trajectory runs 2.5x too slow. The
cheapest fix is to let the model learn the burst itself -- turn OFF mask_ties, drop the
gather, lower t_min -- because then "the next token is later today" becomes a thing the model
can represent, and no external visit-size distribution is needed.

The objection is leakage: with mask_ties off, a token can see its same-visit siblings, so
"CDRSUM is severe at this visit" makes "NACCUDSD is Dementia at this visit" easy. This
measures how big that is, on the DATA, with count-based predictors -- no model needed and no
retraining, so the answer does not depend on any checkpoint.

TWO STRUCTURAL FACTS worth having in front of you first (both read off model.py):

  * mask_ties is applied only `if targets is not None` (model.py:404). GENERATION NEVER
    APPLIES IT. So the delivered setup already trains with the mask and generates without it
    -- turning it off for training REMOVES a train/inference mismatch rather than creating
    one.
  * figure2 scores through predict_adapter._forward_logits(tie_mask=True), which passes dummy
    targets specifically to switch the mask ON. So the evaluation can keep blocking siblings
    no matter how the model is trained; the leakage would show up as a train/eval mismatch
    (honest, visible in the metrics), not as an inflated score.

WHAT IS REPORTED
  [1] how often a scored position even HAS a same-visit sibling before it
  [2] cross-entropy of the next content token under three count-based predictors, fitted on
      train and evaluated on test:
          marginal                      -- knows nothing
          | previous visit's tokens     -- the information mask_ties leaves available
          | same-visit tokens so far    -- what mask_ties removes
      compared against the trained model's own loss_ce (~2.71 nats)
  [3] the specific worry: predict the visit's NACCUDSD value from the OTHER tokens recorded at
      that same visit. Accuracy and cross-entropy vs the marginal.
"""
import argparse
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Delphi-2M")
sys.path.insert(0, PKG)
from delphi.utils import get_p2i, patient_stream                # noqa: E402

DAY = 365.25
TIE_DAYS = 30.0
CONTENT_LO = 22                      # ignore_tokens = 0..21
UDSD = (106, 107, 108, 109)          # MODEL space


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


def to_visits(ages, toks, tol=TIE_DAYS):
    """[(age, [tokens])] -- consecutive ages within tol days are one visit; age 0 excluded."""
    m = ages > 0
    a, t = ages[m], toks[m]
    if a.size == 0:
        return []
    o = np.argsort(a, kind="stable")
    a, t = a[o], t[o]
    out = [(a[0], [int(t[0])])]
    for x, y in zip(a[1:], t[1:]):
        if x - out[-1][0] <= tol:
            out[-1][1].append(int(y))
        else:
            out.append((x, [int(y)]))
    return out


def streams(data, p2i, idxs):
    for k in idxs:
        ages, toks = patient_stream(data, int(p2i[k, 0]), int(p2i[k, 1]))
        v = to_visits(ages, toks)
        if v:
            yield v


class CountPredictor:
    """P(target | key) with add-alpha smoothing, backing off to the marginal."""

    def __init__(self, alpha=0.5):
        self.tab = defaultdict(Counter)
        self.marg = Counter()
        self.alpha = alpha
        self.vocab = set()

    def fit(self, key, target):
        self.tab[key][target] += 1
        self.marg[target] += 1
        self.vocab.add(target)

    def nll(self, key, target):
        V = max(len(self.vocab), 1)
        c = self.tab.get(key)
        if c:
            n = sum(c.values())
            p = (c.get(target, 0) + self.alpha) / (n + self.alpha * V)
        else:
            n = sum(self.marg.values())
            p = (self.marg.get(target, 0) + self.alpha) / (n + self.alpha * V)
        return -math.log(max(p, 1e-12))

    def argmax(self, key):
        c = self.tab.get(key) or self.marg
        return c.most_common(1)[0][0] if c else -1

    def marginal_nll(self, target):
        V = max(len(self.vocab), 1)
        n = sum(self.marg.values())
        return -math.log(max((self.marg.get(target, 0) + self.alpha) / (n + self.alpha * V), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(PKG, "data"))
    ap.add_argument("--dataset", default="nacc-dedup-s42")
    ap.add_argument("--ckpt", default=None, help="optional: also flip mask_ties on a checkpoint")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    def load(split):
        d = np.memmap(os.path.join(a.data_root, a.dataset, f"{split}.bin"),
                      dtype=np.uint32, mode="r").reshape(-1, 3)
        return d, filter_cohort(d, get_p2i(d))

    tr, tr_p = load("train")
    te, te_p = load("test")
    print(f"train cohort {len(tr_p):,} patients | test cohort {len(te_p):,}")

    # ---------------------------------------------------------------- [1] + [2]
    p_next_sib = CountPredictor()      # key = same-visit tokens so far
    p_next_prev = CountPredictor()     # key = previous visit's token set
    n_with_sib = n_tot = 0

    def walk(visits, fit):
        nonlocal n_with_sib, n_tot
        rows = []
        prev = ()
        for _, toks in visits:
            seen = []
            for t in toks:
                if t >= CONTENT_LO:
                    rows.append((tuple(sorted(seen)), prev, t, len(seen) > 0))
                seen.append(t)
            prev = tuple(sorted(toks))
        return rows

    for v in streams(tr, tr_p, range(len(tr_p))):
        for sib, prev, tgt, has in walk(v, True):
            p_next_sib.fit(sib, tgt)
            p_next_prev.fit(prev, tgt)

    nll_sib, nll_prev, nll_marg = [], [], []
    for v in streams(te, te_p, range(len(te_p))):
        for sib, prev, tgt, has in walk(v, False):
            n_tot += 1; n_with_sib += int(has)
            nll_sib.append(p_next_sib.nll(sib, tgt))
            nll_prev.append(p_next_prev.nll(prev, tgt))
            nll_marg.append(p_next_sib.marginal_nll(tgt))

    print("\n" + "=" * 92)
    print("[1] how often does a scored position have a same-visit sibling before it?")
    print("=" * 92)
    print(f"  {100 * n_with_sib / max(n_tot, 1):.1f}% of {n_tot:,} scored content targets "
          f"({n_with_sib:,}) already have >=1 token at their own visit.")
    print(f"  Those are exactly the positions mask_ties blinds.")

    print("\n" + "=" * 92)
    print("[2] cross-entropy of the next content token, nats (count predictors, train->test)")
    print("=" * 92)
    print(f"  {'predictor':<44s}{'CE (nats)':>12s}{'vs marginal':>14s}")
    m = float(np.mean(nll_marg))
    for lab, v in (("marginal (knows nothing)", nll_marg),
                   ("| previous visit's tokens  (mask_ties keeps)", nll_prev),
                   ("| same-visit tokens so far (mask_ties REMOVES)", nll_sib)):
        v = float(np.mean(v))
        print(f"  {lab:<44s}{v:>12.4f}{v - m:>+14.4f}")
    print(f"  {'the trained model, loss_ce':<44s}{2.7098:>12.4f}{2.7098 - m:>+14.4f}"
          f"   (L8E120H6_s42, for scale)")

    # ---------------------------------------------------------------- [3]
    print("\n" + "=" * 92)
    print("[3] THE WORRY: predict this visit's NACCUDSD from the OTHER tokens at the same visit")
    print("=" * 92)
    p_ud = CountPredictor()
    for v in streams(tr, tr_p, range(len(tr_p))):
        for _, toks in v:
            u = [t for t in toks if t in UDSD]
            o = tuple(sorted(t for t in toks if t not in UDSD and t >= CONTENT_LO))
            if u:
                p_ud.fit(o, u[0])
    hit = tot = 0
    ce, ce_m = [], []
    for v in streams(te, te_p, range(len(te_p))):
        for _, toks in v:
            u = [t for t in toks if t in UDSD]
            o = tuple(sorted(t for t in toks if t not in UDSD and t >= CONTENT_LO))
            if not u:
                continue
            tot += 1
            hit += int(p_ud.argmax(o) == u[0])
            ce.append(p_ud.nll(o, u[0])); ce_m.append(p_ud.marginal_nll(u[0]))
    if tot:
        base = Counter()
        for v in streams(te, te_p, range(len(te_p))):
            for _, toks in v:
                for t in toks:
                    if t in UDSD:
                        base[t] += 1
        maj = base.most_common(1)[0][1] / sum(base.values())
        print(f"  n = {tot:,} visits carrying a NACCUDSD token")
        print(f"  accuracy from same-visit siblings : {100 * hit / tot:5.1f}%")
        print(f"  accuracy of always-guess-majority : {100 * maj:5.1f}%")
        print(f"  cross-entropy  siblings {float(np.mean(ce)):.4f}  "
              f"vs marginal {float(np.mean(ce_m)):.4f}  "
              f"-> {float(np.mean(ce_m)) - float(np.mean(ce)):+.4f} nats of leakage")

    # ---------------------------------------------------------------- optional model check
    if a.ckpt:
        import contextlib, io, torch
        from delphi.model import Delphi, DelphiConfig
        from delphi.utils import get_batch
        c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        args = {k: v for k, v in c["model_args"].items() if k != "_ckpt_sig"}
        valid = set(DelphiConfig.__dataclass_fields__)
        with contextlib.redirect_stdout(io.StringIO()):
            mdl = Delphi(DelphiConfig(**{k: v for k, v in args.items() if k in valid}))
        mdl.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in c["model"].items()})
        mdl.eval()
        print("\n" + "=" * 92)
        print("[4] the same checkpoint scored with mask_ties ON vs OFF")
        print("    OUT OF DISTRIBUTION -- this model was TRAINED with the mask, so switching it")
        print("    off at eval shows what it does with inputs it never learned to use, not what")
        print("    a model trained without the mask would achieve. Indicative only.")
        print("=" * 92)
        for flag in (True, False):
            mdl.config.mask_ties = flag
            torch.manual_seed(42)
            ces, dts = [], []
            for _ in range(20):
                ix = torch.randint(len(tr_p), (128,))
                X, A, Y, B = get_batch(ix, tr, tr_p, block_size=args["block_size"], device="cpu",
                                       padding="random", lifestyle_augmentations=True,
                                       select="left", no_event_token_rate=5, cut_batch=True)
                with torch.no_grad():
                    _, l, _ = mdl(X, A, Y, B)
                ces.append(float(l["loss_ce"])); dts.append(float(l["loss_dt"]))
            print(f"  mask_ties={str(flag):<5s}  loss_ce {np.mean(ces):.4f}   "
                  f"loss_dt {np.mean(dts):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
