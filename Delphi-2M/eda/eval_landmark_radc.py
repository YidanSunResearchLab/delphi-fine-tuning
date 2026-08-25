"""
eval_landmark_radc.py -- landmark evaluation of a RADC checkpoint, for the BMI ablation.

WHY THIS EXISTS. figure2/ hard-codes NACC token ids (NACCUDSD 106-109, Death 110) and cannot
read a RADC checkpoint. This is the minimum honest substitute: the same landmark design
figure2 uses, on the RADC vocabulary.

THE DESIGN, and why each piece is what it is.

  Landmark.  Condition on birth statics + the ENTIRE first visit, nothing after. This is
  figure2_core._baseline's rule. It matters for the ablation: the BMI-missingness artifact
  lives ~10 years later in these subjects' lives, so it is not in the prompt -- any effect it
  has must arrive through the learned dynamics, which is exactly the question.

  Outcome.  First occurrence of the AD-dementia token (model id 67) within h years of the
  landmark.

  Censoring.  For horizon h a subject counts only if the answer is KNOWN: either AD occurred
  within h years (label 1), or the subject was still observed at landmark+h without it
  (label 0). Everyone else is dropped, and the dropped count is reported -- a silently
  shrinking denominator would make the two arms incomparable.

  Prediction.  n_mc autoregressive rollouts from the prompt; risk = fraction of rollouts
  containing the AD token within h years.

  generate() settings that are NOT the defaults, and why:
    no_repeat=False   -- the default masks every token already in the sequence. RADC is mostly
                         oscillating ordinal scales (that is what keep-transitions encodes) and
                         has only 67 content tokens, so no_repeat both forbids recovery and
                         exhausts the vocabulary inside a 96-token rollout.
    termination_tokens=[68] -- RADC's Death id, which is RESERVED but never emitted
                         (CFG["EMIT_DEATH"]=False). So nothing terminates early; rollouts stop
                         at max_age. Passing [] would build an empty tensor for torch.isin.

  visit_sizes is left None (one token per sampled time) rather than resampled from the
  observed distribution. That understates events-per-visit, but identically in both arms, and
  the ablation reads the DIFFERENCE.

Metrics are paired across arms by construction: same split, same landmark rule, same
evaluable subjects (the script intersects them and says so).

Run:  python eda/eval_landmark_radc.py --ckpt out-radc-ablation/bmi_s42/ckpt.pt \
                                       --dataset radc-dedup-s42 [--limit 300] [--n-mc 50]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Delphi-2M/
sys.path.insert(0, HERE)
from delphi.model import Delphi, DelphiConfig     # noqa: E402
from delphi.utils import get_p2i                  # noqa: E402

D = 365.25
AD_TOKEN = 67          # model space: "Alzheimer's dementia diagnosis"
DEATH_TOKEN = 68       # reserved, never emitted -- used only as a no-op terminator
HORIZONS = (3, 5, 10)


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = Delphi(DelphiConfig(**ck["model_args"])).to(device).eval()
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    return m, ck


def subjects(split_path):
    d = np.fromfile(split_path, dtype=np.uint32).reshape(-1, 3)
    p2i = get_p2i(d)
    for s, n in p2i:
        sl = d[int(s):int(s) + int(n)]
        yield int(sl[0, 0]), sl[:, 1].astype(np.float64), sl[:, 2].astype(np.int64) + 1


def landmark(ages, toks):
    """(base_day, prompt_mask) or None. Same rule as figure2_core._baseline: the first
    distinct visit age, prompt = everything at or before it."""
    va = np.unique(ages[ages > 0])
    if len(va) < 2:
        return None
    base = float(va[0])
    if (toks[ages <= base] == AD_TOKEN).any():
        return None                      # already diagnosed at the landmark: nothing to predict
    return base, ages <= base


def evaluate(ckpt_path, dataset, split="test", device="cpu", n_mc=50, limit=None,
             max_new_tokens=96, seed=0, batch_subjects=24):
    """Landmark risk for every evaluable subject.

    Subjects are rolled out in BATCHES, not one at a time: a per-subject generate() call is
    96 sequential forward passes on a batch of n_mc, and at 797 subjects x 6 checkpoints that
    is hours. Batching `batch_subjects` subjects together makes each call (batch_subjects *
    n_mc) rows wide at the same 96 steps -- the model is 0.3M params, so width is nearly free
    and the step count is what costs.

    Prompts are LEFT-padded to a common length with token 0 / age -10000 (the mask_time
    get_batch uses). Left, not right: generate() reads logits[:, -1] and appends there, so the
    final column must be the last REAL token. delphi's attention mask is built from (idx > 0),
    so the pads are not attended to.
    """
    model, ck = load(ckpt_path, device)
    path = os.path.join(HERE, "data", dataset, f"{split}.bin")

    pend = []
    for pid, ages, toks in subjects(path):
        lm = landmark(ages, toks)
        if lm is None:
            continue
        base, pm = lm
        after = ages > base
        ad_hit = (toks == AD_TOKEN) & after
        pend.append(dict(pid=pid, base=base,
                         p_tok=toks[pm], p_age=ages[pm],
                         t_ad=(ages[ad_hit].min() - base) / D if ad_hit.any() else np.inf,
                         t_last=(ages[ages > 0].max() - base) / D))
        if limit is not None and len(pend) >= limit:
            break

    rows = []
    for b0 in range(0, len(pend), batch_subjects):
        chunk = pend[b0:b0 + batch_subjects]
        L = max(len(c["p_tok"]) for c in chunk)
        T = np.zeros((len(chunk), L), dtype=np.int64)
        A = np.full((len(chunk), L), -10000.0, dtype=np.float32)
        for i, c in enumerate(chunk):                     # LEFT-pad
            n = len(c["p_tok"])
            T[i, L - n:] = c["p_tok"]; A[i, L - n:] = c["p_age"]
        T = np.repeat(T, n_mc, axis=0); A = np.repeat(A, n_mc, axis=0)
        base_v = np.repeat(np.array([c["base"] for c in chunk]), n_mc)

        torch.manual_seed(seed + b0)
        with torch.no_grad():
            gi, ga, _ = model.generate(
                torch.as_tensor(T, device=device), torch.as_tensor(A, device=device),
                max_new_tokens=max_new_tokens,
                max_age=float(base_v.max()) + max(HORIZONS) * D,
                no_repeat=False, termination_tokens=[DEATH_TOKEN])
        gi = gi.cpu().numpy(); ga = ga.cpu().numpy()
        hit = (gi[:, L:] == AD_TOKEN) & (ga[:, L:] > base_v[:, None])
        t_sim = np.where(hit.any(1),
                         (np.where(hit, ga[:, L:], np.inf).min(1) - base_v) / D, np.inf)
        t_sim = t_sim.reshape(len(chunk), n_mc)

        for i, c in enumerate(chunk):
            rows.append(dict(pid=c["pid"], base_y=c["base"] / D, t_ad=c["t_ad"],
                             t_last=c["t_last"],
                             **{f"risk{h}": float((t_sim[i] <= h).mean()) for h in HORIZONS}))
    return rows, ck


def auroc(y, s):
    """Rank-based AUROC, ties averaged. Returns nan if one class is empty."""
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    sv = s[order]                                   # average ranks within ties
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def score(rows):
    out = {}
    for h in HORIZONS:
        y, s, dropped = [], [], 0
        for r in rows:
            if r["t_ad"] <= h:
                y.append(True); s.append(r[f"risk{h}"])
            elif r["t_last"] >= h:              # observed past the horizon, still no AD
                y.append(False); s.append(r[f"risk{h}"])
            else:
                dropped += 1                    # censored before the horizon: answer unknown
        out[f"h{h}"] = dict(n=len(y), n_pos=int(np.sum(y)), censored_dropped=dropped,
                            auroc=auroc(y, s), mean_risk=float(np.mean(s)) if s else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-mc", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-subjects", type=int, default=24)
    # 48 is enough: rollouts already reach a median 20y / p10 10y of simulated time,
    # identical coverage to 96 at half the sequential forward passes.
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    rows, ck = evaluate(a.ckpt, a.dataset, a.split, a.device, a.n_mc, a.limit,
                        seed=a.seed, batch_subjects=a.batch_subjects,
                        max_new_tokens=a.max_new_tokens)
    res = score(rows)
    print(f"\n{a.ckpt}  ({a.dataset}/{a.split}, iter {ck['iter_num']}, "
          f"best_val {ck['best_val_loss']:.4f})")
    print(f"  evaluable subjects: {len(rows)}   n_mc={a.n_mc}")
    for h in HORIZONS:
        r = res[f"h{h}"]
        print(f"  {h:>2}y: AUROC {r['auroc']:.4f}   n={r['n']} (pos {r['n_pos']}, "
              f"censored-dropped {r['censored_dropped']})   mean risk {r['mean_risk']:.3f}")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(dict(ckpt=a.ckpt, dataset=a.dataset, n_mc=a.n_mc,
                           iter_num=ck["iter_num"], best_val_loss=ck["best_val_loss"],
                           n_eval=len(rows), result=res, rows=rows), f, indent=1)
        print(f"  -> {a.json_out}")


if __name__ == "__main__":
    main()
