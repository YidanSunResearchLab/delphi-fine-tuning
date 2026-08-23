"""
golden.py -- regression baseline for the Delphi model. The safety net for framework changes.

WHY: this repo has no tests. A change to model.py (enabling flash attention, deleting the
unused attention stacking, adding a KV cache) can silently alter BEHAVIOUR while still
training fine and producing plausible figures. The classic case: the flash-attention branch
passes attn_mask=None, which drops the padding mask AND mask_ties -- the model then peeks at
same-visit tokens, loss improves, AUC goes up, and nothing errors out.

HOW: run the model on a handful of hand-written, fixed inputs and store every output number
to disk ("golden.npz"). After changing the framework, re-run and diff against the stored
numbers. A pure speed optimisation must not move them.

USAGE
    python Delphi-2M/tests/golden.py save     # record the baseline (do this BEFORE changing anything)
    python Delphi-2M/tests/golden.py check    # compare current code against the baseline

    Optional: --ckpt PATH   (default: <repo root>/ckpt.pt)
              --atol 1e-6   (tolerance; use 0 to demand bit-identical output)

Needs only the checkpoint -- no data/*.bin, so it runs on a laptop CPU.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))          # Delphi-2M/tests/
PKG = os.path.dirname(HERE)                                # Delphi-2M/
REPO = os.path.dirname(PKG)                                # repo root
sys.path.insert(0, PKG)

from delphi.ad_engine import load_model                     # noqa: E402  (the live loader)

Y = 365.25                                                  # days per year
GOLDEN = os.path.join(HERE, "golden.npz")

# ---------------------------------------------------------------- the fixed inputs
# Tokens are MODEL space (== labels.csv row index): 0=Padding, 1=No-event, 3=Female,
# 24..28=CDRSUM, 29..32=MOCA, 33..36=FAQ, 106..109=NACCUDSD, 110=Death.
# Ages are DAYS from birth; -10000 is the padding age produced by get_batch.
# These are SYNTHETIC. They do not need to be clinically realistic -- they only need to be
# FIXED, and to collectively exercise every branch in Delphi.forward().
CASES = {
    # smallest possible input: catches shape/indexing bugs. Its two LOSSES are NaN and are
    # meant to be: with one token the only target is the shifted-in padding 0, which
    # ignore_tokens drops, so both losses average over an empty selection. The logits and
    # the (b, 1) gather path are still worth pinning -- squeeze((1, 2)) used to crash here.
    "single_token": dict(tokens=[3], ages=[0.0]),

    # strictly increasing ages: no same-visit ties, the easy path
    "no_ties": dict(tokens=[3, 24, 106, 29],
                    ages=[0.0, 70 * Y, 71 * Y, 72 * Y]),

    # THE important case: two visits, 4 tokens each at IDENTICAL ages.
    # This is the only case that exercises the mask_ties branch of the attention mask.
    "with_ties": dict(tokens=[3, 24, 29, 33, 106, 25, 30, 107],
                      ages=[0.0, 70 * Y, 70 * Y, 70 * Y, 70 * Y, 72 * Y, 72 * Y, 72 * Y]),

    # leading padding, exactly as get_batch emits it: token 0 at age -10000.
    # Exercises the "do not attend to padded positions" + diagonal-restore logic.
    "with_padding": dict(tokens=[0, 0, 3, 24, 106],
                         ages=[-10000.0, -10000.0, 0.0, 70 * Y, 71 * Y]),

    # contains the Death token, which terminates trajectories in generate()
    "with_death": dict(tokens=[3, 24, 106, 110],
                       ages=[0.0, 70 * Y, 71 * Y, 75 * Y]),
}

# generate() is stochastic; these make it reproducible.
GEN_SEEDS = ["no_ties", "with_ties"]      # which cases to roll trajectories from
GEN_SEED = 1234                           # RNG seed, fixed
GEN_N = 8                                 # trajectories per case
GEN_MAX_NEW = 16                          # sampling steps
GEN_MAX_AGE = 95 * Y                      # stop at age 95
DEATH = 110


def _tensors(case, device):
    idx = torch.tensor([case["tokens"]], dtype=torch.long, device=device)
    age = torch.tensor([case["ages"]], dtype=torch.float32, device=device)
    return idx, age


@torch.no_grad()
def snapshot(model, device):
    """Run every case and return {name: numpy array} -- the 'photograph'."""
    out = {}

    for name, case in CASES.items():
        idx, age = _tensors(case, device)

        # --- path A: inference forward (targets=None -> mask_ties inactive, no loss)
        logits, loss, _ = model(idx, age)
        assert loss is None
        out[f"{name}/logits_infer"] = logits[0].float().cpu().numpy()

        # --- path B: training forward (targets given -> mask_ties ACTIVE, losses returned)
        # Targets are the next token / next age, the same shift get_batch applies.
        tgt = torch.cat([idx[:, 1:], torch.zeros_like(idx[:, :1])], dim=1)
        tgt_age = torch.cat([age[:, 1:], age[:, -1:]], dim=1)
        logits, loss, att = model(idx, age, targets=tgt, targets_age=tgt_age)
        out[f"{name}/logits_train"] = logits[0].float().cpu().numpy()
        # The two scalar losses are the most sensitive tripwires here: if the attention mask
        # changes at all, loss_ce moves. Cheap to compare, impossible to fake.
        out[f"{name}/loss_ce"] = np.array(float(loss["loss_ce"]))
        out[f"{name}/loss_dt"] = np.array(float(loss["loss_dt"]))
        # attention shape only (the values are huge); catches accidental layout changes
        out[f"{name}/att_shape"] = np.array(att.shape, dtype=np.int64)

    # --- path C: trajectory sampling (the eval hot path; needs a fixed RNG seed)
    for name in GEN_SEEDS:
        case = CASES[name]
        idx = torch.tensor([case["tokens"]] * GEN_N, dtype=torch.long, device=device)
        age = torch.tensor([case["ages"]] * GEN_N, dtype=torch.float32, device=device)
        torch.manual_seed(GEN_SEED)          # <-- reset immediately before, every time
        gi, ga, _ = model.generate(idx, age, max_new_tokens=GEN_MAX_NEW,
                                   max_age=GEN_MAX_AGE, termination_tokens=[DEATH])
        out[f"{name}/gen_tokens"] = gi.cpu().numpy().astype(np.int64)
        out[f"{name}/gen_ages"] = ga.float().cpu().numpy()

    return out


def cmd_save(args):
    model, margs = _load(args)
    snap = snapshot(model, args.device)
    np.savez_compressed(GOLDEN, **snap)
    print(f"\nwrote {GOLDEN}  ({len(snap)} arrays)")
    print("checkpoint model_args:", {k: margs[k] for k in sorted(margs) if k != "_ckpt_sig"})
    print("\nCommit golden.npz together with this script, then start changing the framework.")


def cmd_check(args):
    if not os.path.exists(GOLDEN):
        sys.exit(f"no baseline at {GOLDEN} -- run `python {__file__} save` first")
    ref = np.load(GOLDEN)
    model, _ = _load(args)
    now = snapshot(model, args.device)

    missing = sorted(set(ref.files) - set(now))
    extra = sorted(set(now) - set(ref.files))
    names = sorted(set(ref.files) & set(now))

    failed = []
    print(f"\n{'array':34s} {'max abs diff':>14s}   result")
    print("-" * 62)
    for n in names:
        a, b = np.asarray(ref[n]), np.asarray(now[n])
        if a.shape != b.shape:
            print(f"{n:34s} {'SHAPE ' + str(a.shape) + '->' + str(b.shape):>14s}   FAIL")
            failed.append(n)
            continue
        if a.dtype.kind in "iu":                    # integers must match exactly
            d = float(np.abs(a.astype(np.int64) - b.astype(np.int64)).max()) if a.size else 0.0
            ok = d == 0
        else:
            # NaN in BOTH is "unchanged", not a failure. `single_token`'s two losses are NaN
            # by construction (see CASES) and nan != nan, so a naive max|a-b| would report a
            # permanent FAIL and make this harness useless. NaN in only one side still fails,
            # which is the case that matters: a change that introduced a NaN.
            d = float(np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b)).max()) if a.size else 0.0
            ok = d <= args.atol
        print(f"{n:34s} {d:14.3e}   {'PASS' if ok else 'FAIL'}")
        if not ok:
            failed.append(n)

    for n in missing:
        print(f"{n:34s} {'MISSING':>14s}   FAIL")
    for n in extra:
        print(f"{n:34s} {'NEW':>14s}   (not in baseline)")

    print("-" * 62)
    if failed or missing:
        print(f"FAILED: {len(failed) + len(missing)} array(s) changed (atol={args.atol:g}).")
        print("Your change altered model BEHAVIOUR, not just speed. Look at the first FAIL above:")
        print("  *_ties/*        -> the same-visit (mask_ties) masking changed")
        print("  *_padding/*     -> the padding mask changed")
        print("  */gen_*         -> the sampling path changed (RNG call order counts!)")
        sys.exit(1)
    print(f"ALL PASS ({len(names)} arrays, atol={args.atol:g}). Behaviour is unchanged.")


def _load(args):
    if not os.path.exists(args.ckpt):
        sys.exit(f"checkpoint not found: {args.ckpt}")
    try:
        model, margs = load_model(args.ckpt, device=args.device)
    except TypeError as e:
        sys.exit(f"ad_engine.load_model failed ({e}).\n"
                 f"The checkpoint's model_args holds a key DelphiConfig does not accept. "
                 f"delphi.model.load_checkpoint filters these out; ad_engine.load_model does not.")
    model.eval()        # dropout OFF -- otherwise the forward pass is random and nothing matches
    return model, margs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["save", "check"])
    p.add_argument("--ckpt", default=os.path.join(REPO, "ckpt.pt"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--atol", type=float, default=1e-6,
                   help="tolerance for float comparison; 0 demands bit-identical output")
    a = p.parse_args()
    (cmd_save if a.mode == "save" else cmd_check)(a)
