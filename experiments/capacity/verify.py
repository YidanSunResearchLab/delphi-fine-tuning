"""
verify.py -- preflight for the capacity sweep. Run this BEFORE submitting anything.

Six checks, each guarding a failure mode that would otherwise waste GPU hours or, worse,
produce a comparison that is quietly confounded:

  1  param counts   shapes.n_params() agrees with the LIVE model, for every shape
  2  head geometry  n_embd % n_head == 0 (Delphi asserts this, but only at build time)
  3  config keys    every key a config sets exists in train.py's globals -- configurator
                    raises ValueError on unknown keys, and it does so AFTER the job has
                    already been scheduled and the conda env activated
  4  one variable   each config differs from config/train_delphi2m_mask_dedup.py in the
                    swept keys and eval_interval ONLY. This is the check that defends the
                    whole claim of the sweep; without it "we changed one thing" is a
                    statement of intent, not a fact.
  5  manifest       covers every (shape, seed) exactly once, indices 1..N contiguous
  6  data           the split exists and the event counts match the known fingerprint

    python experiments/capacity/verify.py                     # local (skips check 6 if no data)
    python experiments/capacity/verify.py --data-root <dir>   # point at data/ to include check 6
"""
import argparse
import ast
import io
import contextlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PKG = os.path.join(REPO, "Delphi-2M")
sys.path.insert(0, HERE)
sys.path.insert(0, PKG)

from shapes import SHAPES, SEEDS, n_params, manifest   # noqa: E402

DELIVERED = os.path.join(PKG, "config", "train_delphi2m_mask_dedup.py")
SWEPT = {"n_layer", "n_head", "n_embd"}
# eval_interval is a KNOWN, DOCUMENTED difference from the delivered config (see shapes.py).
# out_dir / seed / wandb_run_name are ABSENT from the sweep configs by design: the runner
# injects them per array task, so one config serves all three seeds. (The first run of this
# check flagged `seed` -- correctly: it IS a difference. It is whitelisted deliberately, not
# because the check was too strict.)
ALLOWED_DIFF = SWEPT | {"eval_interval", "out_dir", "seed", "wandb_run_name"}
# Known-good fingerprint of nacc-dedup-s42 (events, not bytes). test.bin is the value
# figure2/run_figure2.sh already hard-codes; the other two are recorded here for the first time.
FINGERPRINT = {"train": 800110, "val": 114143, "test": 228057}

ok_all = True


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return ok


def load_cfg(path):
    ns = {}
    with open(path) as f:
        exec(compile(f.read(), path, "exec"), ns)
    return {k: v for k, v in ns.items() if not k.startswith("_")}


def train_py_globals():
    """Top-level names train.py binds BEFORE configurator runs -- i.e. the legal config keys."""
    src = open(os.path.join(PKG, "training", "train.py")).read()
    src = src.split("config_keys = [")[0]        # everything above the configurator hand-off
    names = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(PKG, "data"))
    ap.add_argument("--dataset", default="nacc-dedup-s42")
    a = ap.parse_args()

    print("=" * 78)
    print("CAPACITY SWEEP PREFLIGHT")
    print("=" * 78)

    print("\n[1] parameter counts: analytic formula vs the live model")
    from delphi.model import Delphi, DelphiConfig
    for tag, (nl, nh, ne, _) in SHAPES.items():
        check(f"{tag}: n_embd {ne} divisible by n_head {nh}", ne % nh == 0,
              f"head_dim {ne // nh}")
        with contextlib.redirect_stdout(io.StringIO()):
            m = Delphi(DelphiConfig(n_layer=nl, n_head=nh, n_embd=ne, vocab_size=111,
                                    block_size=96, bias=False))
        live = sum(p.numel() for p in m.parameters())
        pred = n_params(nl, nh, ne)
        check(f"{tag}: params", live == pred, f"{live:,} (formula {pred:,})")

    print("\n[2] config integrity")
    legal = train_py_globals()
    check("train.py globals parsed", len(legal) > 20, f"{len(legal)} legal config keys")
    delivered = load_cfg(DELIVERED)
    for tag in SHAPES:
        p = os.path.join(HERE, "configs", f"cap_{tag}.py")
        if not check(f"{tag}: config exists", os.path.exists(p), p):
            continue
        cfg = load_cfg(p)
        unknown = sorted(set(cfg) - legal)
        check(f"{tag}: all keys known to train.py", not unknown,
              f"unknown: {unknown}" if unknown else "")
        diff = sorted(k for k in set(cfg) | set(delivered)
                      if cfg.get(k, "<absent>") != delivered.get(k, "<absent>"))
        bad = [k for k in diff if k not in ALLOWED_DIFF]
        check(f"{tag}: differs from delivered config in allowed keys only",
              not bad, f"unexpected: {bad}" if bad else f"diff = {diff}")

    # ---- head geometry actually reaches the computation --------------------
    # The L12E120H6 arm is a null experiment unless n_head genuinely changes the forward
    # pass. It does today -- but the flash-attention branch in CausalSelfAttention is
    # commented out one edit away from being enabled, and tests/golden.py's own docstring
    # warns that path silently drops masking. If a refactor ever made n_head inert, this
    # sweep would report "head geometry has no effect" as a FINDING rather than as a bug.
    # Verified with IDENTICAL weights in both models, so the only difference is the split.
    print("\n[2b] head geometry reaches the forward pass")
    import torch
    with contextlib.redirect_stdout(io.StringIO()):
        ref = Delphi(DelphiConfig(n_layer=2, n_head=12, n_embd=120, vocab_size=111,
                                  block_size=96, bias=False))
    with torch.no_grad():
        for prm in ref.parameters():
            prm.mul_(8.0)          # at init the logits are ~0 and softmax is uniform either way
    sd = ref.state_dict()
    Y = 365.25
    idx = torch.tensor([[3, 24, 106, 29, 107, 33]], dtype=torch.long)
    age = torch.tensor([[0., 70 * Y, 70 * Y, 71 * Y, 72 * Y, 72 * Y]])
    outs = {}
    for nh in (12, 6):
        with contextlib.redirect_stdout(io.StringIO()):
            m = Delphi(DelphiConfig(n_layer=2, n_head=nh, n_embd=120, vocab_size=111,
                                    block_size=96, bias=False))
        m.load_state_dict(sd); m.eval()
        with torch.no_grad():
            outs[nh] = m(idx, age)[0][0, -1]
    d = float((outs[12] - outs[6]).abs().max())
    check("n_head changes the forward pass at identical weights", d > 1e-3,
          f"max|logit diff| = {d:.4f} (12 heads vs 6 heads)")

    print("\n[3] manifest")
    rows = manifest()
    idxs = [r[0] for r in rows]
    pairs = [(r[1], r[2]) for r in rows]
    check("indices contiguous from 1", idxs == list(range(1, len(rows) + 1)), f"1..{len(rows)}")
    check("every (shape, seed) exactly once",
          sorted(pairs) == sorted((t, s) for t in SHAPES for s in SEEDS),
          f"{len(SHAPES)} shapes x {len(SEEDS)} seeds = {len(rows)}")
    mpath = os.path.join(HERE, "manifest.tsv")
    if check("manifest.tsv exists", os.path.exists(mpath)):
        lines = open(mpath).read().strip().split("\n")
        check("manifest.tsv in sync with shapes.py", len(lines) - 1 == len(rows),
              f"{len(lines) - 1} data rows")

    print("\n[4] dataset")
    import numpy as np
    ddir = os.path.join(a.data_root, a.dataset)
    if not os.path.isdir(ddir):
        print(f"  [SKIP] {ddir} not present (expected off-cluster) -- check runs on the cluster")
    else:
        for split, expect in FINGERPRINT.items():
            f = os.path.join(ddir, f"{split}.bin")
            if not check(f"{split}.bin exists", os.path.exists(f)):
                continue
            n = os.path.getsize(f) // 12
            check(f"{split}.bin event count", n == expect, f"{n:,} (expect {expect:,})")
        d = np.fromfile(os.path.join(ddir, "train.bin"), dtype=np.uint32).reshape(-1, 3)
        check("token ids within disk vocab (<=109)", int(d[:, 2].max()) <= 109,
              f"max token {int(d[:, 2].max())}")
        check("ages non-negative", int(d[:, 1].min()) >= 0, f"min age {int(d[:, 1].min())} days")

    print("\n" + "=" * 78)
    print("PREFLIGHT PASSED -- safe to submit" if ok_all else "PREFLIGHT FAILED -- DO NOT SUBMIT")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
