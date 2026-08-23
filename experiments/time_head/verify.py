"""
verify.py -- preflight for the time-head experiment. Run this BEFORE submitting anything.

    python experiments/time_head/verify.py                     # local
    python experiments/time_head/verify.py --data-root <dir>   # include the data check

Nine checks. Four of them ([2]-[5]) are not bookkeeping -- they are the experiment's premise.
The claim being made is "these runs differ from the capacity sweep's runs in exactly one
thing: the timing objective got its own head". Everything that could quietly falsify that is
checked here rather than asserted in prose:

  1  wiring        every arm's architecture comes from its control's entry in the capacity
                   sweep's shapes.py, and the seeds match, so the controls exist
  2  param count   the live model is trunk + (n_embd + 1) and nothing else -- catches a head
                   that accidentally became an MLP, or a config key that silently did nothing
  3  controlled    at one seed, time_head=True and time_head=False are BIT-IDENTICAL on every
     pair         shared parameter. If a future edit builds the head before the shared init
                   again, the pair stops being controlled and every delta becomes seed noise
                   wearing a lab coat. This is the check that caught exactly that bug.
  4  identities    softmax(logits) unchanged; logsumexp(logits) == the head's output;
                   loss_ce bit-identical to the single-head model at the same weights. These
                   are what let generate(), ad_engine, predict_adapter and figure2 pick up the
                   new head with NO code change and still mean what they say.
  5  routing       d loss_ce / d(time head) == 0, and d loss_dt / d(time head) != 0. i.e. the
                   head is trained by the timing objective ALONE. Without this the arm is a
                   reparameterisation of the old model, not an independent head.
  6  config        every key is legal to train.py, and each config differs from its control's
                   capacity config in {time_head} only
  7  manifest      covers every (arm, seed) exactly once, indices 1..N contiguous
  8  controls      the capacity runs this experiment pairs against are actually on disk
  9  data          the split exists and the event counts match the known fingerprint
"""
import argparse
import ast
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PKG = os.path.join(REPO, "Delphi-2M")
sys.path.insert(0, HERE)
sys.path.insert(0, PKG)

from arms import (ARMS, SEEDS, CAP, CAP_SHAPES, control, shape, n_params,   # noqa: E402
                  head_params, manifest)

CAPCFG = os.path.join(CAP, "configs")
CAPRUNS = os.path.join(CAP, "runs")
# The one key that may differ from the control's config. out_dir / seed / wandb_run_name are
# absent from both by design (the runner injects them per array task).
ALLOWED_DIFF = {"time_head", "out_dir", "seed", "wandb_run_name"}
FINGERPRINT = {"train": 800110, "val": 114143, "test": 228057}
Y = 365.25

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
    src = src.split("config_keys = [")[0]
    names = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def build(time_head, nl, nh, ne, seed=None):
    """A model at a fixed seed, stdout suppressed (Delphi prints its parameter count)."""
    import torch
    from delphi.model import Delphi, DelphiConfig
    if seed is not None:
        torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return Delphi(DelphiConfig(n_layer=nl, n_head=nh, n_embd=ne, vocab_size=111,
                                   block_size=96, bias=False, t_min=Y / 12,
                                   ignore_tokens=list(range(22)), time_head=time_head))


def fixture(device="cpu"):
    """One hand-written stream that exercises ties, padding and a real target sequence."""
    import torch
    idx = torch.tensor([[3, 24, 106, 29, 107, 33, 108, 0]], dtype=torch.long, device=device)
    age = torch.tensor([[0., 70 * Y, 70 * Y, 71 * Y, 72 * Y, 72 * Y, 74 * Y, -10000.]],
                       device=device)
    tgt = torch.cat([idx[:, 1:], torch.zeros_like(idx[:, :1])], dim=1)
    tga = torch.cat([age[:, 1:], age[:, -1:]], dim=1)
    return idx, age, tgt, tga


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(PKG, "data"))
    ap.add_argument("--dataset", default="nacc-dedup-s42")
    a = ap.parse_args()

    import torch

    print("=" * 78)
    print("TIME-HEAD EXPERIMENT PREFLIGHT")
    print("=" * 78)

    # ------------------------------------------------------------------ [1] wiring
    print("\n[1] arms are wired to real capacity-sweep controls")
    for arm in ARMS:
        ctl = control(arm)
        check(f"{arm}: control {ctl} exists in capacity/shapes.py", ctl in CAP_SHAPES,
              f"{shape(arm)}" if ctl in CAP_SHAPES else "unknown tag")
    from shapes import SEEDS as CAP_SEEDS
    check("seeds match the capacity sweep", tuple(SEEDS) == tuple(CAP_SEEDS),
          f"{tuple(SEEDS)}")

    # ------------------------------------------------------------------ [2] param counts
    print("\n[2] parameter counts: analytic vs the live model, and the head's exact size")
    for arm in ARMS:
        nl, nh, ne = shape(arm)
        off, on = build(False, nl, nh, ne), build(True, nl, nh, ne)
        n_off = sum(p.numel() for p in off.parameters())
        n_on = sum(p.numel() for p in on.parameters())
        check(f"{arm}: params", n_on == n_params(arm), f"{n_on:,} (formula {n_params(arm):,})")
        check(f"{arm}: head is exactly n_embd+1", n_on - n_off == head_params(ne),
              f"+{n_on - n_off} params ({100 * (n_on - n_off) / n_on:.3f}% of the model)")
        extra = sorted(set(on.state_dict()) - set(off.state_dict()))
        check(f"{arm}: adds only the head's tensors", extra == ["time_head.bias", "time_head.weight"],
              str(extra))

    # ------------------------------------------------------------------ [3] controlled pair
    # THE check. The controls are not re-run, so if the trunk init diverges the whole
    # experiment silently degrades into comparing two unrelated initialisations.
    print("\n[3] time_head=True shares its trunk init with the control, bit-for-bit")
    for arm in ARMS:
        nl, nh, ne = shape(arm)
        for s in (SEEDS[0],):
            sd_off = build(False, nl, nh, ne, seed=s).state_dict()
            sd_on = build(True, nl, nh, ne, seed=s).state_dict()
            bad = [k for k in sd_off if not torch.equal(sd_off[k], sd_on[k])]
            check(f"{arm} @ seed {s}: every shared parameter identical", not bad,
                  f"{len(sd_off)} tensors" if not bad else f"DIVERGED: {bad[:4]}")

    # ------------------------------------------------------------------ [4] identities
    print("\n[4] the algebraic identities the drop-in relies on")
    nl, nh, ne = shape(next(iter(ARMS)))
    off = build(False, nl, nh, ne, seed=SEEDS[0])
    on = build(True, nl, nh, ne, seed=SEEDS[0])
    on.load_state_dict(off.state_dict(), strict=False)      # identical shared weights
    off.eval(); on.eval()
    idx, age, tgt, tga = fixture()
    with torch.no_grad():
        rate_logits, _, _ = on(idx, age)
        tok_logits, _, _ = off(idx, age)
        d_soft = float((torch.softmax(rate_logits, -1) - torch.softmax(tok_logits, -1)).abs().max())
        # the head's output, recomputed independently of forward()
        h = on.transformer.wte(idx) + on.transformer.wae(age.unsqueeze(-1))
        T = idx.size(1)
        tril = torch.tril(torch.ones(T, T))[None, None] > 0
        am = (idx > 0).view(1, 1, 1, T) * (idx > 0).view(1, 1, T, 1) * tril
        am = (am + (idx == 0).view(1, 1, 1, T) * (torch.diag(torch.ones(T)) > 0)) * tril
        for blk in on.transformer.h:
            h, _ = blk(h, am)
        log_lambda = on.time_head(on.transformer.ln_f(h)).squeeze(-1)
        d_lse = float((torch.logsumexp(rate_logits, -1) - log_lambda).abs().max())
        _, l_off, _ = off(idx, age, tgt, tga)
        _, l_on, _ = on(idx, age, tgt, tga)
        d_ce = abs(float(l_off["loss_ce"]) - float(l_on["loss_ce"]))
    check("softmax(logits) unchanged by the head", d_soft < 1e-6,
          f"max|d| = {d_soft:.2e}  (a per-position constant cancels in softmax)")
    check("logsumexp(logits) == the head's output", d_lse < 1e-5,
          f"max|d| = {d_lse:.2e}  (so generate()'s competing exponentials get rate_k = lambda*p_k)")
    check("loss_ce identical at identical shared weights", d_ce == 0.0,
          f"|d| = {d_ce:.2e}  ('which event' is untouched)")
    check("loss_dt actually changed", abs(float(l_off["loss_dt"]) - float(l_on["loss_dt"])) > 1e-6,
          f"{float(l_off['loss_dt']):.4f} -> {float(l_on['loss_dt']):.4f}  (the point of the arm)")

    # ------------------------------------------------------------------ [5] gradient routing
    print("\n[5] the head is trained by the timing objective alone")
    on.train()
    _, loss, _ = on(idx, age, tgt, tga)
    hp = list(on.time_head.parameters())
    g_ce = torch.autograd.grad(loss["loss_ce"], hp, retain_graph=True, allow_unused=True)
    g_dt = torch.autograd.grad(loss["loss_dt"], hp, allow_unused=True)
    m_ce = max(float(g.abs().max()) for g in g_ce)
    m_dt = max(float(g.abs().max()) for g in g_dt)
    check("d loss_ce / d(time head) == 0", m_ce < 1e-5,
          f"max|grad| = {m_ce:.2e}  (analytically zero; softmax drops the constant)")
    check("d loss_dt / d(time head) != 0", m_dt > 1e-3,
          f"max|grad| = {m_dt:.3f}")

    # Same objective, same floor regime, same starting point. loss_dt reaches the trunk only
    # through the floor, whose derivative is exp(-lse)/(exp(-lse) + t_min); the control starts
    # deeply saturated at lse = log(vocab_size). A zero-biased head would start ~111x further
    # from the floor and take ~1100x more dt gradient at step 0 -- any loss_dt win would then
    # be confounded with simply having started outside the floor. model.py warm-starts the
    # bias to log(vocab_size); these two lock that in.
    import math as _math
    off.train()
    _, l_off_tr, _ = off(idx, age, tgt, tga)
    d0 = float(l_off_tr["loss_dt"]); d1 = float(loss["loss_dt"])
    check("head is warm-started to the control's intensity",
          abs(float(on.time_head.bias) - _math.log(on.config.vocab_size)) < 1e-6,
          f"bias = {float(on.time_head.bias):.4f}  vs  log(vocab_size) = "
          f"{_math.log(on.config.vocab_size):.4f}")
    check("both arms start in the same floor regime",
          abs(d1 - d0) / max(abs(d0), 1e-9) < 1e-3,
          f"loss_dt at init {d0:.4f} vs {d1:.4f}  (rel {abs(d1-d0)/max(abs(d0),1e-9):.2e}; "
          f"a zero bias gives a whole-number gap)")

    # Weight decay must not fall on one arm's intensity and not the other's. The control's is
    # logsumexp(lm_head(x)) and lm_head.weight is tied to the blacklisted transformer.wte.weight,
    # so it is undecayed; decaying time_head.weight pulls log lambda toward a constant, which is
    # the null this arm tests against.
    with contextlib.redirect_stdout(io.StringIO()):
        opt = on.configure_optimizers(0.2, 1e-3, (0.9, 0.95), "cpu")
    decayed = {id(q) for q in opt.param_groups[0]["params"]}
    check("the head's intensity is not weight-decayed, like the control's",
          id(on.time_head.weight) not in decayed and id(on.time_head.bias) not in decayed,
          "both time_head tensors are in the no_decay group")

    # ------------------------------------------------------------------ [6] configs
    print("\n[6] config integrity")
    legal = train_py_globals()
    check("train.py globals parsed", len(legal) > 20, f"{len(legal)} legal config keys")
    check("train.py knows `time_head`", "time_head" in legal,
          "configurator raises ValueError on unknown keys -- AFTER the job is scheduled")
    for arm in ARMS:
        p = os.path.join(HERE, "configs", f"th_{arm}.py")
        if not check(f"{arm}: config exists", os.path.exists(p), p):
            continue
        cfg = load_cfg(p)
        unknown = sorted(set(cfg) - legal)
        check(f"{arm}: all keys known to train.py", not unknown,
              f"unknown: {unknown}" if unknown else "")
        check(f"{arm}: sets time_head=True", cfg.get("time_head") is True,
              repr(cfg.get("time_head", "<absent>")))
        ctlcfg = load_cfg(os.path.join(CAPCFG, f"cap_{control(arm)}.py"))
        diff = sorted(k for k in set(cfg) | set(ctlcfg)
                      if cfg.get(k, "<absent>") != ctlcfg.get(k, "<absent>"))
        bad = [k for k in diff if k not in ALLOWED_DIFF]
        check(f"{arm}: differs from cap_{control(arm)}.py in {{time_head}} only",
              not bad and diff == ["time_head"],
              f"unexpected: {bad}" if bad else f"diff = {diff}")

    # ------------------------------------------------------------------ [7] manifest
    print("\n[7] manifest")
    rows = manifest()
    idxs = [r[0] for r in rows]
    pairs = [(r[1], r[3]) for r in rows]
    check("indices contiguous from 1", idxs == list(range(1, len(rows) + 1)), f"1..{len(rows)}")
    check("every (arm, seed) exactly once",
          sorted(pairs) == sorted((t, s) for t in ARMS for s in SEEDS),
          f"{len(ARMS)} arms x {len(SEEDS)} seeds = {len(rows)}")
    mpath = os.path.join(HERE, "manifest.tsv")
    if check("manifest.tsv exists", os.path.exists(mpath)):
        lines = open(mpath).read().strip().split("\n")
        check("manifest.tsv in sync with arms.py", len(lines) - 1 == len(rows),
              f"{len(lines) - 1} data rows")

    # ------------------------------------------------------------------ [8] controls on disk
    print("\n[8] the control runs this experiment pairs against")
    if not os.path.isdir(CAPRUNS):
        print(f"  [SKIP] {CAPRUNS} not present -- pull the capacity runs before analyzing")
    else:
        for arm in ARMS:
            for s in SEEDS:
                rd = os.path.join(CAPRUNS, f"{control(arm)}_s{s}")
                have_log = os.path.exists(os.path.join(rd, "train.log"))
                have_ck = os.path.exists(os.path.join(rd, "ckpt.pt"))
                check(f"control {control(arm)}_s{s}", have_log,
                      "train.log" + (" + ckpt.pt" if have_ck else " (no ckpt.pt: "
                                     "val-loss pairing only, no decomposition)"))

    # ------------------------------------------------------------------ [9] data
    print("\n[9] dataset")
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

    print("\n" + "=" * 78)
    print("PREFLIGHT PASSED -- safe to submit" if ok_all else "PREFLIGHT FAILED -- DO NOT SUBMIT")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
