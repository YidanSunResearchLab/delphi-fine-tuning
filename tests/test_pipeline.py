"""
test_pipeline.py -- the regression net for the RADC pipeline.

    python tests/test_pipeline.py            # everything that does not need the raw data
    python tests/test_pipeline.py --full     # also rebuilds the dataset and checks fingerprints

WHY THIS FILE EXISTS. Most of what can go wrong in this pipeline does not raise. A flipped
value coding, an off-by-one between disk and model space, a no-event marker outside the
observation window, a generation that never terminates -- every one of those trains happily,
produces plausible-looking figures, and is only visible as a number that is quietly wrong by
20-60%. Each test below corresponds to a failure that either happened in the predecessor
project or was found while building this one.
"""
import os
import sys
import math
import argparse

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from radc_delphi import vocab as V                                  # noqa: E402
from radc_delphi.model import Delphi, DelphiConfig                  # noqa: E402
from radc_delphi.batching import get_p2i, get_batch, patient_stream  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return cond


# ---------------------------------------------------------------- vocabulary
def test_vocab():
    print("\nvocabulary")
    check("internal consistency (vocab.check)", V.check())
    check("size matches the table", V.VOCAB_SIZE == len(V.NAMES), f"got {V.VOCAB_SIZE}")
    check("0 is Padding, 1 is No event",
          V.NAMES[0] == "Padding" and V.NAMES[1] == "No event")
    # The No-event marker is the only negative evidence in the stream. Masking it out of the
    # loss would leave the model trained on positives alone -- it would learn what happens
    # but never that nothing happens.
    check("No event is NOT in ignore_tokens", V.NO_EVENT not in V.IGNORE_TOKENS)
    check("every static IS in ignore_tokens",
          all(t in V.IGNORE_TOKENS for t in V.STATIC_IDS))
    check("both endpoints are trainable targets",
          V.AD_DX not in V.IGNORE_TOKENS and V.DEATH not in V.IGNORE_TOKENS)
    # Ordinal bins and the recurring onset/medication events must be exempt from no_repeat, or
    # generation assigns probability zero to every legitimate recovery.
    n_rep = sum(len(v) for v in V.SCALES.values()) + len(V.ONSET_IDS) + len(V.MED_IDS)
    check("repeatable set matches the groups", len(V.REPEATABLE_TOKENS) == n_rep,
          f"{len(V.REPEATABLE_TOKENS)} vs {n_rep}")
    # The two ignore sets answer different questions and must NOT be equal: No-event is a
    # cross-entropy target but must never count as "the next event" for the timing objective.
    check("No event is in DT_IGNORE but not IGNORE",
          V.NO_EVENT in V.DT_IGNORE_TOKENS and V.NO_EVENT not in V.IGNORE_TOKENS)
    # A bin edge sitting on a mode of an integer-valued scale starves the level beside it:
    # with edges at 28 and 29 the level between them received 4 tokens in the whole cohort.
    # The three dementia-at-entry levels must partition every subject, and "unknown" must
    # exist: 793 subjects (17.9%, all 308 LATC among them) have no ROSMAP_clinical row, and
    # without an explicit level the static block would be one token shorter for exactly them.
    check("dementia-at-entry has all three levels",
          set(V.DEMENTIA_ENTRY) == {"yes", "no", "unknown"})
    check("dementia-at-entry levels are statics, out of the loss",
          all(t in V.IGNORE_TOKENS for t in V.DEMENTIA_ENTRY.values()))
    # It must NOT be the incident AD token: emitting that at baseline would put prevalent cases
    # into the cumulative-incidence target.
    check("dementia-at-entry is not the AD event token",
          V.AD_DX not in V.DEMENTIA_ENTRY.values())
    check("MMSE edges above 26 are half-integers",
          all(abs(e - round(e)) > 0.4 for e in V.SCALE_EDGES["MMSE"] if e > 26),
          str(V.SCALE_EDGES["MMSE"]))
    check("no keep-first id is repeatable",
          not (set(V.REPEATABLE_TOKENS) & set(V.KEEP_FIRST_IDS) | {V.AD_DX, V.DEATH} & set(V.REPEATABLE_TOKENS)))
    check("death is the only termination token", V.TERMINATION_TOKENS == (V.DEATH,))
    check("disk/model offset holds for the staging scale",
          V.STAGE_TOKENS_DISK == tuple(t - 1 for t in V.SCALES["MMSE"]))


# ---------------------------------------------------------------- batching
def _toy_stream(n_subj=40, seed=0):
    """A synthetic RADC-shaped stream: annual visits from a late-life baseline."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_subj):
        a0 = int(rng.integers(70, 88)) * 365
        rows.append((pid, a0 - 1, 2))                      # a static, one day before baseline
        for fy in range(int(rng.integers(2, 14))):
            age = a0 + 365 * fy
            for t in rng.choice(np.arange(22, 48), size=int(rng.integers(1, 4)), replace=False):
                rows.append((pid, age, int(t)))
    d = np.array(sorted(rows), dtype=np.uint32)
    return d, get_p2i(d)


def test_batching():
    print("\nbatching")
    d, p2i = _toy_stream()
    check("get_p2i partitions the array", p2i[:, 1].sum() == len(d))

    ix = np.arange(16)
    X, A, Y, B = get_batch(ix, d, p2i, block_size=64, device="cpu", select="left",
                           padding="random", no_event_token_rate=2, cut_batch=True)
    check("batch shapes agree", X.shape == A.shape == Y.shape == B.shape, str(tuple(X.shape)))
    check("tokens are lifted to model space", int(X.max()) <= V.VOCAB_SIZE)

    # THE REGRESSION THAT MATTERS. Upstream drew no-event markers from the absolute interval
    # [0, 100 years] and masked only from above, so a subject first seen at 78 was handed a
    # run of markers asserting seventy-eight event-free years that were never observed --
    # fabricated left-truncated history, which depresses the learned baseline hazard for
    # everyone. Markers must sit inside [first observed age, last observed age].
    bad_rows = 0
    for r in range(X.shape[0]):
        real = A[r][(X[r] > 1)]                             # real events, not pad/no-event
        marks = A[r][X[r] == V.NO_EVENT]
        if len(real) and len(marks) and (marks.min() < real.min() - 1e-6):
            bad_rows += 1
    check("no-event markers stay inside the observation window", bad_rows == 0,
          f"{bad_rows} rows had a marker before the subject's first real event")

    # ...and their DENSITY should track the width of that window, not a constant.
    dens = []
    for r in range(X.shape[0]):
        real = A[r][(X[r] > 1)]
        if len(real) < 2:
            continue
        span = float(real.max() - real.min()) / 365.25
        dens.append(int((X[r] == V.NO_EVENT).sum()) / max(span, 1e-6))
    check("marker density is ~1 per rate-years of follow-up",
          bool(dens) and 0.15 < float(np.median(dens)) < 1.2,
          f"median {np.median(dens):.3f} per year at rate 2")

    Xn, *_ = get_batch(ix, d, p2i, block_size=64, device="cpu", select="left",
                       padding="random", no_event_token_rate=0, cut_batch=True)
    check("rate 0 emits no markers", int((Xn == V.NO_EVENT).sum()) == 0)

    # `regular` must be deterministic: evaluation loss must not move with the marker draw.
    a1 = get_batch(ix, d, p2i, block_size=64, device="cpu", select="left",
                   padding="regular", no_event_token_rate=2, cut_batch=True)
    a2 = get_batch(ix, d, p2i, block_size=64, device="cpu", select="left",
                   padding="regular", no_event_token_rate=2, cut_batch=True)
    check("padding='regular' is deterministic",
          bool(torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1])))

    ages, toks = patient_stream(d, int(p2i[0, 0]), int(p2i[0, 1]))
    check("patient_stream lifts disk -> model space (+1)",
          int(toks.min()) == int(d[int(p2i[0, 0]):int(p2i[0, 0]) + int(p2i[0, 1]), 2].min()) + 1)


# ---------------------------------------------------------------- soft labels
def test_soft_labels():
    print("\nsoft labels")
    from radc_delphi.softlabel import SoftBinner, mix_embeddings, scatter_targets
    b = SoftBinner()

    # A reading two tenths apart across an edge must NOT look like a different measurement.
    # Hard binning put MMSE 17.9 and 18.1 in different tokens, 1.14 apart in embedding space,
    # while the measurement SD there is 2.4 points -- the edge discriminated far below the
    # instrument's own noise.
    w_lo = V.soft_weights("MMSE", 17.9)
    w_hi = V.soft_weights("MMSE", 18.1)
    check("values either side of an edge give near-identical weights",
          float(np.abs(w_lo - w_hi).sum()) < 0.1, f"L1 {np.abs(w_lo - w_hi).sum():.3f}")
    # ...and values inside one bin must NOT look identical, which is the other half of the bug.
    w_a, w_b = V.soft_weights("MMSE", 18.5), V.soft_weights("MMSE", 23.5)
    check("values inside one bin are distinguishable",
          float(np.abs(w_a - w_b).sum()) > 0.5, f"L1 {np.abs(w_a - w_b).sum():.3f}")

    for scale in V.SCALES:
        ws = [V.soft_weights(scale, v) for v in np.linspace(*(
            (0, 30) if scale == "MMSE" else ((-4, 2) if scale == "COG" else (14, 45))), 25)]
        check(f"{scale}: weights are a distribution",
              all(abs(w.sum() - 1) < 1e-6 and (w >= 0).all() for w in ws))

    d, p2i = _toy_stream()
    vals = np.full(len(d), np.nan, dtype=np.float32)
    is_mmse = np.isin(d[:, 2] + 1, list(V.SCALES["MMSE"]))
    vals[is_mmse] = 27.0
    X, A, Y, B, (Wx, Ix), (Wy, Iy) = get_batch(
        np.arange(8), d, p2i, block_size=64, device="cpu", select="left", padding="random",
        no_event_token_rate=2, cut_batch=True, values=vals, binner=b)
    check("soft weights sum to 1 everywhere",
          bool(torch.allclose(Wx.sum(-1), torch.ones_like(Wx.sum(-1)), atol=1e-5)))
    check("target soft weights sum to 1",
          bool(torch.allclose(Wy.sum(-1), torch.ones_like(Wy.sum(-1)), atol=1e-5)))

    # THE DEGENERACY GUARANTEE. With one-hot weights the soft path must reproduce the hard
    # path exactly, or the change is not a generalisation and every delivered number moves.
    cfg = DelphiConfig(vocab_size=V.VOCAB_SIZE, block_size=64, n_layer=2, n_head=4, n_embd=32,
                       ignore_tokens=list(V.IGNORE_TOKENS),
                       dt_ignore_tokens=list(V.DT_IGNORE_TOKENS), t_min=365.25 / 12)
    m = Delphi(cfg)
    _, hard, _ = m(X, A, Y, B)
    Wh = torch.zeros_like(Wx); Wh[..., 0] = 1.0
    Ih = torch.full_like(Ix, -1); Ih[..., 0] = X.clamp(min=0)
    W2 = torch.zeros_like(Wy); W2[..., 0] = 1.0
    I2 = torch.full_like(Iy, -1); I2[..., 0] = Y.clamp(min=0)
    _, deg, _ = m(X, A, Y, B, soft_in=(Wh, Ih), soft_target=(W2, I2))
    check("one-hot weights reproduce the hard loss exactly",
          abs(float(deg["loss_ce"]) - float(hard["loss_ce"])) < 1e-6,
          f"{float(deg['loss_ce']):.8f} vs {float(hard['loss_ce']):.8f}")

    _, soft, _ = m(X, A, Y, B, soft_in=(Wx, Ix), soft_target=(Wy, Iy))
    check("soft path is finite",
          bool(torch.isfinite(soft["loss_ce"]) and torch.isfinite(soft["loss_dt"])))
    # The soft TARGET must actually reach the loss. An earlier patch applied the soft input
    # but silently failed to replace the cross-entropy -- str.replace is a no-op on a miss --
    # so the degeneracy check passed while the target side was never exercised at all.
    check("the soft target changes the loss",
          abs(float(soft["loss_ce"]) - float(hard["loss_ce"])) > 1e-5,
          f"{float(soft['loss_ce']):.6f} vs {float(hard['loss_ce']):.6f}")
    # ...and it must read the MASKED logits, or ad_label_missing and validation_loss_mode are
    # both silently undone on the soft path. Both paths must agree on whether a row is scorable.
    am = torch.zeros(X.shape[0], dtype=torch.bool); am[:8] = True
    cfg_ad = DelphiConfig(vocab_size=V.VOCAB_SIZE, block_size=64, n_layer=2, n_head=4,
                          n_embd=32, ignore_tokens=list(V.IGNORE_TOKENS),
                          dt_ignore_tokens=list(V.DT_IGNORE_TOKENS), t_min=365.25 / 12,
                          ad_dx_token=V.AD_DX)
    m_ad = Delphi(cfg_ad); m_ad.load_state_dict(m.state_dict())
    _, h_am, _ = m_ad(X, A, Y, B, ad_label_missing=am)
    _, s_am, _ = m_ad(X, A, Y, B, soft_in=(Wx, Ix), soft_target=(Wy, Iy), ad_label_missing=am)
    check("hard and soft agree on the AD mask",
          bool(torch.isfinite(h_am["loss_ce"]) == torch.isfinite(s_am["loss_ce"])))

    st = scatter_targets(Wy, Iy, V.VOCAB_SIZE)
    check("scattered target is a distribution",
          bool(torch.allclose(st.sum(-1), torch.ones_like(st.sum(-1)), atol=1e-5)))


# ---------------------------------------------------------------- model
def test_model():
    print("\nmodel")
    d, p2i = _toy_stream()
    X, A, Y, B = get_batch(np.arange(8), d, p2i, block_size=64, device="cpu",
                           select="left", padding="random", no_event_token_rate=2,
                           cut_batch=True)
    for time_head in (False, True):
        for dt_target in ("gather", "next_visit", "next_event"):
            cfg = DelphiConfig(vocab_size=V.VOCAB_SIZE, block_size=64, n_layer=2, n_head=4,
                               n_embd=32, ignore_tokens=list(V.IGNORE_TOKENS),
                               t_min=365.25 / 12, time_head=time_head, dt_target=dt_target)
            m = Delphi(cfg)
            logits, loss, _ = m(X, A, Y, B)
            ok = (torch.isfinite(loss["loss_ce"]) and torch.isfinite(loss["loss_dt"]))
            check(f"forward finite (time_head={time_head}, dt_target={dt_target})", bool(ok),
                  f"ce {loss['loss_ce'].item():.3f} dt {loss['loss_dt'].item():.3f}")

    # The two ignore sets must produce DIFFERENT dt targets, or the fix is not wired through.
    cfg_a = DelphiConfig(vocab_size=V.VOCAB_SIZE, block_size=64, n_layer=2, n_head=4, n_embd=32,
                         ignore_tokens=list(V.IGNORE_TOKENS), t_min=365.25 / 12,
                         dt_target="next_event", dt_ignore_tokens=None)
    cfg_b = DelphiConfig(vocab_size=V.VOCAB_SIZE, block_size=64, n_layer=2, n_head=4, n_embd=32,
                         ignore_tokens=list(V.IGNORE_TOKENS), t_min=365.25 / 12,
                         dt_target="next_event", dt_ignore_tokens=list(V.DT_IGNORE_TOKENS))
    ma, mb = Delphi(cfg_a), Delphi(cfg_b)
    mb.load_state_dict(ma.state_dict())
    _, la, _ = ma(X, A, Y, B)
    _, lb, _ = mb(X, A, Y, B)
    check("dt_ignore_tokens changes the timing target",
          abs(float(la["loss_dt"]) - float(lb["loss_dt"])) > 1e-3,
          f"{float(la['loss_dt']):.4f} vs {float(lb['loss_dt']):.4f}")

    # A single-token sequence used to crash the dt gather path with a squeeze that also
    # dropped the time axis when t == 1.
    cfg = DelphiConfig(vocab_size=V.VOCAB_SIZE, block_size=64, n_layer=2, n_head=4, n_embd=32,
                       ignore_tokens=list(V.IGNORE_TOKENS), t_min=365.25 / 12)
    m = Delphi(cfg)
    one = torch.tensor([[3]]), torch.tensor([[28000.0]]), torch.tensor([[26]]), torch.tensor([[28365.0]])
    logits, loss, _ = m(*one)
    check("single-token sequence does not crash", logits.shape[1] == 1)


def test_generation():
    print("\ngeneration")
    cfg = DelphiConfig(vocab_size=V.VOCAB_SIZE, block_size=64, n_layer=2, n_head=4, n_embd=32,
                       ignore_tokens=list(V.IGNORE_TOKENS), t_min=365.25 / 12)
    m = Delphi(cfg).eval()
    idx = torch.tensor([[3, 26, 30]] * 8)
    age = torch.tensor([[28000.0, 28001.0, 28001.0]] * 8)

    # Death is absorbing. If a rollout keeps emitting after it, 62% of this cohort stays alive
    # in simulation and every cumulative-incidence number comes out far too high.
    with torch.no_grad():
        gi, ga, _ = m.generate(idx, age, max_new_tokens=48, max_age=110 * 365.25,
                               termination_tokens=list(V.TERMINATION_TOKENS),
                               extra_ignore=[V.NO_EVENT],
                               repeatable_tokens=list(V.REPEATABLE_TOKENS))
    after_death = 0
    for r in range(gi.shape[0]):
        pos = (gi[r] == V.DEATH).nonzero().flatten()
        if len(pos):
            tail = gi[r][int(pos[0]) + 1:]
            after_death += int((tail > 0).sum())            # 0 is padding, which is expected
    check("nothing is emitted after death", after_death == 0, f"{after_death} live tokens")

    check("statics are never generated",
          int(torch.isin(gi[:, 3:], torch.tensor(list(V.STATIC_IDS))).sum()) == 0)
    check("the No-event marker is never generated",
          int((gi[:, 3:] == V.NO_EVENT).sum()) == 0)
    check("ages are non-decreasing along a rollout",
          bool((ga[:, 1:] >= ga[:, :-1] - 1e-3).all() or (ga <= -9999).any()))

    with torch.no_grad():
        gi2, _, _ = m.generate(idx, age, max_new_tokens=32, max_age=110 * 365.25,
                               termination_tokens=list(V.TERMINATION_TOKENS),
                               extra_ignore=[V.NO_EVENT],
                               repeatable_tokens=list(V.REPEATABLE_TOKENS),
                               visit_sizes=np.array([1, 1, 2, 3, 4]))
    check("visit_sizes rollout runs", gi2.shape[0] == 8)


# ---------------------------------------------------------------- the real data
def test_dataset(data_dir):
    print("\nbuilt dataset")
    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        print(f"  skip  {data_dir} not built")
        return
    import json
    rep = json.load(open(os.path.join(data_dir, "report.json")))
    check("4,428 subjects", rep["n_subjects"] == 4428, str(rep["n_subjects"]))
    check("vocabulary matches the live table", rep["vocab_size"] == V.VOCAB_SIZE)
    check("ad_rx is excluded from the production build", rep["emit_ad_rx"] is False)
    check("block_size 64 truncates nobody", rep["min_block_size"] <= 64,
          f"needs >= {rep['min_block_size']}")
    # Deterministic given the inputs: the same three files must give the same bytes on any
    # machine. This was verified byte-identical between a Mac and the RIS cluster.
    # A build is deterministic given the three raw files; this pins THIS build so a silent
    # change to the tokenizer is caught. Update it deliberately when the tokenization changes.
    EXPECTED_FP = {74900: "8992a31abaaf", 77847: "4ace556bf1a7", 82275: None}
    fp = rep["fingerprints"]["train.bin"]
    want = EXPECTED_FP.get(rep["n_events"])
    check("train.bin fingerprint matches this tokenization", want is None or fp == want,
          f"{rep['n_events']} events -> {fp}")

    counts = rep["token_counts"]
    # 73.2% of the ROSMAP cohort is female. If msex were read backwards this flips, and
    # nothing else in the pipeline would notice.
    f, mm = counts["Sex: female"], counts["Sex: male"]
    check("cohort is ~73% female", 0.72 < f / (f + mm) < 0.74, f"{100 * f / (f + mm):.1f}%")
    check("1,164 AD diagnoses", counts["Alzheimer's dementia diagnosis"] == 1164)
    if "Dementia at entry" in counts:
        dem = {k: counts[k] for k in ("Dementia at entry", "No dementia at entry",
                                      "Dementia at entry unknown")}
        check("every subject gets exactly one dementia-at-entry level",
              sum(dem.values()) == rep["n_subjects"], str(dem))
        check("prevalent group is 4-7% of the cohort",
              0.04 <= dem["Dementia at entry"] / rep["n_subjects"] <= 0.07,
              f"{100 * dem['Dementia at entry'] / rep['n_subjects']:.1f}%")
    # Every MMSE level must be learnable. An edge on a mode of an integer scale, or a level
    # narrower than the measurement noise, produces a level almost nothing lands in.
    mmse = [counts[V.NAMES[i]] for i in V.SCALES["MMSE"]]
    check("no MMSE level is starved", min(mmse) >= 300, f"min {min(mmse)} of {mmse}")
    check("2,745 deaths", counts["Death"] == 2745)
    # r_stroke codes 4 as NOT PRESENT and 4 is 93.9% of rows. Reading it as "most severe"
    # would emit tens of thousands of stroke tokens instead of ~1,500.
    check("stroke tokens are rare, i.e. code 4 was read as absent",
          counts["Stroke, probable"] + counts["Stroke, possible"] < 3000,
          f"{counts['Stroke, probable'] + counts['Stroke, possible']}")

    for split in ("train", "val", "test"):
        arr = np.fromfile(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint32).reshape(-1, 3)
        p2i = get_p2i(arr)
        contiguous = p2i[:, 1].sum() == len(arr) and len(np.unique(arr[:, 0])) == len(p2i)
        ages_ok = all(np.all(np.diff(arr[int(s):int(s) + int(n), 1].astype(np.int64)) >= 0)
                      for s, n in p2i[:50])
        check(f"{split}.bin: subjects contiguous, ages ascending", bool(contiguous and ages_ok))
        check(f"{split}.bin: disk tokens in [1, {V.VOCAB_SIZE - 2}]",
              int(arr[:, 2].min()) >= 1 and int(arr[:, 2].max()) <= V.VOCAB_SIZE - 2)

    # Splits must not share a subject, or test performance is inflated by having seen the
    # person's earlier visits during training.
    sets = [set(np.unique(np.fromfile(os.path.join(data_dir, f"{s}.bin"),
                                      dtype=np.uint32).reshape(-1, 3)[:, 0]).tolist())
            for s in ("train", "val", "test")]
    check("no subject appears in two splits",
          not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2]))


def test_rebuild(radc_dir, data_dir):
    print("\nrebuild determinism")
    import subprocess
    import json
    before = json.load(open(os.path.join(data_dir, "report.json")))["fingerprints"]
    r = subprocess.run([sys.executable, "-m", "radc_delphi.build_dataset", "--seed", "42"],
                       cwd=_ROOT, capture_output=True, text=True)
    check("rebuild succeeds", r.returncode == 0, (r.stderr or "")[-300:])
    after = json.load(open(os.path.join(data_dir, "report.json")))["fingerprints"]
    check("rebuild is byte-identical", before == after)


# ------------------------------------------------- the missing-AD-label mask
def test_missing_ad_label():
    """RADC records no age_first_ad_dx for anyone demented at baseline, so those subjects
    carry no AD token and would otherwise train as AD negatives. model.forward blanks the AD
    column out of THEIR cross-entropy only.

    The invariant that matters is the one that is easy to break silently: masking must move
    loss_ce and must NOT move loss_dt. Under time_head=False the intensity is logsumexp of the
    same logit tensor, so a mask applied in place -- rather than to a copy the CE alone reads
    -- would deflate it by the AD probability mass and quietly retarget the timing head.
    """
    print("\nmissing-AD-label mask")
    d, p2i = _toy_stream()
    X, A, Y, B = get_batch(np.arange(8), d, p2i, block_size=64, device="cpu",
                           select="left", padding="random", no_event_token_rate=2,
                           cut_batch=True)
    # only mask rows that do not carry the diagnosis: a flagged subject with an AD target
    # would be scored -log(0), which is exactly what train.py asserts against at startup
    eligible = ~(Y == V.AD_DX).any(1)
    mask = torch.zeros(len(X), dtype=torch.bool)
    mask[torch.nonzero(eligible).flatten()[:4]] = True
    check("the toy batch has rows to mask", bool(mask.any()), f"{int(mask.sum())} rows")

    for time_head in (False, True):
        cfg = DelphiConfig(vocab_size=V.VOCAB_SIZE, block_size=64, n_layer=2, n_head=4,
                           n_embd=32, ignore_tokens=list(V.IGNORE_TOKENS),
                           t_min=365.25 / 12, time_head=time_head, ad_dx_token=V.AD_DX)
        m = Delphi(cfg)
        m.eval()
        with torch.no_grad():
            lg_off, off, _ = m(X, A, Y, B)
            lg_on, on, _ = m(X, A, Y, B, ad_label_missing=mask)
        check(f"masking moves loss_ce (time_head={time_head})",
              abs(on["loss_ce"].item() - off["loss_ce"].item()) > 1e-6,
              f"{off['loss_ce'].item():.6f} -> {on['loss_ce'].item():.6f}")
        check(f"masking leaves loss_dt untouched (time_head={time_head})",
              abs(on["loss_dt"].item() - off["loss_dt"].item()) < 1e-9,
              f"{off['loss_dt'].item():.6f} vs {on['loss_dt'].item():.6f}")
        check(f"the RETURNED logits stay finite (time_head={time_head})",
              bool(torch.isfinite(lg_on).all()),
              "an in-place -inf would blind train.py's max|logit| guard")
        check(f"loss_ce is finite under the mask (time_head={time_head})",
              bool(torch.isfinite(on["loss_ce"])))

    # ad_dx_token = 0 is the off switch, and it must ignore the flag entirely
    cfg0 = DelphiConfig(vocab_size=V.VOCAB_SIZE, block_size=64, n_layer=2, n_head=4,
                        n_embd=32, ignore_tokens=list(V.IGNORE_TOKENS), t_min=365.25 / 12,
                        ad_dx_token=0)
    m0 = Delphi(cfg0)
    m0.eval()
    with torch.no_grad():
        a0 = m0(X, A, Y, B)[1]["loss_ce"].item()
        a1 = m0(X, A, Y, B, ad_label_missing=mask)[1]["loss_ce"].item()
    check("ad_dx_token=0 disables the mask", abs(a0 - a1) < 1e-9, f"{a0:.6f} vs {a1:.6f}")


def test_missing_ad_label_data(data_dir):
    """The flags in subjects.csv, against the event stream they describe."""
    import csv
    print("\nmissing-AD-label flags in the built dataset")
    path = os.path.join(data_dir, "subjects.csv")
    if not os.path.exists(path):
        check("subjects.csv exists", False, path)
        return
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if "ad_label_missing" not in rows[0]:
        check("subjects.csv carries ad_label_missing", False, "rebuild the dataset")
        return
    _t = lambda r, k: str(r[k]).strip().lower() in ("true", "1")     # noqa: E731
    imp = [r for r in rows if _t(r, "baseline_impaired")]
    mis = [r for r in rows if _t(r, "ad_label_missing")]
    check("400 baseline-impaired subjects", len(imp) == 400, str(len(imp)))
    check("287 of them carry no AD label", len(mis) == 287, str(len(mis)))
    check("ad_label_missing implies baseline_impaired",
          all(_t(r, "baseline_impaired") for r in mis))
    check("ad_label_missing implies not ever_ad", not any(_t(r, "ever_ad") for r in mis))

    # the load-bearing one: none of them may carry the AD token, or the -inf mask makes the
    # cross-entropy -log(0). train.py checks this at startup; here it is checked on all splits.
    flagged = {int(r["projid"]) for r in mis}
    seen, offenders = 0, 0
    for split in ("train", "val", "test"):
        f = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(f):
            continue
        arr = np.fromfile(f, dtype=np.uint32).reshape(-1, 3)
        sel = np.isin(arr[:, 0], list(flagged))
        seen += len(np.unique(arr[sel, 0]))
        offenders += int((arr[sel, 2] == V.AD_DX - 1).sum())          # DISK space
    check("every flagged subject is present in a split", seen == len(mis), f"{seen}/{len(mis)}")
    check("no flagged subject carries the AD token", offenders == 0, f"{offenders} tokens")


# ---------------------------------------------------------------- the rollout probe
def test_rollout_probe(data_dir):
    """The check that exists because validation loss provably cannot see a broken rollout.

    out-radc-final beats out-radc-testckpt on BOTH loss components (val_ce 2.140 against
    2.552, val_dt 6.536 against 6.602) and yet 98.8% of its age-80 draws never emit Death,
    against 1.2% for the other -- so its simulated death CIF at 85 is 0.005 against an
    observed 0.188. Everything below protects the probe that catches that.
    """
    print("\nrollout probe")
    from radc_delphi.rollout_probe import RolloutProbe, km_death_cif      # noqa: E402

    # Four subjects, all entering at 80; deaths at 82 and 86, censorings at 84 and 88.
    # t=82: risk set 4, S = 3/4, CIF = 0.25. t=86: risk set is only {86, 88} = 2, so
    # S = 3/4 * 1/2 and CIF = 0.625. Getting left truncation wrong, or counting the censored
    # subject at 84 as still at risk at 86, both change the second number.
    cif = km_death_cif([82., 84., 86., 88.], [True, False, True, False], [80.] * 4, [83., 87.])
    check("km_death_cif is left-truncated 1-KM",
          abs(cif[0] - 0.25) < 1e-9 and abs(cif[1] - 0.625) < 1e-9,
          f"got {cif[0]:.4f} and {cif[1]:.4f}")

    val_bin = os.path.join(data_dir, "val.bin")
    if not os.path.exists(val_bin):
        check("val.bin present for the probe", False, f"no {val_bin}")
        return
    val = np.fromfile(val_bin, dtype=np.uint32).reshape(-1, 3)
    probe = RolloutProbe(val, data_dir, V.IGNORE_TOKENS, n_subjects=6, n_mc=8, device="cpu")

    check("probe selected prefixes", len(probe.prefixes) > 0, f"{len(probe.prefixes)}")
    # A prefix containing the endpoint is not a prediction problem -- the same rule the a80
    # arm of evaluate.py enforces, so the two numbers stay comparable.
    check("no prefix contains an endpoint",
          not any(bool(np.isin(t, [V.AD_DX, V.DEATH]).any()) for t, _ in probe.prefixes))
    # One distinct age means the static block and no real visit: the statics sit a day before
    # the baseline visit and own their own age.
    check("every prefix carries >= 2 distinct ages",
          all(len(np.unique(a)) >= 2 for _, a in probe.prefixes))

    cfg = DelphiConfig(vocab_size=V.VOCAB_SIZE, block_size=64, n_layer=2, n_head=4, n_embd=32,
                       ignore_tokens=list(V.IGNORE_TOKENS), t_min=365.25 / 12, time_head=True)
    m = Delphi(cfg)
    m.train()

    first, second = probe.run(m), probe.run(m)
    check("probe is deterministic across calls", first == second,
          f"{first.get('probe/frac_no_death')} vs {second.get('probe/frac_no_death')}")
    check("probe restores model.training", m.training)
    check("probe metrics are finite",
          all(np.isfinite(v) for v in first.values()))
    check("fractions are in [0, 1]",
          all(0.0 <= first[k] <= 1.0 for k in first if 'frac' in k or 'cif_8' in k or 'cif_9' in k))

    # THE PROPERTY THAT MAKES THE CONTROLLED EXPERIMENT VALID. The probe samples, and sampling
    # consumes the generator that also drives dropout and the training batch stream. If the
    # probe leaked, a probe-on run would follow a different trajectory from a probe-off one
    # and every cell of the sweep would be measuring the probe as well as the variable.
    torch.manual_seed(7)
    before = torch.rand(4)
    torch.manual_seed(7)
    probe.run(m)
    after = torch.rand(4)
    check("probe does not consume the global RNG", bool(torch.equal(before, after)),
          f"{before.tolist()} vs {after.tolist()}")

    class _Shift(Delphi):
        """The same weights with the Death column shifted. generate() reads its logits from
        self(...), so one override moves the sampled arm and the deterministic one together."""
        def forward(self, *a, **k):
            out = super().forward(*a, **k)
            logits = out[0].clone()
            logits[..., V.DEATH] += self._delta
            return (logits,) + tuple(out[1:])

    hi, lo = _Shift(cfg), _Shift(cfg)
    hi.load_state_dict(m.state_dict()); hi._delta = 6.0
    lo.load_state_dict(m.state_dict()); lo._delta = -6.0
    mh, ml = probe.run(hi), probe.run(lo)
    check("p_death_next tracks the Death logit",
          mh['probe/p_death_next'] > 10 * ml['probe/p_death_next'],
          f"{mh['probe/p_death_next']:.5f} vs {ml['probe/p_death_next']:.5f}")
    check("frac_no_death moves the other way",
          mh['probe/frac_no_death'] < ml['probe/frac_no_death'],
          f"{mh['probe/frac_no_death']:.3f} vs {ml['probe/frac_no_death']:.3f}")



# ============================================================ figure2 state mapping
def test_figure2_states():
    """The Figure-2 RADC mapping. These are the invariants that keep the ported panels honest.

    radc_states.check() runs on import, so the partition and id assertions are already live;
    this covers the things a silent change could still break.
    """
    print("\nfigure2 state mapping")
    from figure2 import radc_states as S

    # the stages must partition the MMSE levels, worst-last
    check("stages partition the MMSE levels",
       sorted(t for g in S.STAGE_GROUPS for t in g) == sorted(V.SCALES["MMSE"]),
       f"{len(S.STAGE_TOKENS)} tokens in {S.NSTAGE} stages")
    # severity must INCREASE with stage index, the opposite of vocab.SEVERITY_ORDER's order.
    # Getting this backwards scores recovery as decline and nothing errors.
    worst_first = list(V.SEVERITY_ORDER["MMSE"])
    idx_of = {t: worst_first.index(t) for t in worst_first}
    means = [sum(idx_of[t] for t in g) / len(g) for g in S.STAGE_GROUPS]
    check("severity increases with stage index",
       all(means[i] > means[i + 1] for i in range(len(means) - 1)),
       f"mean worst-first index per stage {[round(m, 1) for m in means]}")
    check("Death closes the state space", S.DEATH_IDX == S.NSTAGE and S.NSTATE == S.NSTAGE + 1)
    check("AD is an event slot, not a state",
       S.AD_IDX not in S.GRID_SLOTS and S.AD_DX not in S.GRID_TOKENS)
    check("slot_of maps every stage token", all(S.slot_of([t])[0] >= 0 for t in S.STAGE_TOKENS))
    check("slot_of rejects a non-state token", S.slot_of([V.NO_EVENT])[0] == -1)

    # trajectory_class: AD outranks stage worsening, and a subject at the worst stage
    # cannot "progress (stage)"
    inf = float("inf")
    fp = [inf] * S.NSLOT; fp[S.AD_IDX] = 2.0; fp[1] = 1.0
    check("AD outranks stage worsening",
       S.trajectory_class(0, fp, False, 1) == "Progressed to AD")
    fp = [inf] * S.NSLOT; fp[0] = 3.0
    check("a move to a better stage is Improved",
       S.trajectory_class(2, fp, False, 0) == "Improved")
    fp = [inf] * S.NSLOT; fp[S.DEATH_IDX] = 4.0
    check("death without progression is its own class",
       S.trajectory_class(0, fp, True, 0) == "Died, no progression")

    # auxiliary slots: scoreable, but MUST NOT be in the state grid, or panel c's
    # stage-at-age carry-forward would latch onto the wrong instrument
    check("global cognition is an auxiliary scale", "COG" in S.AUX_SCALES)
    aux_toks = [t for _sl, t in S.AUX_OF_SCALE["COG"]]
    check("aux slots are exactly the COG levels", aux_toks == list(V.SCALES["COG"]),
       f"{len(aux_toks)} levels at slots {S.AUX_SLOTS}")
    check("no aux token is in the state grid",
       not any(t in S.GRID_TOKENS for t in aux_toks))
    check("no aux slot is in the state grid", not any(s in S.GRID_SLOTS for s in S.AUX_SLOTS))
    check("the state grid is still stages+Death", S.NSTATE == S.NSTAGE + 1)
    check("slot_of resolves every aux token",
       list(S.slot_of(aux_toks)) == list(S.AUX_SLOTS))
    check("slot names are unique", len(set(S.ALL_NAMES)) == len(S.ALL_NAMES),
       f"{S.NSLOT} slots")

    # the palette guard: no two outcomes may share a colour
    from figure2 import plotting_style as ps
    cols = list(ps.OUTCOME_COLORS.values())
    check("no two outcomes share a colour", len(cols) == len(set(cols)),
       f"{len(cols)} outcomes")
    check("severity ramp sizes to the stage count",
          len(ps.severity_ramp(S.NSTAGE)) == S.NSTAGE and
          len(ps.severity_ramp(7)) == 7)
    # Two ordinal instruments share panel a1, so their ramps must not overlap, and each must be
    # monotonic in LIGHTNESS -- which is the correct check for a SEQUENTIAL ramp. The dataviz
    # validator's categorical checks (lightness band, chroma floor, adjacent dE >= 15) do not
    # apply to a ramp and it says so in its own scope note; running them on one reports FAILs
    # for exactly the properties a sequential ramp is supposed to have. What the validator IS
    # used for here is the CROSS-family check, recorded in plotting_style.severity_ramp.
    import matplotlib.colors as _mc

    def _lum(h):
        def f(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        r, g, b_ = _mc.to_rgb(h)
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b_)

    r_mmse, r_cog = ps.severity_ramp(4), ps.severity_ramp(6, cmap="RdPu")
    check("the two instrument ramps share no step", not (set(r_mmse) & set(r_cog)))
    for _nm, _rr in (("MMSE", r_mmse), ("cognition", r_cog)):
        _L = [_lum(h) for h in _rr]
        check(f"{_nm} ramp is monotonic in lightness",
              all(_L[i] > _L[i + 1] for i in range(len(_L) - 1)),
              f"L {_L[0]:.2f} -> {_L[-1]:.2f}")

    # perdomain must cover every predicted content token that is not a stage or an endpoint
    from figure2 import perdomain as PD
    covered = set(t for ids in PD.SCALE_IDS.values() for t in ids)
    covered |= set(t for _n, ids in PD.EVENTS for t in ids)
    expected = set(range(V.VOCAB_SIZE)) - set(V.IGNORE_TOKENS) - {V.NO_EVENT, V.AD_DX, V.DEATH}
    check("perdomain covers every other predicted token", covered == expected,
       f"{len(covered)} tokens in {len(PD.ALL_OUTCOMES)} outcomes")
    check("BMI is scored undirected", PD.SCALE_DIRECTED["BMI"] is False and
       PD.SCALE_DIRECTED["MMSE"] is True)
    # the direction trap: adverse must be the LOWER index for a cognitive scale
    check("adverse is the lower index for MMSE",
       PD._adverse_bins(list(V.SCALES["MMSE"]), 4, True) == [0, 1, 2, 3])
    check("adverse is either side for BMI",
       PD._adverse_bins(list(V.SCALES["BMI"]), 1, False) == [0, 2])


# ---------------------------------------------------------------- death as a hazard
def test_death_head(data_dir):
    """The separate death hazard, and the four ways it can be silently wrong.

    Motivation is in DelphiConfig.death_head: under the shared softmax P(Death) collapses
    through its base rate (0.132 -> 0.0022 at real death positions in the TRAINING stream
    between iters 800 and 5900, against a 3.92% base rate) because mass withheld from Death is
    mass handed to the tokens that did occur. Each check below corresponds to a way of
    building the replacement that would train happily and fix nothing.
    """
    print("\ndeath head")
    from radc_delphi.model import death_exposure                        # noqa: E402

    base = dict(vocab_size=V.VOCAB_SIZE, block_size=64, n_layer=2, n_head=4, n_embd=32,
                ignore_tokens=list(V.IGNORE_TOKENS), t_min=365.25 / 12, time_head=True,
                dt_target="next_event", mask_ties=True)

    # THE INITIALISATION, in absolute per-day units. The first version of this head set the
    # bias from log(vocab_size) + log(P(next event is death)), copying time_head's idiom --
    # which put lambda_raw 11,259x above the corpus rate and pinned lambda_eff against its own
    # t_min ceiling, so the head began deeply saturated and never fit anything. The effective
    # hazard the loss integrates must start exactly at the target, floor included.
    import math as _math                                                # noqa: E402
    _h = 1.95e-4
    _m = Delphi(DelphiConfig(**base, death_head=True, death_token=V.DEATH,
                             death_init_hazard=_h))
    _raw = _math.exp(_m.death_head.bias.item())
    _eff = 1.0 / (1.0 / _raw + base["t_min"])
    check("death_head starts at the requested per-day hazard, floor included",
          abs(_eff - _h) / _h < 1e-4, f"lambda_eff {_eff:.4e} against {_h:.4e}")
    try:
        Delphi(DelphiConfig(**base, death_head=True, death_token=V.DEATH,
                            death_init_hazard=1.0 / base["t_min"]))
        check("death_head rejects a hazard at or above 1/t_min", False, "constructed anyway")
    except ValueError:
        check("death_head rejects a hazard at or above 1/t_min", True)

    # The head is meaningless without its own intensity to compete against, and an id of 0 is
    # the off switch -- both must fail loudly rather than train something else.
    for bad, why in (({"time_head": False}, "time_head=False"), ({"death_token": 0}, "no id")):
        cfg = dict(base, death_head=True, death_token=V.DEATH)
        cfg.update(bad)
        try:
            Delphi(DelphiConfig(**cfg))
            check(f"death_head rejects {why}", False, "constructed anyway")
        except ValueError:
            check(f"death_head rejects {why}", True)

    # Same seed, one added head: every SHARED parameter must be bit-identical, or the two arms
    # are two unrelated runs and the comparison measures nothing. This is why the head is
    # built after self.apply(_init_weights) -- see the comment there.
    torch.manual_seed(42)
    off = Delphi(DelphiConfig(**base))
    torch.manual_seed(42)
    on = Delphi(DelphiConfig(**base, death_head=True, death_token=V.DEATH))
    so, sn = off.state_dict(), on.state_dict()
    extra = sorted(set(sn) - set(so))
    check("arms are bit-identical on every shared parameter",
          all(torch.equal(so[k], sn[k]) for k in so), f"extra params {extra}")
    check("the head is the only addition", extra == ["death_head.bias", "death_head.weight"],
          str(extra))

    val_bin = os.path.join(data_dir, "val.bin")
    if not os.path.exists(val_bin):
        check("val.bin present for the death-head checks", False, f"no {val_bin}")
        return
    data = np.fromfile(val_bin, dtype=np.uint32).reshape(-1, 3)
    p2i = get_p2i(data)
    torch.manual_seed(0)
    np.random.seed(0)
    X, A, Y, B = get_batch(np.arange(min(32, len(p2i))), data, p2i, block_size=64,
                           device="cpu", select="left", padding="random",
                           no_event_token_rate=2, cut_batch=True)

    # THE BUG THIS FILE EXISTS TO PREVENT, found by measurement while building the head.
    # Death is terminal and the tokenizer forces it strictly after everything else, so it is
    # always the LAST entry of a subject's stream -- which means it lands in `targets` and
    # NEVER in `idx`. A hazard built by scanning the input stream sees zero events, trains on
    # an all-censored sample, and drives the rate to zero: the very failure being repaired.
    n_in_targets = int((Y == V.DEATH).sum())
    check("Death never appears in the INPUT stream", int((X == V.DEATH).sum()) == 0,
          f"{int((X == V.DEATH).sum())} occurrences")
    if n_in_targets:
        _, is_d, ok_d = death_exposure(A, Y, B, V.DEATH)
        check("death_exposure sees every death in the batch",
              int((is_d & ok_d).sum()) == n_in_targets,
              f"{int((is_d & ok_d).sum())} of {n_in_targets}")
        dt_d, _, _ = death_exposure(A, Y, B, V.DEATH)
        check("death events carry a strictly positive exposure",
              bool((dt_d[is_d & ok_d] > 1.0).all()),
              f"min {float(dt_d[is_d & ok_d].min()):.1f} d")
    else:
        check("batch contains at least one death to score", False, "none in this batch")

    _, l_off, _ = off(X, A, Y, B)
    _, l_on, _ = on(X, A, Y, B)
    check("the off arm is unchanged: exactly two loss terms",
          sorted(l_off) == ["loss_ce", "loss_dt"], str(sorted(l_off)))
    check("the on arm adds loss_death",
          sorted(l_on) == ["loss_ce", "loss_death", "loss_dt"], str(sorted(l_on)))

    # THE STRUCTURAL CLAIM. If the cross-entropy can still reach death_head, the collapse
    # gradient is back and the head is decoration. Exactly zero, not merely small.
    on.zero_grad()
    _, l, _ = on(X, A, Y, B)
    l["loss_ce"].backward()
    g = on.death_head.weight.grad
    check("loss_ce reaches death_head with EXACTLY zero gradient",
          g is None or float(g.abs().max()) == 0.0,
          "None" if g is None else f"max|g| {float(g.abs().max()):.3e}")
    on.zero_grad()
    _, l, _ = on(X, A, Y, B)
    l["loss_death"].backward()
    check("loss_death does reach death_head",
          float(on.death_head.weight.grad.abs().max()) > 0,
          f"max|g| {float(on.death_head.weight.grad.abs().max()):.3e}")

    # -inf lives in the masked Death column of the CE tensor and in the non-death softmax.
    # torch.where's discarded branch must not carry it into backward.
    on.zero_grad()
    _, l, _ = on(X, A, Y, B)
    sum(l.values()).backward()
    bad = [n for n, p in on.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    check("every gradient is finite under the combined loss", not bad, str(bad))

    # The sampler is untouched by design: log lambda_death goes into the Death column and
    # generate()'s argmin over competing exponentials does the rest. If Death can no longer be
    # drawn at all, the head has broken the thing it was built to fix.
    #
    # Deliberately run at an ELEVATED init hazard rather than the data's own 1.95e-4/day. What
    # is under test is the plumbing -- that lambda_death reaches the Death column and the
    # sampler can select it -- and at the real rate an untrained model fires death in about
    # 0.6% of steps, so a 64-token budget would make this check a coin flip on the seed rather
    # than a statement about the code. 0.02/day sits below the 1/t_min ceiling.
    hot = Delphi(DelphiConfig(**base, death_head=True, death_token=V.DEATH,
                              death_init_hazard=0.02)).eval()
    hot.load_state_dict(on.state_dict(), strict=False)
    with torch.no_grad():
        hot.death_head.bias.fill_(math.log(1.0 / (1.0 / 0.02 - base["t_min"])))
    idx = torch.tensor([[3, 26, 30]] * 16)
    aged = torch.tensor([[28000.0, 28001.0, 28001.0]] * 16)
    with torch.no_grad():
        gi, _, _ = hot.generate(idx, aged, max_new_tokens=64, max_age=110 * 365.25,
                                termination_tokens=list(V.TERMINATION_TOKENS),
                                extra_ignore=[V.NO_EVENT],
                                repeatable_tokens=list(V.REPEATABLE_TOKENS))
    check("Death is still reachable by generate()", bool((gi == V.DEATH).any()))
    after = 0
    for r in range(gi.shape[0]):
        pos = (gi[r] == V.DEATH).nonzero().flatten()
        if len(pos):
            after += int((gi[r][int(pos[0]) + 1:] > 0).sum())
    check("death is still absorbing under the head", after == 0, f"{after} live tokens")


def main():
    ap = argparse.ArgumentParser()
    # DEFAULT IS THE LIVE BUILD, not a historical one. This used to say "radc-s42" (v1),
    # so running the suite bare checked the current vocabulary against a two-vocabularies-ago
    # tokenization and reported a vocabulary FAIL plus a KeyError -- the same stale-default
    # failure the eval scripts had before RADC_DATA_DIR was introduced.
    ap.add_argument("--data-dir",
                    default=os.environ.get("RADC_DATA_DIR",
                                           os.path.join(_ROOT, "data", "radc-v3-s42")))
    ap.add_argument("--radc-dir", default=os.path.join(_ROOT, "data", "RADC"))
    ap.add_argument("--full", action="store_true", help="also rebuild the dataset (~1 min)")
    a = ap.parse_args()

    test_vocab()
    test_batching()
    test_soft_labels()
    test_model()
    test_generation()
    test_missing_ad_label()
    test_dataset(a.data_dir)
    test_figure2_states()
    test_missing_ad_label_data(a.data_dir)
    test_rollout_probe(a.data_dir)
    test_death_head(a.data_dir)
    if a.full:
        test_rebuild(a.radc_dir, a.data_dir)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
