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
    check("size is 50", V.VOCAB_SIZE == 50, f"got {V.VOCAB_SIZE}")
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
    check("21 repeatable ids", len(V.REPEATABLE_TOKENS) == 21, str(len(V.REPEATABLE_TOKENS)))
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
    check("74,900 events over 4,428 subjects",
          rep["n_events"] == 74900 and rep["n_subjects"] == 4428,
          f"{rep['n_events']} / {rep['n_subjects']}")
    check("vocabulary is 50", rep["vocab_size"] == 50)
    check("ad_rx is excluded from the production build", rep["emit_ad_rx"] is False)
    check("block_size 64 truncates nobody", rep["min_block_size"] <= 64,
          f"needs >= {rep['min_block_size']}")
    # Deterministic given the inputs: the same three files must give the same bytes on any
    # machine. This was verified byte-identical between a Mac and the RIS cluster.
    check("train.bin fingerprint unchanged",
          rep["fingerprints"]["train.bin"] == "8992a31abaaf",
          rep["fingerprints"]["train.bin"])

    counts = rep["token_counts"]
    # 73.2% of the ROSMAP cohort is female. If msex were read backwards this flips, and
    # nothing else in the pipeline would notice.
    f, mm = counts["Sex: female"], counts["Sex: male"]
    check("cohort is ~73% female", 0.72 < f / (f + mm) < 0.74, f"{100 * f / (f + mm):.1f}%")
    check("1,164 AD diagnoses", counts["Alzheimer's dementia diagnosis"] == 1164)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(_ROOT, "data", "radc-s42"))
    ap.add_argument("--radc-dir", default=os.path.join(_ROOT, "data", "RADC"))
    ap.add_argument("--full", action="store_true", help="also rebuild the dataset (~1 min)")
    a = ap.parse_args()

    test_vocab()
    test_batching()
    test_model()
    test_generation()
    test_dataset(a.data_dir)
    if a.full:
        test_rebuild(a.radc_dir, a.data_dir)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
