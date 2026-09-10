"""
build_dataset.py -- one command: the three raw RADC files -> data/radc-s<seed>/.

    python -m radc_delphi.build_dataset                 # seed 42
    python -m radc_delphi.build_dataset --seed 7
    python -m radc_delphi.build_dataset --emit-ad-rx    # the leakage ablation arm

Writes, under data/radc-s<seed>/:
    train.bin val.bin test.bin   uint32 (pid, age_days, DISK token) triples, subjects
                                 contiguous and ages ascending -- the layout train.py mmaps
    labels.csv                   the vocabulary, one row per MODEL id, in id order
    subjects.csv                 per-subject metadata + split assignment, for evaluation
    visit_sizes.npy              the observed tokens-per-visit distribution
    report.json                  counts and fingerprints, so a build is reproducible/auditable

DISK vs MODEL space: the .bin stores `model_id - 1`, because batching.get_batch adds the +1
back (it reserves 0 for padding). labels.csv is written in MODEL space. Getting this backwards
shifts every clinical label by one and the model still trains happily, so the offset is applied
in exactly this one place and asserted on the way out.

WHY visit_sizes.npy IS BUILT HERE AND NOT MEASURED AD HOC. Generation samples one token per
step, each with a strictly positive waiting time, but a real RADC visit records several tokens
at ONE age. Without the observed distribution to draw from, one real visit costs several
simulated steps and trajectories run far too slowly. That distribution is a CALIBRATION
CONSTANT OF THE TOKENIZATION -- change a bin edge and it changes -- so it is rebuilt with the
.bin every time rather than carried along from a previous build.
"""
import os
import sys
import json
import hashlib
import argparse

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

from radc_delphi import vocab as V                       # noqa: E402
from radc_delphi.tokenizer import tokenize, DAYS_PER_YEAR  # noqa: E402
from radc_delphi.splits import build_strata, split_by_subject, write_splits, SPLIT_NAMES  # noqa: E402

# One clinic visit can land its records a few days apart, and RADC's nominal grid is annual, so
# events are grouped into a visit by a tie window rather than by exact equal age. 30 days is
# comfortably inside the annual spacing and comfortably outside any within-visit spread.
TIE_DAYS = 30.0


def visit_size_distribution(events):
    """Tokens per visit, for the SECOND and later visits of each subject.

    The first visit is excluded because it carries every scale's baseline value at once and is
    therefore systematically larger than a follow-up visit -- and generation always continues
    from a seed prefix, so it is a follow-up visit that is being simulated. The statics, which
    sit a day before baseline, are excluded for the same reason.
    """
    d = events[events[:, 1] > 0]
    order = np.lexsort((d[:, 1], d[:, 0]))
    pid, age = d[order, 0], d[order, 1].astype(np.float64)
    sizes, n_visits, n_subj = [], 0, 0
    i = 0
    while i < len(pid):
        j = i
        while j < len(pid) and pid[j] == pid[i]:
            j += 1
        n_subj += 1
        starts, counts = [age[i]], [1]
        for x in age[i + 1:j]:
            if x - starts[-1] <= TIE_DAYS:
                counts[-1] += 1
            else:
                starts.append(x)
                counts.append(1)
        n_visits += len(counts)
        sizes.extend(counts[1:])                 # second and later only
        i = j
    return np.asarray(sizes, dtype=np.int64), n_visits, n_subj


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the RADC train/val/test splits.")
    ap.add_argument("--radc-dir", default=os.path.join(_HERE, "data", "RADC"),
                    help="directory holding the three raw RADC files")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="output dir (default data/radc-s<seed>)")
    ap.add_argument("--emit-ad-rx", action="store_true",
                    help="include the ad_rx tokens (a MEASURED leak of the AD outcome -- the "
                         "ablation arm only; AD-onset metrics from this build are not prediction)")
    a = ap.parse_args(argv)

    out_dir = a.out or os.path.join(_HERE, "data", f"radc-s{a.seed}")
    missing = [f for f in ("cross-sectional-data-gk.xlsx", "longitudinal_data_gk.xlsx",
                           "ROSMAP_clinical.csv")
               if not os.path.exists(os.path.join(a.radc_dir, f))]
    if missing:
        sys.exit(f"[build] missing from {a.radc_dir}:\n    " + "\n    ".join(missing))

    print(f"[build] tokenizing {a.radc_dir}")
    events, subjects, report = tokenize(a.radc_dir, emit_ad_rx=a.emit_ad_rx)

    print(f"\n[build] stratified by-subject split, seed {a.seed}")
    strata = build_strata(subjects)
    print(f"  {strata.nunique()} strata over {len(subjects):,} subjects "
          f"(study x follow-up bucket x AD status, rare cells merged upward)")
    assignment = split_by_subject(subjects.index.to_numpy(), strata.to_numpy(), seed=a.seed)

    # DISK space, and this is the ONLY place the -1 is applied
    disk = events.copy()
    disk[:, 2] -= 1
    assert disk[:, 2].min() >= 1, "a reserved id leaked into the event stream"
    assert disk[:, 2].max() == V.VOCAB_SIZE - 2, \
        f"disk token max {disk[:, 2].max()} != vocab_size-2 ({V.VOCAB_SIZE - 2})"
    disk = disk.astype(np.uint32)

    stats = write_splits(disk, assignment, out_dir, V.labels())

    # per-subject metadata, with the split, for the evaluation stack
    split_of = {}
    for name in SPLIT_NAMES:
        for p in assignment[name]:
            split_of[p] = name
    subjects = subjects.copy()
    subjects["split"] = [split_of[p] for p in subjects.index]
    subjects["stratum"] = strata.to_numpy()
    subjects.to_csv(os.path.join(out_dir, "subjects.csv"))

    sizes, n_visits, n_subj = visit_size_distribution(disk)
    np.save(os.path.join(out_dir, "visit_sizes.npy"), sizes)
    print(f"\n  visits: {n_visits:,} ({TIE_DAYS:.0f}-day tie window) over {n_subj:,} subjects")
    print(f"  tokens/visit (2nd+): mean {sizes.mean():.2f} median {np.median(sizes):.0f} "
          f"p95 {np.percentile(sizes, 95):.0f} max {sizes.max()}")

    # block_size must not truncate anybody: training uses select='left', so truncation drops a
    # subject's LATEST events -- which are exactly the conversions and the deaths.
    per_subj = np.bincount(disk[:, 0].astype(np.int64))
    longest = int(per_subj.max())
    n_pad = int(100 / 5)                    # get_batch's random no-event budget at rate 5
    print(f"  longest subject: {longest} events (+ up to {n_pad} no-event markers) "
          f"-> block_size must be >= {longest + n_pad}")

    # split sanity: the stratified split should reproduce the cohort's rates in every part
    print("\n  --- split balance (should be near-identical across the three) ---")
    for name in SPLIT_NAMES:
        s = subjects[subjects.split == name]
        print(f"  {name:<5} n={len(s):>5,}  AD {100 * s.ever_ad.mean():5.1f}%  "
              f"died {100 * (s.died == 1).mean():5.1f}%  "
              f"ROS/MAP/LATC {100 * (s.study == 'ROS').mean():4.1f}/"
              f"{100 * (s.study == 'MAP').mean():4.1f}/{100 * (s.study == 'LATC').mean():4.1f}  "
              f"median fu {s.followup_y.median():.0f}y")

    def _md5(p):
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()[:12]

    rep = dict(
        seed=a.seed, radc_dir=os.path.abspath(a.radc_dir), out_dir=os.path.abspath(out_dir),
        vocab_size=V.VOCAB_SIZE, emit_ad_rx=bool(a.emit_ad_rx),
        n_events=int(report["n_events"]), n_subjects=int(report["n_subjects"]),
        events_per_subject=round(report["events_per_subject"], 3),
        longest_subject=longest, min_block_size=longest + n_pad,
        token_counts=report["counts"],
        splits={k: dict(events=v[0], subjects=v[1]) for k, v in stats.items()},
        fingerprints={f: _md5(os.path.join(out_dir, f))
                      for f in ("train.bin", "val.bin", "test.bin", "labels.csv")},
    )
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)

    print(f"\n[build] DONE -> {out_dir}/")
    print(f"  next:  python train.py configs/radc_base.py --device=cuda")
    return rep


if __name__ == "__main__":
    main()
