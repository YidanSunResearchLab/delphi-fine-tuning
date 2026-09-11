"""
splits.py -- partition the tokenized RADC event stream into train/val/test BY SUBJECT.

Two properties, both load-bearing:

  1. NO SUBJECT CROSSES A SPLIT. Splitting by row would put a subject's early visits in train
     and their later ones in test, so test performance would be inflated by having already seen
     that person. Everything here operates on subject ids and then selects whole subjects.

  2. THE SPLIT IS STRATIFIED, not a plain shuffle. ROS, MAP and LATC are not interchangeable
     cohorts -- measured on these files they differ by 8.1 years of median education, 9.4 years
     of baseline age, and 63 percentage points of mortality, and their measurement protocols
     differ by up to 91.5 pp of coverage on individual variables. A random split of 4,428
     subjects leaves those proportions to chance, and LATC is small enough that the sampling
     noise is material. Follow-up length and AD status are stratified for the same reason: the
     evaluable population is the subjects with enough history to predict from, and an unlucky
     draw can move the test set's conversion rate by several points, which then looks like a
     model effect.

     Stratifying on the outcome is deliberate and is not leakage: it equalises the CLASS BALANCE
     across splits, it is applied before any model sees anything, and no subject-level label
     crosses the boundary.

Usage (normally via `python -m radc_delphi.build_dataset`):

    from radc_delphi.splits import split_by_subject
    parts = split_by_subject(events, strata, seed=42)
"""
import numpy as np
import pandas as pd

RATIO = (0.70, 0.10, 0.20)          # train, val, test
SPLIT_NAMES = ("train", "val", "test")


def followup_bucket(years):
    """Coarse follow-up strata. Edges chosen so no bucket is too small to stratify on:
    a single-visit subject can never be predicted from, 1-4 y is the bulk of LATC, and >=10 y
    is the long-trajectory tail that most of the evaluation actually rests on."""
    if years < 1:
        return "0"
    if years < 5:
        return "1-4"
    if years < 10:
        return "5-9"
    return "10+"


def build_strata(subjects):
    """subjects: DataFrame indexed by integer pid with columns study, followup_y, ever_ad.

    Returns a Series of stratum labels. Rare strata are merged upward (study alone, then a
    single pooled stratum) so that every stratum has at least `MIN_PER_STRATUM` members --
    a stratum of 2 cannot be split 70/10/20 in any meaningful way, and leaving it alone
    silently puts both members in train.
    """
    MIN_PER_STRATUM = 10
    full = (subjects["study"].astype(str) + "|" +
            subjects["followup_y"].map(followup_bucket) + "|" +
            np.where(subjects["ever_ad"].to_numpy(dtype=bool), "AD", "noAD"))
    counts = full.value_counts()
    small = set(counts[counts < MIN_PER_STRATUM].index)
    if small:
        # first fallback: drop the follow-up bucket, keep study x outcome
        fallback = (subjects["study"].astype(str) + "|" +
                    np.where(subjects["ever_ad"].to_numpy(dtype=bool), "AD", "noAD"))
        full = full.where(~full.isin(small), fallback)
        counts = full.value_counts()
        still_small = set(counts[counts < MIN_PER_STRATUM].index)
        if still_small:
            full = full.where(~full.isin(still_small), "pooled")
    return full


def split_by_subject(pids, strata, seed=42, ratio=RATIO):
    """Assign each subject id to exactly one of train/val/test, stratum by stratum.

    Within a stratum the subjects are shuffled with a seeded generator and cut at the ratio
    boundaries, so each stratum contributes its own 70/10/20 and the overall proportions hold
    inside every study x follow-up x outcome cell rather than only on average.

    Returns dict[str, np.ndarray of pids].
    """
    pids = np.asarray(pids)
    strata = np.asarray(strata)
    assert len(pids) == len(strata)
    rng = np.random.default_rng(seed)
    out = {k: [] for k in SPLIT_NAMES}
    for s in sorted(set(strata.tolist())):
        members = pids[strata == s]
        members = members[rng.permutation(len(members))]
        n = len(members)
        n_tr = int(round(ratio[0] * n))
        n_va = int(round(ratio[1] * n))
        # A stratum of 1-2 cannot honour the ratio; give the first to train, the next to test,
        # so tiny strata do not silently become train-only.
        if n <= 2:
            out["train"].append(members[:1])
            if n == 2:
                out["test"].append(members[1:2])
            continue
        n_tr = max(1, min(n_tr, n - 2))          # leave at least one each for val and test
        n_va = max(1, min(n_va, n - n_tr - 1))
        out["train"].append(members[:n_tr])
        out["val"].append(members[n_tr:n_tr + n_va])
        out["test"].append(members[n_tr + n_va:])
    res = {k: np.sort(np.concatenate(v)) if v else np.array([], dtype=pids.dtype)
           for k, v in out.items()}

    allp = np.concatenate([res[k] for k in SPLIT_NAMES])
    assert len(allp) == len(pids), f"subject lost or duplicated: {len(allp)} vs {len(pids)}"
    assert len(np.unique(allp)) == len(pids), "a subject landed in more than one split"
    return res


def write_splits(events, assignment, out_dir, labels, values=None, verbose=True):
    """Write {train,val,test}.bin + labels.csv under out_dir.

    `events` is the uint32 (pid, age_days, disk_token) array, already sorted so that subjects
    are contiguous and ages ascending; boolean row selection preserves that order, so the .bin
    files inherit both properties and get_p2i works on them unchanged.

    `values` is the parallel float32 array of RAW scale readings (NaN elsewhere), written to
    <split>_values.bin. It is a SEPARATE file rather than a fourth column so that every
    existing reader of the uint32 (N, 3) layout keeps working untouched, and so a build made
    before soft labels existed still loads -- get_batch treats a missing values file as
    "hard tokens only".
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for name in SPLIT_NAMES:
        keep = np.isin(events[:, 0], assignment[name])
        part = events[keep]
        part.tofile(os.path.join(out_dir, f"{name}.bin"))
        if values is not None:
            values[keep].astype(np.float32).tofile(
                os.path.join(out_dir, f"{name}_values.bin"))
        stats[name] = (part.shape[0], len(np.unique(part[:, 0])) if len(part) else 0)
        if verbose:
            print(f"  {name:<5} {stats[name][1]:>6,} subjects  {stats[name][0]:>9,} events")
    pd.DataFrame({"event_name": labels}).to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    return stats
