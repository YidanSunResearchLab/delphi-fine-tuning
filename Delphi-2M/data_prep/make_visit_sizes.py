"""
make_visit_sizes.py -- the tokens-per-visit distribution that `model.generate(visit_sizes=...)`
samples from.

WHY THIS FILE EXISTS. Delphi samples one token per step, each with a strictly positive dt.
Real data records several tokens at ONE age per visit, so a single real visit costs several
simulated steps and trajectories run too slowly -- measured at 2.5x (9.27 y to Dementia
against a real 3.67 y; experiments/time_head/gen_steps_probe.py). `visit_sizes` fixes that:
generation emits a WHOLE visit per step, with k drawn from the observed distribution.

That distribution is a CALIBRATION CONSTANT OF THE TOKENIZATION. Split a sum-score into its
items and tokens-per-visit goes up, so the constant must be regenerated with the .bin or the
sampler silently reintroduces exactly the bug it was added to fix. It had no producer script
before -- it was measured ad hoc -- which is why this exists.

DEFINITION -- taken verbatim from experiments/time_head/make_visit_sizes.py, which is where
this measurement was first done ad hoc. Events are grouped into a visit by a 30-DAY TIE
WINDOW, not by exact equal age: one clinical visit can land its records a few days apart, and
grouping on exact age would split it into several tiny "visits" and pull the mean down. Only
each patient's SECOND and later visits count -- the first carries every scale's baseline value
at once, while generation always continues from a seed prefix. Age-0 statics are not a visit.

This file exists so the constant is rebuilt with the .bin by make_dataset.py; the copy under
experiments/time_head/ is the original one-off and is left alone so that experiment still
reproduces byte-for-byte.

  python data_prep/make_visit_sizes.py                  # -> out_ad/visit_sizes.npy
  DELPHI_VISIT_SIZES=out_ad/visit_sizes.npy python ...  # how predict_adapter picks it up
"""
import os, argparse, numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Delphi-2M/

ap = argparse.ArgumentParser()
ap.add_argument("--bin", default=os.path.join(HERE, "out_ad", "nacc_all.bin"))
ap.add_argument("--out", default=os.path.join(HERE, "out_ad", "visit_sizes.npy"))
a = ap.parse_args()

TIE_DAYS = 30.0            # same window as experiments/time_head/make_visit_sizes.py

d = np.fromfile(a.bin, dtype=np.uint32).reshape(-1, 3)
d = d[d[:, 1] > 0]                                   # drop age-0 statics: not a visit
order = np.lexsort((d[:, 1], d[:, 0]))
pid, age = d[order, 0], d[order, 1].astype(np.float64)

k, n_visits, n_pat = [], 0, 0
i = 0
while i < len(pid):
    j = i
    while j < len(pid) and pid[j] == pid[i]:
        j += 1
    n_pat += 1
    starts, counts = [age[i]], [1]                   # collapse ages within TIE_DAYS
    for x in age[i + 1:j]:
        if x - starts[-1] <= TIE_DAYS:
            counts[-1] += 1
        else:
            starts.append(x); counts.append(1)
    n_visits += len(counts)
    k.extend(counts[1:])                             # second and later visits only
    i = j
k = np.asarray(k, dtype=np.int64)

np.save(a.out, k)
print(f"{os.path.basename(a.bin)}: {n_visits:,} visits ({TIE_DAYS:.0f}-day tie window), {n_pat:,} patients")
print(f"  second-and-later visits: {len(k):,}")
print(f"  tokens/visit: mean {k.mean():.2f}  median {np.median(k):.0f}  "
      f"p95 {np.percentile(k, 95):.0f}  max {k.max()}")
print(f"  -> {a.out}")
