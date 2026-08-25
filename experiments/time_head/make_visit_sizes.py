"""Write the empirical tokens-per-visit distribution (NON-FIRST visits) for visit-batched
generation. Non-first because the first visit of a stream carries every baseline scale at once
(p95 = 16), while generation always continues from a seed prefix."""
import numpy as np, os, sys
sys.path.insert(0, ".")
from delphi.utils import get_p2i, patient_stream
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
TIE = 30.0
d = np.memmap(sys.argv[1], dtype=np.uint32, mode="r").reshape(-1, 3)
p2i = get_p2i(d)
out = []
for k in range(len(p2i)):
    ages, toks = patient_stream(d, int(p2i[k, 0]), int(p2i[k, 1]))
    a = ages[ages > 0]
    if a.size == 0:
        continue
    a = np.sort(a)
    va, cnt = [a[0]], [1]
    for x in a[1:]:
        if x - va[-1] <= TIE: cnt[-1] += 1
        else: va.append(x); cnt.append(1)
    if len(cnt) > 1:
        out.extend(cnt[1:])
v = np.asarray(out, dtype=np.int64)
np.save(sys.argv[2], v)
print(f"{len(v):,} non-first visits: median {np.median(v):.0f}, mean {v.mean():.2f}, "
      f"p95 {np.percentile(v,95):.0f} -> {sys.argv[2]}")
