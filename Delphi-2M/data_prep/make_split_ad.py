"""
make_split_ad.py -- split out_ad/nacc_all.bin into train/val/test BY PATIENT.

Same logic as Delphi-AD2/make_split.py (70/10/20, by-patient, fixed seed, each seed
its own folder so splits never clobber and a model is always eval'd on its own split).
Only the source/output paths differ (AD bin instead of the Delphi-AD2 one).

Splitting by PATIENT (not by row) is what keeps the evaluation honest: no subject ever
appears in two splits, so test performance cannot be inflated by having seen that person's
earlier visits during training.

  python make_split_ad.py --seed 42     # -> out_ad/nacc-dedup-s42/{train,val,test}.bin

Normally invoked via make_dataset.py, which also creates the data/ symlinks.

Removed 2026-07-31: the --cohort / --fu-thresh / --no-split options, which selected the
"follow-up >10y OR >=1 NACCUDSD transition" subset for the descriptive EDA figures. Both the
EDA scripts and cohort_stats_ad.py (which produced the stats CSV they needed) are gone. To
restore that rule you would need to re-add cohort_stats_ad.py as well as these options.
"""
import numpy as np, os, argparse, shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Delphi-2M/ (package root)
RATIO = (0.70, 0.10, 0.20)   # train, val, test

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
# --dir / --prefix let the SAME by-patient split logic serve a second tokenised source
# (out_radc/radc_all.bin, from data_prep/tokenize_radc.py) without a forked copy of this
# file. Defaults reproduce the original NACC behaviour byte-for-byte.
ap.add_argument("--dir", default="out_ad",
                help="subdir of Delphi-2M/ holding <prefix>_all.bin + labels.csv")
ap.add_argument("--prefix", default="nacc",
                help="dataset prefix: reads <dir>/<prefix>_all.bin, writes <dir>/<prefix>-dedup-s<seed>/")
args = ap.parse_args()

SRC = os.path.join(HERE, args.dir, f"{args.prefix}_all.bin")
LBL = os.path.join(HERE, args.dir, "labels.csv")

d = np.fromfile(SRC, dtype=np.uint32).reshape(-1, 3)
# "dedup" = the keep-transitions tokenisation this pipeline produces. The name is explicit so a
# differently-tokenised build can never silently overwrite it.
OUT = os.path.join(HERE, args.dir, f"{args.prefix}-dedup-s{args.seed}")

os.makedirs(OUT, exist_ok=True)
pids = np.unique(d[:, 0])
rng  = np.random.default_rng(args.seed)
shuf = pids.copy(); rng.shuffle(shuf)          # shuffle PATIENTS, not rows
n = len(shuf); n_tr = int(RATIO[0] * n); n_va = int(RATIO[1] * n)
sets = {"train": set(shuf[:n_tr].tolist()),
        "val":   set(shuf[n_tr:n_tr + n_va].tolist()),
        "test":  set(shuf[n_tr + n_va:].tolist())}

print(f"seed {args.seed} | patients {n} -> train {len(sets['train'])} val {len(sets['val'])} test {len(sets['test'])}")
for name, S in sets.items():
    mask = np.isin(d[:, 0], np.fromiter(S, dtype=np.uint32))   # keeps row order -> patients stay contiguous
    part = d[mask]
    part.tofile(os.path.join(OUT, f"{name}.bin"))
    print(f"  {name}.bin: {part.shape[0]:>9} events, {len(np.unique(part[:,0])):>6} patients")

if os.path.exists(LBL):
    shutil.copy(LBL, os.path.join(OUT, "labels.csv"))          # keep vocab alongside the split

assert len(sets['train'] | sets['val'] | sets['test']) == n, "patient leakage / loss"
assert not (sets['train'] & sets['val']) and not (sets['train'] & sets['test']) and not (sets['val'] & sets['test'])
print(f"OK: by-patient, no overlap -> {OUT}")
