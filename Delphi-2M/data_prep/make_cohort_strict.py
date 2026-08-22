"""
make_cohort_strict.py -- build the STRICT long-trajectory cohort from an existing dedup split.

Keeps a patient IFF all three hold:
    true follow-up  >  --fu-years        (default 8)
    distinct visits >= --min-visits      (default 6)
    NACCUDSD transitions >= --min-trans  (default 2)

WHY THE RAW CSV: follow-up and visit counts are taken from the raw NACC export, not from the
.bin. Keep-transitions dedup collapses stable runs, so the .bin systematically UNDER-measures
both quantities -- and follow-up is exactly what we threshold on. Transition counts do come
from the .bin, where they are exact by construction (tokens = 1 + transitions).

WHY IT FILTERS AN EXISTING SPLIT rather than re-splitting: the strict test set is then a
SUBSET of the original test set, so a model trained on this cohort and the general model can
be scored on the very same patients. Re-splitting would destroy that comparison.

  python data_prep/make_cohort_strict.py --csv /path/to/investigator_nacc72.csv
      -> out_ad/nacc-strict-s42/{train,val,test}.bin  (+ labels.csv, + data/ symlink)

Run data_prep/make_dataset.py first: this reads the dedup split it produces.
"""
import os, sys, argparse, shutil
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Delphi-2M/ (package root)
OUT_AD = os.path.join(HERE, "out_ad")
UDSD_DISK = (105, 106, 107, 108)          # model 106-109 minus the +1 disk shift
SPLITS = ("train", "val", "test")


def visit_stats(csv_path):
    """Per-NACCID (n_visits, followup_y) from the raw export.

    Date reconstruction is byte-identical to tokenize_nacc_ad.py: a missing VISITDAY falls back
    to mid-month rather than dropping the visit, and sentinels are clipped into [1, 28].
    """
    cols = ["NACCID", "VISITYR", "VISITMO", "VISITDAY"]
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    df["visit"] = pd.to_datetime(dict(year=num("VISITYR"), month=num("VISITMO"),
                                      day=num("VISITDAY").fillna(15).clip(1, 28)), errors="coerce")
    df = df.dropna(subset=["visit"]).drop_duplicates(["NACCID", "visit"])
    g = df.groupby("NACCID")["visit"]
    return pd.DataFrame({"n_visits": g.size(),
                         "followup_y": (g.max() - g.min()).dt.days / 365.25}).reset_index()


def main():
    ap = argparse.ArgumentParser(description="Build the strict long-trajectory cohort.")
    ap.add_argument("--csv", required=True, help="raw NACC investigator CSV (for true visit counts)")
    ap.add_argument("--source", default="nacc-dedup-s42", help="split under out_ad/ to filter")
    ap.add_argument("--tag", default="nacc-strict-s42", help="output split name")
    ap.add_argument("--fu-years", type=float, default=8.0)
    ap.add_argument("--min-visits", type=int, default=6)
    ap.add_argument("--min-trans", type=int, default=2)
    ap.add_argument("--no-link", action="store_true")
    a = ap.parse_args()

    src = os.path.join(OUT_AD, a.source)
    pidmap = os.path.join(OUT_AD, "nacc_pidmap.csv")
    for p in (src, pidmap, a.csv):
        if not os.path.exists(p):
            sys.exit(f"[strict] missing {p}\n  run data_prep/make_dataset.py first")

    print(f"[strict] rule: follow-up > {a.fu_years:g} y  AND  visits >= {a.min_visits}  "
          f"AND  transitions >= {a.min_trans}")
    raw = visit_stats(a.csv)
    pm = pd.read_csv(pidmap)
    idcol = "NACCID" if "NACCID" in pm.columns else pm.columns[1]
    raw = pm.merge(raw, left_on=idcol, right_on="NACCID", how="inner")[["pidi", "n_visits", "followup_y"]]

    out = os.path.join(OUT_AD, a.tag)
    os.makedirs(out, exist_ok=True)
    kept_total = 0
    for s in SPLITS:
        d = np.fromfile(os.path.join(src, f"{s}.bin"), dtype=np.uint32).reshape(-1, 3)
        # transitions are exact in the .bin: keep-transitions => tokens = 1 + transitions
        pids, counts = np.unique(d[np.isin(d[:, 2], UDSD_DISK), 0], return_counts=True)
        tr = pd.DataFrame({"pidi": pids, "n_trans": counts - 1})
        st = raw.merge(tr, on="pidi", how="inner")
        keep = st[(st.followup_y > a.fu_years) &
                  (st.n_visits >= a.min_visits) &
                  (st.n_trans >= a.min_trans)]["pidi"].to_numpy(np.uint32)
        # row order is preserved, so patients stay contiguous and age-sorted
        part = d[np.isin(d[:, 0], keep)]
        part.tofile(os.path.join(out, f"{s}.bin"))
        before = len(np.unique(d[:, 0]))
        print(f"  {s:<5} {before:>6,} -> {len(keep):>6,} patients   {part.shape[0]:>8,} events")
        kept_total += len(keep)

    lbl = os.path.join(OUT_AD, "labels.csv")
    if os.path.exists(lbl):
        shutil.copy(lbl, os.path.join(out, "labels.csv"))

    if not a.no_link:
        os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
        link = os.path.join(HERE, "data", a.tag)
        rel = os.path.relpath(out, os.path.join(HERE, "data"))
        if os.path.islink(link):
            os.remove(link)
        elif os.path.exists(link):
            sys.exit(f"[strict] refusing to replace a real file at {link}")
        os.symlink(rel, link)
        print(f"[strict] linked data/{a.tag} -> {rel}")

    print(f"[strict] DONE -> {out}/   {kept_total:,} patients total")
    print("  next:")
    print("    python training/train.py config/train_delphi2m_strict.py --device=cuda")


if __name__ == "__main__":
    main()
