"""
make_dataset.py -- one command: raw NACC CSV  ->  train/val/test splits.

Thin wrapper over the two existing steps (kept separate so each is still runnable alone):
  1) tokenize_nacc_ad.py : raw NACC CSV -> out_ad/nacc_all.bin
        Dedup is applied HERE and baked into the .bin permanently:
        keep-first for non-cognitive tokens, KEEP-TRANSITIONS (per scale) for ALL
        five cognitive scales -- NACCUDSD + CDR-SB + MoCA + FAST + NPI-Q
        (see tokenize_nacc_ad.py). Nothing downstream re-derives it.
  2) make_split_ad.py    : nacc_all.bin -> out_ad/nacc-dedup-s<seed>/{train,val,test}.bin
        By-patient 70/10/20 (no patient crosses splits).

Reads the CSV from --csv. With no --csv it auto-finds investigator_nacc72.csv next to
this script (Delphi-2M/), then <repo>/data/. You do NOT need to place/symlink it inside
this dir yourself -- this script wires that up.

It also symlinks the finished split to data/nacc-dedup-s<seed>, which is the name
config/train_*.py and figure2/figure2_core.py read.

Usage:
  python make_dataset.py --csv /path/to/investigator_nacc72.csv       # seed 42
  python make_dataset.py --csv /path/to/nacc.csv --seed 1234
  python make_dataset.py --csv ... --no-link                          # don't touch ./data/
"""
import os, sys, argparse, subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Delphi-2M/ (package root)
SCRIPTS = os.path.dirname(os.path.abspath(__file__))   # data_prep/
REPO = os.path.dirname(HERE)
TOKENIZER_CSV = os.path.join(HERE, "investigator_nacc72.csv")   # the fixed path tokenize_nacc_ad.py reads


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def point_symlink(link, target):
    """Point symlink `link` at `target`. NEVER deletes a real (non-symlink) file at `link`."""
    if os.path.realpath(link) == os.path.realpath(target):
        return                               # already resolves to the target (incl. link IS the file)
    if os.path.islink(link):
        os.remove(link)                      # safe: only ever removing a symlink
    elif os.path.exists(link):
        # a REAL file/dir sits here (e.g. the user copied their CSV to this canonical path).
        # Deleting it would be data loss -- refuse and tell them how to proceed.
        sys.exit(f"[make_dataset] refusing to delete real file at:\n    {link}\n"
                 f"  It is not a symlink. Either (a) run WITHOUT --csv to use that file in place,\n"
                 f"  or (b) move/rename it and re-run with --csv pointing at your CSV.")
    os.symlink(target, link)


def main():
    ap = argparse.ArgumentParser(description="Build train/val/test splits from a raw NACC CSV.")
    ap.add_argument("--csv", default=None,
                    help="raw NACC investigator CSV. Default: look for investigator_nacc72.csv "
                         "next to this script (Delphi-2M/), then <repo>/data/investigator_nacc72.csv.")
    ap.add_argument("--seed", type=int, default=42, help="split seed -> out_ad/nacc-dedup-s<seed>/")
    ap.add_argument("--no-link", action="store_true",
                    help="do NOT symlink the split under ./data/ (default: do link it)")
    args = ap.parse_args()

    if args.csv is not None:
        csv = os.path.abspath(args.csv)
    else:
        # No --csv given: try the two conventional locations, in order.
        candidates = [os.path.join(HERE, "investigator_nacc72.csv"),         # next to the tokenizer
                      os.path.join(REPO, "data", "investigator_nacc72.csv")]  # <repo>/data/
        csv = next((os.path.abspath(c) for c in candidates if os.path.exists(c)), None)
        if csv is None:
            sys.exit("[make_dataset] no CSV found. Looked for investigator_nacc72.csv in:\n"
                     f"    {candidates[0]}\n    {candidates[1]}\n"
                     "  pass --csv /path/to/investigator_nacc72.csv")

    if not os.path.exists(csv):
        sys.exit(f"[make_dataset] CSV not found: {csv}\n  pass --csv /path/to/investigator_nacc72.csv")

    # 1) point the tokenizer's fixed input path at the requested CSV (no manual symlink needed)
    point_symlink(TOKENIZER_CSV, csv)
    print(f"[make_dataset] input CSV : {csv}")

    # 2) tokenize (dedup + keep-transitions happen inside here) -> out_ad/nacc_all.bin
    run([sys.executable, os.path.join(SCRIPTS, "tokenize_nacc_ad.py")])

    # 3) by-patient 70/10/20 split -> out_ad/nacc-dedup-s<seed>/
    run([sys.executable, os.path.join(SCRIPTS, "make_split_ad.py"), "--seed", str(args.seed)])

    split_dir = os.path.join(HERE, "out_ad", f"nacc-dedup-s{args.seed}")

    # 4) make it usable by train.py / the eval stack (they read data/<dataset>/)
    if not args.no_link:
        os.makedirs(os.path.join(HERE, "data"), exist_ok=True)   # data/ is not committed; create on demand
        rel = os.path.relpath(split_dir, os.path.join(HERE, "data"))
        name = f"nacc-dedup-s{args.seed}"
        point_symlink(os.path.join(HERE, "data", name), rel)
        print(f"[make_dataset] linked data/{name} -> {rel}")

    print(f"\n[make_dataset] DONE -> {split_dir}/  (train.bin, val.bin, test.bin, labels.csv)")
    if not args.no_link:
        print("  next (run from Delphi-2M/):")
        print("    python training/train.py config/train_delphi2m_mask_dedup.py --device=cuda")
        print("    ./figure2/run_figure2.sh")
        print("  or on SLURM, from the repo root (see README sec.1 for AD_BASE / AD_CONDA):")
        print("    sbatch Delphi-2M/slurm/train_delphi2m_mask_dedup.sbatch")
        print("    sbatch Delphi-2M/slurm/figure2.sbatch")


if __name__ == "__main__":
    main()
