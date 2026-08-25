"""
make_dataset_radc.py -- one command: the four RADC files -> train/val/test splits.

RADC sibling of make_dataset.py. Thin wrapper over the two steps (each still runnable alone):
  1) tokenize_radc.py   : <repo>/data/RADC/*  -> out_radc/radc_all.bin + labels.csv
        Dedup is applied HERE and baked into the .bin permanently: static one-per-subject,
        KEEP-TRANSITIONS per ordinal scale (ten of them), keep-first for everything else.
        Nothing downstream re-derives it.
  2) make_split_ad.py   : radc_all.bin -> out_radc/radc-dedup-s<seed>/{train,val,test}.bin
        By-subject 70/10/20 (no subject crosses splits). Same script the NACC arm uses --
        invoked with --dir out_radc --prefix radc.

Unlike make_dataset.py there is no --csv: tokenize_radc.py reads the four files straight from
<repo>/data/RADC/, which is where they already live.

It also symlinks the finished split to data/radc-dedup-s<seed>, the name config/train_radc.py
reads.

Usage:
  python data_prep/make_dataset_radc.py                  # seed 42
  python data_prep/make_dataset_radc.py --seed 1234
  python data_prep/make_dataset_radc.py --no-link         # don't touch ./data/
"""
import os, sys, argparse, subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Delphi-2M/
SCRIPTS = os.path.dirname(os.path.abspath(__file__))                 # data_prep/
REPO = os.path.dirname(HERE)
RADC = os.path.join(REPO, "data", "RADC")

REQUIRED = ["cross-sectional-data-gk.xlsx", "longitudinal_data_gk.xlsx", "ROSMAP_clinical.csv"]


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def point_symlink(link, target):
    """Point symlink `link` at `target`. NEVER deletes a real (non-symlink) file at `link`."""
    if os.path.realpath(link) == os.path.realpath(target):
        return
    if os.path.islink(link):
        os.remove(link)
    elif os.path.exists(link):
        sys.exit(f"[make_dataset_radc] refusing to delete real file at:\n    {link}\n"
                 f"  It is not a symlink. Move/rename it and re-run.")
    os.symlink(target, link)


def main():
    ap = argparse.ArgumentParser(description="Build train/val/test splits from the RADC files.")
    ap.add_argument("--seed", type=int, default=42,
                    help="split seed -> out_radc/radc-dedup-s<seed>/")
    ap.add_argument("--no-link", action="store_true",
                    help="do NOT symlink the split under ./data/ (default: do link it)")
    args = ap.parse_args()

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(RADC, f))]
    if missing:
        sys.exit(f"[make_dataset_radc] missing from {RADC}:\n    " + "\n    ".join(missing))
    print(f"[make_dataset_radc] input dir : {RADC}")

    # 1) merge + tokenise (dedup happens inside here)
    run([sys.executable, os.path.join(SCRIPTS, "tokenize_radc.py")])

    # 2) by-subject 70/10/20 split, via the shared splitter
    run([sys.executable, os.path.join(SCRIPTS, "make_split_ad.py"),
         "--dir", "out_radc", "--prefix", "radc", "--seed", str(args.seed)])

    split_dir = os.path.join(HERE, "out_radc", f"radc-dedup-s{args.seed}")

    # 3) make it usable by train.py (it reads data/<dataset>/)
    if not args.no_link:
        os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
        rel = os.path.relpath(split_dir, os.path.join(HERE, "data"))
        name = f"radc-dedup-s{args.seed}"
        point_symlink(os.path.join(HERE, "data", name), rel)
        print(f"[make_dataset_radc] linked data/{name} -> {rel}")

    print(f"\n[make_dataset_radc] DONE -> {split_dir}/  (train.bin, val.bin, test.bin, labels.csv)")
    if not args.no_link:
        print("  next (from Delphi-2M/):")
        print("    python training/train.py config/train_radc.py --device=cuda")
        print("  NOTE: figure2/ and delphi/ad_engine.py hard-code NACC token ids "
              "(NACCUDSD 106-109, Death 110)")
        print("        and will NOT read a RADC checkpoint correctly without their own pass.")


if __name__ == "__main__":
    main()
