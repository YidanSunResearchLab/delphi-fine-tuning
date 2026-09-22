"""AUC evaluation of the ROSMAP Delphi model, using gerstung-lab/delphi's own method.

The scoring, case/control construction, age-sex stratification, one-observation-per-person
sampling and DeLong variance all come from `evaluate_auc.evaluate_auc_pipeline`, which is
imported and called unmodified. This wrapper only supplies the three things upstream
hard-codes to UK Biobank:

  1. a labels table with the schema `get_common_diseases()` expects. Upstream reads
     `delphi_labels_chapters_colours_icd.csv` (1,270 ICD-10 codes with chapters and
     training counts); ROSMAP's labels.csv is a bare list of names, so the "chapter" is
     derived from the token-name prefix and `count` is counted off train.bin.
  2. age brackets. Upstream's default arange(40, 80, 5) covers only 42% of ROSMAP
     records (median age 82); see ROSMAP_AGE_GROUPS.
  3. DeLong confidence intervals (n_bootstrap=1). The bootstrap path upstream is
     CUDA-only and raises on CPU/MPS.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch

from model import Delphi, DelphiConfig
from utils import get_batch, get_p2i
from evaluate_auc import evaluate_auc_pipeline

# ROSMAP is an elderly cohort: val ages are p25=75.6, p50=82.0, p75=87.8, p99=99.4.
# 65-100 covers 95.7% of val records; upstream's 40-80 covers 42.2%.
ROSMAP_AGE_GROUPS = np.arange(65, 100, 5)

# Token-name prefix -> "chapter". Order matters: first match wins.
CHAPTER_RULES = [
    (r"^(Padding|No event)$", "Technical"),
    (r"^(Female|Male)$", "Sex"),
    (r"^(STUDY|EDU|RACE|HISP|APOE|SMOKING|ALCOHOL)_", "Background"),
    (r"_PREVALENT$", "Background"),
    (r"^Death$", "Death"),
    (r"^(AD_DX|STROKE|DEPRESSION)$", "Events"),
    (r"_ONSET$", "Events"),
    (r"_(ON|OFF)$", "Medications"),
    (r"^(BRAAK|CERAD|GPATH|AMYL|TANG|TDP|LEWY|ARTSCL|CAA|CVDA|MICROINF|INFARCT)_", "Neuropathology"),
]
# Everything else is a longitudinal clinical measure binned into levels (BMI_low, COGN_high, ...)
DEFAULT_CHAPTER = "Clinical measures"

# Neuropathology is excluded by default: those tokens are emitted only after [DEATH], so
# "predicting" them is a different task from incident-event prediction and the age-stratified
# control design does not apply. Background/Sex/Technical are the ignore_tokens.
DEFAULT_CHAPTERS_OF_INTEREST = ["Events", "Death", "Medications", "Clinical measures"]

CHAPTER_COLORS = {
    "Events": "#d62728",
    "Death": "#000000",
    "Medications": "#1f77b4",
    "Clinical measures": "#2ca02c",
    "Neuropathology": "#9467bd",
    "Background": "#9467bd",
    "Sex": "#bcbd22",
    "Technical": "#2a52be",
}


def chapter_of(name):
    for pattern, chapter in CHAPTER_RULES:
        if re.search(pattern, name):
            return chapter
    return DEFAULT_CHAPTER


def build_labels(data_dir, train_counts):
    """The ROSMAP stand-in for delphi_labels_chapters_colours_icd.csv."""
    names = pd.read_csv(os.path.join(data_dir, "labels.csv"))["event_name"].tolist()
    chapters = [chapter_of(n) for n in names]
    df = pd.DataFrame({
        "index": np.arange(len(names)),
        "name": names,
        "count": [train_counts.get(i, 0) for i in range(len(names))],
        "ICD-10 Chapter": chapters,
        "ICD-10 Chapter (short)": chapters,
        "color": [CHAPTER_COLORS[c] for c in chapters],
    })
    # Technical tokens have no training count upstream either
    df.loc[df["ICD-10 Chapter (short)"] == "Technical", "count"] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser(description="Delphi-style AUC evaluation on ROSMAP")
    ap.add_argument("--ckpt", default="Delphi-ROSMAP/ckpt.pt")
    ap.add_argument("--data_dir", default="data/rosmap")
    ap.add_argument("--output_path", default="Delphi-ROSMAP/auc")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--offset", type=float, default=0.1,
                    help="days the prediction token must precede the event. "
                         "0.1 = 'no gap' (upstream pipeline default), 365.25 = one-year gap")
    ap.add_argument("--no_event_token_rate", type=int, default=5)
    ap.add_argument("--filter_min_total", type=int, default=100)
    ap.add_argument("--dataset_subset_size", type=int, default=-1)
    ap.add_argument("--chapters", nargs="*", default=DEFAULT_CHAPTERS_OF_INTEREST)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = Delphi(DelphiConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    block_size = checkpoint["model_args"]["block_size"]
    print(f"loaded {args.ckpt}: iter {checkpoint['iter_num']}, "
          f"best val loss {checkpoint['best_val_loss']:.4f}, block_size {block_size}")

    train = np.fromfile(f"{args.data_dir}/train.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    val = np.fromfile(f"{args.data_dir}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    val_p2i = get_p2i(val)

    # token ids in the .bin are pre-shift; get_batch adds 1. Count in post-shift space.
    ids, counts = np.unique(train[:, 2] + 1, return_counts=True)
    train_counts = dict(zip(ids.tolist(), counts.tolist()))

    delphi_labels = build_labels(args.data_dir, train_counts)

    n = len(val_p2i) if args.dataset_subset_size == -1 else args.dataset_subset_size
    d = get_batch(range(n), val, val_p2i, select="left", block_size=block_size,
                  device=device, padding="random", no_event_token_rate=args.no_event_token_rate)
    print(f"evaluation batch: {d[0].shape[0]} trajectories x {d[0].shape[1]} positions")

    eligible = delphi_labels[
        delphi_labels["ICD-10 Chapter (short)"].isin(args.chapters)
        & (delphi_labels["count"] > args.filter_min_total)
    ]
    print(f"{len(eligible)} tokens pass chapter + count>{args.filter_min_total} filter:")
    for chapter, grp in eligible.groupby("ICD-10 Chapter (short)"):
        print(f"  {chapter:20s} {len(grp):3d}  {', '.join(grp['name'].tolist()[:8])}"
              f"{' ...' if len(grp) > 8 else ''}")

    df_unpooled, df_auc = evaluate_auc_pipeline(
        model,
        d,
        output_path=None,          # upstream writes parquet; we write csv below
        delphi_labels=delphi_labels,
        diseases_of_interest=eligible["index"].tolist(),
        filter_min_total=args.filter_min_total,
        age_groups=ROSMAP_AGE_GROUPS,
        offset=args.offset,
        device=device,
        seed=args.seed,
        n_bootstrap=1,             # DeLong; the bootstrap path upstream requires CUDA
        meta_info={"offset_days": args.offset},
    )

    os.makedirs(args.output_path, exist_ok=True)
    tag = "nogap" if args.offset < 1 else f"gap{int(round(args.offset))}d"
    df_unpooled.to_csv(f"{args.output_path}/auc_by_age_sex_{tag}.csv", index=False)

    df_auc["auc_ci95"] = 1.96 * np.sqrt(df_auc["auc_variance_delong"])
    cols = ["token", "name", "ICD-10 Chapter (short)", "count", "auc", "auc_ci95",
            "n_diseased", "n_healthy", "n_samples"]
    df_auc = df_auc[cols].sort_values("auc", ascending=False)
    df_auc.to_csv(f"{args.output_path}/auc_pooled_{tag}.csv", index=False)

    print(f"\n=== pooled AUC (offset={args.offset}d, {len(df_auc)} tokens) ===")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df_auc.to_string(index=False,
                               formatters={"auc": "{:.3f}".format, "auc_ci95": "{:.3f}".format}))
    print(f"\nwrote {args.output_path}/auc_pooled_{tag}.csv and auc_by_age_sex_{tag}.csv")


if __name__ == "__main__":
    main()
