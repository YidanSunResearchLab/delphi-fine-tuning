# AD-projection

Alzheimer's Disease cognitive-decline trajectory prediction using a transformer trained on
longitudinal [NACC](https://naccdata.org/) clinical data. Built on the
[Delphi](https://github.com/gerstung-lab/Delphi) framework.

**Project summary:** [Decoding Synergistic Comorbidity Drivers of Alzheimer's Disease Progression via AI-Driven Trajectory Perturbation](https://di2accelerator.wustl.edu/digital-transformation-corps/2026-projects/decoding-synergistic-comorbidity-drivers-of-alzheimers-disease-progression-via-ai-driven-trajectory-perturbation/)

**This repository is code only** — no data, no checkpoint, no figures. Everything is a *product* of
running the three commands below, reproducible from the raw NACC export (which you obtain
separately, see §1).

```bash
cd Delphi-2M
python data_prep/make_dataset.py --csv /path/to/investigator_nacc72.csv --seed 42   # ①
python training/train.py config/train_delphi2m_mask_dedup.py --device=cuda          # ②
./figure2/run_figure2.sh                                                            # ③
```

### Reference run

Verified end to end on 2026-08-01 from an empty checkout on RIS `compute2` — nothing reused, the
splits and checkpoint were rebuilt from the raw CSV. Compare against these to know your run is sane:

| step | where | wall time | what came out |
|---|---|---|---|
| ① prep | login node | ~2 min | `nacc_all.bin` **1,142,310 events / 55,268 patients**; splits **800,110 / 114,143 / 228,057** |
| ② train | 1× H100 | **3 min 34 s** | loss 39.4 → 9.85, val 10.197, no NaN; `ckpt.pt` **25,782,544 bytes**, 2.10M params |
| ③ Figure 2 | 32 CPU cores | **22 min 51 s** | 8 panels (4 × 2 cohorts), 0 failures; 12 PNG + 12 PDF + 14 CSV + 2 JSON, 28 MB |

Headline numbers from that run, `matched` cohort (**n = 4,589**): median transition AUC **0.668**;
trajectory Jaccard **0.643 [0.634, 0.652]** vs the carry-baseline-forward reference **0.510**;
10-NN embedding purity **0.473** (chance 0.215). Timing is the weak axis — transition-time
R² **−0.81**, MAE **3.14 y**, biased **late by +2.2 y**; read discrimination and calibration as
separate claims (§12).

> **These numbers predate the architecture change.** The reference run was measured with the
> previous 12/12/120, 2.10M-param shape. The config now ships **8/6/120, 1.41M params** (§9.4),
> so expect a smaller `ckpt.pt` (~17 MB) and a shorter ② — and slightly *better* Figure-2
> discrimination, per [`experiments/capacity/RESULTS.md`](experiments/capacity/RESULTS.md).
> The end-to-end run has **not** been repeated on the new shape.

### Contents

**Running it** — §1 [Setup](#1-setup) · §2 [Prep the data](#2-step-1--prep-the-data) ·
§3 [Train](#3-step-2--train) · §4 [Figure 2](#4-step-3--figure-2) ·
§5 [Reading the results](#5-how-to-read-the-results) · §6 [Troubleshooting](#6-troubleshooting) ·
§7 [Repository structure](#7-repository-structure)

**How it works** — §8 [Data](#8-data) · §9 [Model](#9-model) ·
§10 [⚠️ Content-renormalization](#10--content-renormalization--the-gotcha-that-breaks-everything) ·
§11 [Losses](#11-objective--the-two-losses) · §12 [Evaluation](#12-evaluation) ·
§13 [Decisions on record](#13-decisions-on-record) · §14 [Known hazards](#14-known-hazards) ·
§15 [Correctness audit](#15-correctness-audit)

Panel-by-panel Figure 2 method notes live separately in
[`Delphi-2M/figure2/FIGURE2.md`](Delphi-2M/figure2/FIGURE2.md).

---

# Part I — Running it

> **Start here.** Everything needed to reproduce the pipeline end to end: environment, the three
> commands, what a healthy run looks like, and what to do when one fails.

## 1. Setup

```bash
conda env create -f environment.yml
conda activate ad-projection
cd Delphi-2M
```

`environment.yml` installs the **CUDA 12.6** torch wheels — right for a Linux GPU cluster. For
CPU-only or a different CUDA, edit the `--extra-index-url` (e.g. `.../whl/cpu`, `.../whl/cu121`).
`training/train.py` carries a GradScaler shim so it also works on torch < 2.4. `umap-learn` is
needed by **Figure 2 panel d** only (a/b/c render without it).

### What you must obtain

| File | Where it comes from | In repo? |
|---|---|---|
| `investigator_nacc72.csv` | **The only thing you must obtain** — the raw NACC export. Contact **Dr. Yidan Sun** (syidan@wustl.edu). ~516 MB, 1024 columns, one row per visit. | ❌ data-use agreement |
| `Vocabulary NACC.xlsx` | Token index→label table the tokenizer reads to write `labels.csv`. | ✅ `Delphi-2M/data_prep/` |
| `labels.csv` | Index → human-readable token name. No code reads it (names come from constants in `delphi/ad_engine.py`); it exists so a human can look up what token 87 means. | ⚙️ generated at prep |

Keep the raw CSV **outside** the repository and pass `--csv /path/to/it` — nothing requires it to
live inside. The prep step is deterministic, so the counts in §2 should reproduce exactly.

### Running on SLURM

Two scripts, one per compute-heavy step. Set two environment variables once, then submit — you
should not have to edit any file:

```bash
cd /path/to/this/checkout                 # sbatch from the repo root: logs/ is resolved from here
export AD_BASE=$PWD                       # where this checkout lives
export AD_CONDA=/path/to/miniforge3       # only if conda is NOT inside AD_BASE

sbatch Delphi-2M/slurm/train_delphi2m_mask_dedup.sbatch   # general-gpu, 1 GPU, 4 h -> ckpt.pt
sbatch Delphi-2M/slurm/figure2.sbatch                     # general-cpu, 32 cores, 3 h -> figures
```

| variable | default | when to set it |
|---|---|---|
| `AD_BASE` | the original cluster checkout | **always**, unless you are in that exact directory |
| `AD_CONDA` | `$AD_BASE/miniforge3` | when your conda env lives outside the checkout — e.g. you made a fresh checkout and want to reuse an existing `miniforge3`. Both scripts fail fast with the fix in the message if it is wrong. |

Both write `.out`/`.err` into **`logs/`**, resolved relative to where you `sbatch` from — so submit
from the repo root. The directory ships with a `.gitkeep` because SLURM cannot write the job log if
it does not exist (symptom: the job dies instantly with no log at all).

Prep (§2) needs no job — run it on a login node first.

Training wants a GPU; the Figure-2 Monte-Carlo pass is CPU-multiprocess (one worker per core, torch
single-threaded inside each) and belongs on the **CPU** queue, not the GPU one.

> On the RIS login node `scipy`/`sklearn` fail to import (`GLIBCXX_3.4.30 not found` — the login
> node's libstdc++ is older than the compute nodes'). That is expected and does not affect the jobs.
> Prep only needs numpy/pandas/openpyxl, which do work there.

---

## 2. Step 1 — Prep the data

```bash
python data_prep/make_dataset.py --csv /path/to/investigator_nacc72.csv --seed 42
```

Tokenizes each patient's record into an `(age, event)` stream (`data_prep/tokenize_nacc_ad.py`),
splits by patient 70/10/20 (`data_prep/make_split_ad.py`), and symlinks the result to
`data/nacc-dedup-s42` — the name the configs and the evaluation read.

("dedup" is the keep-transitions tokenisation, §8.1. The name says so explicitly, so a
differently-tokenised build can never silently take its place.)

**Expected output — these numbers are deterministic, use them to check your rebuild:**

```
  nacc_all.bin: 1142310 events, 55268 patients (FULL, unsplit)
seed 42 | patients 55268 -> train 38687 val 5526 test 11055
  train.bin:    800110 events,  38687 patients
  val.bin:      114143 events,   5526 patients
  test.bin:     228057 events,  11055 patients
```

---

## 3. Step 2 — Train

```bash
python training/train.py config/train_delphi2m_mask_dedup.py --device=cuda
```

Trains with next-token cross-entropy + an exponential time-to-event loss. Best checkpoint →
`out-delphi2m-dedup-mask-s42/ckpt.pt` — **the delivered model**, and the name the evaluation
expects. Dataset, output dir and hyperparameters all come from the config; CLI flags override any.

- **This config** = 8 layers / 6 heads / 120-dim, **1,412,160 params**, 5000 iters, with
  `ignore_tokens = 0..21` dropped from the **loss** — see §9.3 and §10. The architecture was
  changed away from the upstream Delphi-2M shape (12 layers / 12 heads, 2.10M params) on the
  evidence of [`experiments/capacity/RESULTS.md`](experiments/capacity/RESULTS.md): 1.49× smaller,
  ~1.6× faster, and better on every downstream Figure-2 metric.
- **Superseded:** `config/train_nacc.py` (6 layers / 384-dim / 20k iters) overfit — val loss
  bottomed near step 5000 then rose. `config/train_delphi2m.py` is the no-loss-mask ablation
  baseline — but note that it, and every other `config/*.py`, still carries the **old 12/12/120**
  shape. It is therefore now an ablation of *both* the mask and the architecture; re-run it at
  8/6/120 before reading it as a clean mask ablation.
- **Cohort filter (on by default):** only subjects with ≥4 distinct visits, OR 2–3 visits showing a
  NACCUDSD transition. You'll see `train 38687 -> 16262 | val 5526 -> 2349`. Disable with
  `--cohort_min_visits=1`. **This filter also defines Figure 2's `matched` cohort**, so changing it
  changes which patients the figure reports on.
- wandb is off; add `--wandb online`. Smoke run: `--max_iters=800 --eval_interval=200` (expect
  near-chance AUCs — that is the smoke run, not a bug).

> ⚠️ **Do not run `--eval_only=True` in a directory holding a checkpoint you care about.** See §14.

---

## 4. Step 3 — Figure 2

The only evaluation in this repository: four panels assembled into one combined figure.

```bash
./figure2/run_figure2.sh                 # MC build -> panels a-d -> combined figure, both cohorts
./figure2/run_figure2.sh --limit 300     # smoke run, ~2 min (numbers NOT publishable)
DEVICE=cuda ./figure2/run_figure2.sh     # MC pass on GPU instead of CPU multiprocess
WORKERS=8   ./figure2/run_figure2.sh     # cap the worker pool (default: cores - 2)
```

It preflights the checkpoint and **fingerprints the dataset** before spending any compute — the two
failure modes it catches (missing dataset link, keep-all build instead of dedup) both otherwise
produce a *quietly wrong figure* rather than a crash. Extra arguments are forwarded to all three
steps, so only pass flags every step accepts: `--limit`, `--split`, `--n-mc`.

The three underlying steps, if you want to drive them yourself:

```bash
python figure2/figure2_core.py   --build --device cpu --workers 30   # Monte-Carlo + embedding cache
python figure2/figure2_panels.py --cohort matched                    # panels a-d + combined figure
python figure2/figure2_panels.py --cohort all                        # the *_allcohort variants
```

`--build` is the expensive step (measured 22 min 51 s on 32 cores), cached under `results/figure2/_cache/` keyed
by (checkpoint, split, n_mc); the panel commands then run in ~70 s and can be re-run freely. Both
read `data/nacc-dedup-s42` unless you pass `--dataset`. Delete the cache to force a rebuild.
On SLURM: `sbatch Delphi-2M/slurm/figure2.sbatch` (see §1 for `AD_BASE`/`AD_CONDA`) — it drives the same two programs
with the same arguments; keep the two in step if you change either.

**Outputs → `results/figure2/`.** Every file carries an explicit cohort suffix — `_matched` is the
paper figure, `_allcohort` the generalisation check:

| file | what it is |
|---|---|
| `figure2_combined_matched.{png,pdf}` | **the assembled four-panel figure** |
| `figure2_combined_allcohort.{png,pdf}` | same, scored on the full evaluable test split |
| `fig2a_matched`…`fig2d_matched.{png,pdf}` | the same panels standalone, larger (identical code path) |
| `fig2*_data.csv` | the exact numbers behind every mark in that panel |
| `metrics_matched.json` | every reported statistic, machine-readable |
| `SUMMARY_matched.md` | narrative version, auto-generated |

---

## 5. How to read the results

**Which cohort you are looking at — read this first.** `train.py` does not fit on every patient in
`train.bin` (§3). So the figures come in two versions out of the *same* Monte-Carlo cache:

| version | files | answers |
|---|---|---|
| **matched** (the paper figure) | `*_matched.*` | how the model does on the population it was actually trained for |
| full cohort | `*_allcohort.*` | generalisation — ~43% of these patients are outside the training regime |

Neither has leakage: the split is by patient and test never touched training.

**The four panels**

* **a — transition-prediction accuracy.** AUC per transition type, observed vs predicted transition
  probability side by side, and predicted vs observed transition *time*.
* **b — timing accuracy by interval.** Error in predicted event time binned by how far ahead the
  event was, and discrimination as a function of horizon.
* **c — per-individual trajectories.** Observed vs predicted stage-at-age, scored by Jaccard against
  a carry-baseline-forward reference, with per-state IoU. **The carry-forward baseline is the honest
  comparator — beating it is the claim.**
* **d — embedding structure.** UMAP of patient embeddings coloured by observed trajectory class,
  with k-NN purity. Needs `umap-learn`; reported as failed without it.

**How every predicted number is produced.** The model is autoregressive, so there is **no
closed-form risk at a horizon**. Every predicted quantity is a Monte-Carlo statistic over **100
sampled trajectories per patient**, seeded on that patient's **first visit** (birth-time statics +
the whole baseline visit, nothing after it). That prompt is the entire input — no future leakage —
and it is what makes the panels comparable to each other. Two consequences:

* **Competing risks are handled by construction.** A sampled trajectory that dies stops there, so
  "reaches Dementia within 5 y" counts only when sampled dementia precedes sampled death. On the
  observed side the matching estimator is Aalen–Johansen, not the naive event proportion.
* **Right-censoring is explicit.** A patient whose follow-up ends before the horizon with no event
  has an *unknown* label and is dropped from AUCs rather than counted as a negative.

`metrics_matched.json` carries every statistic machine-readably; each panel's `*_data.csv` holds the exact
numbers behind every mark, so a figure can be re-derived without re-running the MC.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: data/nacc-dedup-s42/train.bin` | You haven't run Step 1. The repo ships code only — `out_ad/` and `data/` don't exist until `make_dataset.py` builds them. |
| `FATAL: checkpoint not found` | You haven't run Step 2. No checkpoint is committed. |
| Figure 2 aborts with "expected the DEDUP test split" | `data/nacc-dedup-s42` points at the keep-all build (~329k events), not keep-transitions (228,057). Re-point the symlink. |
| Panel d reported as "failed" | `pip install umap-learn`. Panels a/b/c are unaffected. |
| `FATAL: no conda at ...` | The job found no conda inside the checkout. `export AD_CONDA=/path/to/miniforge3` — see §1. |
| SLURM job dies instantly, no log | Either `logs/` doesn't exist relative to where you submitted, or `AD_BASE` is wrong. Submit from the repo root; see §1. |
| CUDA out of memory | Lower `--batch_size` (e.g. 64) or request a larger-VRAM GPU. |
| `torch.amp has no attribute GradScaler` | torch < 2.4; the shim in `train.py` handles it. |
| AUCs all ≈0.5 | Checkpoint under-trained (e.g. a smoke run) — train more iters. |
| `CSV not found` | Pass `--csv /path/to/investigator_nacc72.csv`. |

---

## 7. Repository structure

```
AD-projection/
├── README.md                              # this file — everything except Figure-2 panel details
├── environment.yml                        # conda env `ad-projection` (CUDA 12.6 torch wheels)
├── logs/                                  # SLURM writes job .out/.err here (gitignored)
└── Delphi-2M/                             # the package -- ALL commands run from this directory
    │
    ├── data_prep/               ①  raw CSV -> train/val/test splits
    │   ├── make_dataset.py          ENTRY POINT: wraps the two below + makes the data/ symlinks
    │   ├── tokenize_nacc_ad.py      CSV -> out_ad/nacc_all.bin (5 cognitive scales, keep-transitions)
    │   ├── make_split_ad.py         by-patient 70/10/20 (no subject crosses splits)
    │   └── Vocabulary NACC.xlsx     token index -> label table, read by the tokenizer
    │
    ├── training/                ②  fit the model
    │   ├── train.py                 ENTRY POINT: next-token CE + exponential time-to-event loss
    │   └── configurator.py          loads config/*.py, then applies CLI overrides
    ├── config/                      hyperparameters
    │   ├── train_delphi2m_mask_dedup.py   << the DELIVERED model
    │   ├── train_delphi2m.py              same, without the loss mask (ablation)
    │   └── train_nacc.py                  superseded (10.8M params, overfit)
    │
    ├── delphi/                  ⚙  the model and the one abstraction over it
    │   ├── model.py                 the transformer (age-encoding instead of positional)
    │   ├── utils.py                 get_p2i / get_batch / patient_stream -- owns the ±1 token shift
    │   ├── ad_engine.py             checkpoint loading + model-space token constants
    │   └── predict_adapter.py       THE model API: content-renormalization + MC sampling
    │
    ├── figure2/                 ③  the only evaluation
    │   ├── run_figure2.sh           ENTRY POINT: preflight -> build -> panels -> combined figure
    │   ├── figure2_core.py          Monte-Carlo + embedding cache (the expensive part)
    │   ├── figure2_panels.py        panels a-d + the assembled figure (pure plotting)
    │   ├── plotting_style.py        shared palette + PDF/PNG/CSV save helpers
    │   └── FIGURE2.md               panel-by-panel method notes
    │
    └── slurm/                   AD_BASE / AD_CONDA to relocate -- see §1
        ├── train_delphi2m_mask_dedup.sbatch   ②  general-gpu, 1 GPU, 4 h
        └── figure2.sbatch                    ③  general-cpu, 32 cores, 3 h

  created at runtime, never committed:  data/  out_ad/  out-*/  results/
```

**Dependency layering** — no evaluation code should reach past `predict_adapter.py` into model
internals:

```
delphi/model.py + delphi/utils.py     transformer + the ±1 token shift  ← never touch casually
      ↑
delphi/ad_engine.py                   checkpoint loading + token constants
      ↑
delphi/predict_adapter.py             THE single abstraction (content-renorm + MC sampling)
      ↑
figure2/figure2_core.py               Monte-Carlo + embedding cache
      ↑
figure2/figure2_panels.py             pure plotting, reads only the cache
```

---

# Part II — How it works

> **Optional.** Read this if you have extra time, or want a deeper understanding of how things are
> structured and designed — it is not needed to run the pipeline.

## 8. Data

**Source:** NACC investigator dataset — **55,268 patients · 1,142,310 events** (~20 per patient).

### 8.1 Tokenizer and the dedup rule

Each clinical fact becomes a **token**. The tokenizer **deduplicates** so the stream stores
*changes*, not repeated measurements:

- **Non-cognitive tokens: keep-first** — each fact emitted once at age-of-first-onset
  (hypertension recorded at 5 visits → 1 token; sex/APOE → 1 ever).
- **Every cognitive scale: keep-transitions, per scale** — NACCUDSD, the **six CDR box scores**
  (`MEMORY` `ORIENT` `JUDGMENT` `COMMUN` `HOMEHOBB` `PERSCARE` — each its own scale, replacing the
  single `CDRSUM` total; §8.5), MoCA, FAQ and NPI-Q
  are each a longitudinal ordinal *state*, so each is run-length-encoded **within its own scale**:
  first value + every bin **change** (decline *and* recovery), collapsing consecutive same-bin
  repeats.

Consequence: **~20 events describe a whole patient**, *not* per-visit. Dedup is why 5 visits ≠ 5×
the tokens — and it is **baked permanently into the `.bin`**. Nothing downstream re-derives it, and
the delivered checkpoint was trained on exactly this encoding.

> Keeping *recoveries* is precisely why all five scales are run-length-encoded rather than
> keep-first: keep-first deletes ~44% of recovery transitions. It also means a cognitive token is
> **always a state change**, so trajectory plots must be read as transition-anchored, not dense.

### 8.2 On-disk format

A flat `uint32` array reshaped `(-1, 3)`, sorted by `(patient_id, age_days, token_id)`:

| col 0 | col 1 | col 2 |
|---|---|---|
| `patient_id` (0–55,267) | `age_days` from birth | `token_id`, **disk space 1–109** |

```python
data = np.fromfile("Delphi-2M/data/nacc-dedup-s42/train.bin", dtype=np.uint32).reshape(-1, 3)
```

There is **no visit column and no per-patient object** — both boundaries are *implied*:

- **New subject** ⟺ col 0 changes. `get_p2i(data)` → `(n_patients, 2)` of `[start_row, count]`;
  patients are contiguous.
- **New visit** ⟺ col 1 changes. A "visit" is the rows sharing one `(patient_id, age_days)`.

Rows are age-sorted per patient; static tokens (sex, APOE) sit at **age 0**. Patient IDs are
integers; the map back to real NACCIDs lives in `out_ad/nacc_pidmap.csv` (gitignored —
re-identification risk), never in the `.bin`.

### 8.3 ⚠️ Token spaces — the easiest way to break everything silently

The `.bin` stores **disk** ids = model index − 1. `get_batch()` adds **+1** to restore **model
space**, which frees id 0 for padding and 1 for the no-event token. **Model space == `labels.csv`
row index** (row 106 = "Normal cognition", row 110 = "Death"). Get this wrong and every result
shifts by one token, with no error.

`delphi/utils.py` owns both the shift (`get_batch`) and the stream reader that applies it
(`patient_stream`), so all callers work in one consistent space.

### 8.4 Batch construction — `get_batch()`

Each step samples `batch_size` **patients** (with replacement); each becomes **exactly one row**:

1. **One patient = one contiguous window**, cropped to `block_size` (96), `select='left'`. Patients
   are never split across rows or stitched across forward passes — no cross-pass memory. For NACC
   this rarely truncates (~20 events ≪ 96).
2. **No-event tokens** (token 1) injected at ~5-year intervals, so *elapsed time without events* is
   itself signal — only up to the patient's last real event. At inference, no-event tokens can be
   inserted to query risk at arbitrary ages.
3. **+1 shift** (§8.3).
4. Trailing positions are **padding**; padding ages use a large negative sentinel (−10000).

Returns `X` (tokens `(B,96)`), `A` (ages), `Y` (next token = `X` shifted left), `B` (age of the next
token, for the time loss). Padding is excluded from both losses via a `pass_tokens` mask.

### 8.5 Vocabulary (228 tokens) and the outcome encodings

Token order = clinical severity: **higher token = worse**. Raw→token thresholds come from
`data_prep/tokenize_nacc_ad.py`.

| Token ID | Category | Raw variable | Binning (raw → token) |
|---|---|---|---|
| 0 / 1 | Padding / no-event | — | — |
| 2–3 | Sex | | Male, Female |
| 4–6 | BMI | | Low (≤22) / Mid (22–28) / High (>28) |
| 7–9 | Smoking | | Low / Mid / High |
| 10–12 | Alcohol | | Low / Mid / High |
| 13–15 | Education | | ≤12 yr / 13–16 yr / ≥17 yr |
| 16–21 | APOE genotype | | e3/e3, e3/e4, e3/e2, e4/e4, e4/e2, e2/e2 |
| 22–23 | Senses | | Vision loss, Hearing loss |
| ~~24–28~~ | ~~CDR-SB~~ | ~~`CDRSUM`~~ | **RETIRED** — replaced by the six CDR box scores at 111–139. Dead slots, kept so every other token id stays put. |
| **29–32** | **MoCA** | `NACCMOCA` | 26–30→29; 18–25→30; 10–17→31; 0–9→32 (**raw lower=worse, token higher=worse**). Left as a TOTAL: its ~19 sub-items exist but MoCA is UDS-v3-only, 36% visit coverage |
| ~~33–36~~ | ~~FAQ total~~ | ~~Σ 9 FAQ items~~ | **RETIRED** — replaced by the nine FAQ domains at 140–175 |
| ~~37–40~~ | ~~NPI-Q total~~ | ~~NPI severity sum~~ | **RETIRED** — replaced by the twelve NPI-Q symptoms at 176–223 |
| 41–59 | Medications | | Antihypertensives (41–49), lipid-lowering (50), **55 = FDA-approved AD medication**, diabetes (59), SSRIs, … |
| 60–105 | Medical conditions | | CVD, stroke, TIA, diabetes, Parkinson's, psychiatric (ICD-10) |
| **106–109** | **NACCUDSD** | `NACCUDSD` | 1→106 Normal; 2→107 Impaired-not-MCI; 3→108 MCI; 4→109 Dementia |
| **110** | **Death** | `NACCDIED` + `NACCYOD/MOD` | died→110 at age = death_date − birth; else right-censored |
| **111–139** | **CDR box scores** (6 domains) | `MEMORY` `ORIENT` `JUDGMENT` `COMMUN` `HOMEHOBB` `PERSCARE` | raw level → token, no binning: 0 / 0.5 / 1 / 2 / 3 → 5 tokens per domain. `PERSCARE` has **no 0.5 level** → 4 tokens. Blocks: MEMORY 111–115, ORIENT 116–120, JUDGMENT 121–125, COMMUN 126–130, HOMEHOBB 131–135, PERSCARE 136–139 |
| **140–175** | **FAQ domains** (9) | `BILLS` `TAXES` `GAMES` `STOVE` `MEALPREP` `EVENTS` `PAYATTN` `REMDATES` `TRAVEL` | raw level → token: 0 Normal / 1 has difficulty / 2 needs assistance / 3 dependent. 4 tokens per domain, in the order listed (BILLS 140–143 … TRAVEL 172–175). Code 8 "never did this activity" → 0 |
| **176–223** | **NPI-Q symptoms** (12) | `DELSEV` `HALLSEV` `AGITSEV` `DEPDSEV` `ANXSEV` `ELATSEV` `APASEV` `DISNSEV` `IRRSEV` `MOTSEV` `NITESEV` `APPSEV` | 0 absent / 1 mild / 2 moderate / 3 severe. 4 tokens per symptom (DEL 176–179 … APP 220–223). Severity is only recorded when the symptom is present, so the level is severity-if-present, 0-if-administered-and-absent, nothing if not administered |
| **224–227** | **GDS** (new) | `NACCGDS` | 0–4→224 Normal; 5–8→225 Mild; 9–11→226 Moderate; 12–15→227 Severe. The 15-item total, **not** per item — the items are binary and flip constantly (30 tokens, ~991k events); the total is 86.7% covered for 4 tokens |

> **Naming caveat:** tokens 33–36 are the 9-item **Functional Activities Questionnaire** sum. Older
> code and comments call this "FAST"/"FASTOTAL" — it is **not** Reisberg FAST staging. Renamed to
> FAQ throughout the evaluation on 2026-07-22; the underlying tokens are unchanged.

> **Why every sum-score is now its items.** Three totals were retired the same way and for
> the same reasons — `CDRSUM` (was 24–28), the FAQ total (33–36) and the NPI-Q total (37–40).
> A total is a deterministic function of its items, so feeding both is redundant *and* is
> same-visit leakage; and summing destroys **which domain moved**, which is itself a
> prediction target here, not just a feature. Splitting FAQ also retires
> `FAST_REQUIRE_COMPLETE`: the old all-or-nothing rule discarded 3,773 visits (1.8%) that had
> at least one valid item, and now each domain stands alone. The worked case:
>
> **Why CDR is six scales and not one.** `CDRSUM` is the arithmetic sum of the six box scores —
> on every row where both are present they agree exactly — so feeding the model both the total
> and the boxes would be pure redundancy *and* same-visit leakage. The total also has a lumpy
> support: several nominal 0.5-steps in 0–18 are structurally unreachable, because `PERSCARE`
> contributes no 0.5. Splitting into six scales gives six clean ordinal supports, and lets a
> memory-only decline register while the other five domains sit still — which the sum cannot
> represent. Cost across all three splits: **1.14M → 3.06M events** and 113 token ids
> instead of 13.
> `PERSCARE` is the weakest of the six (83% of visits sit at 0) and was flagged `drop` by
> `eda_round3`; it is kept here by explicit decision.

NPI-Q exists but is not one of the five headline outcomes (treat as extra).

### 8.6 Worked example — a 5-visit patient

```
age 70y : Female, APOE e3/e4, BMI_mid + the FIRST value of all 30 scales
          (NACCUDSD, MoCA, 6 CDR domains, 9 FAQ domains, 12 NPI symptoms,
          GDS) — the baseline visit is now the big one   → ~35 events (baseline)
age 72y : NACCUDSD-MCI, MoCA-MCI, Memory-Questionable,
          REMDATES-Difficulty                            → 4 events   (transitions)
age 74y : Hypertension, SSRIs, Memory-Mild,
          Judgment-Questionable, DEPD-Mild, GDS-Mild     → 6 events   (new facts + changes)
age 75y : (nothing changed on any scale)                 → 0 events   (nothing new)
age 77y : NACCUDSD-Dementia, Memory-Moderate, Orient-Mild,
          BILLS-Dependent, TAXES-Dependent, APA-Moderate → 6 events   (transitions)
                                                         ≈ 51 events total
```

Each cognitive **scale** contributes a token only when *its* bin changes, so one visit can add
several cognitive tokens — and with every sum-score split into items there are **30 scales**, so
the baseline visit in particular is large. This is why `block_size` moved 96 → **256**: at a mean
of 55.4 tokens/patient (p95 100, max 250) a 96-window truncated **6.0% of patients**, and since
training runs `select='left'` the part it drops is their *latest* events — the conversions. Those
3,328 patients are not a random 6%: **86% of them have a dementia record against 41% overall**.
At 256 nobody is truncated and the whole patient is still seen in one pass.

### 8.7 Reference distributions (test split)

Useful for sanity-checking a rebuild. **611,673 events / 11,055 patients**; events per patient
median 48 (mean 55.3, max 224); age span 18–106 y; 19,234 static (age-0) tokens.
(This split was 228,057 events before any sum-score was split, and 310,241 after CDR only. The
patient membership never changed — same seed, same by-patient split — so every count below that
does not involve a split scale is byte-identical across all three versions.)

- **Death:** present for **3,034 / 11,055 (27.4%)** → 72.6% right-censored.
- **NACCUDSD:** every patient has ≥1, but only **2,447 (22%) have ≥2** — i.e. a transition to forecast.
- MoCA 29–32: 3050 / 3540 / 1073 / 444 · NACCUDSD 106–109: 5438 / 1174 / 3778 / 4673 · Death 110: 3034.
- GDS 224–227 (Normal / Mild / Moderate / Severe): 9950 / 2554 / 703 / 258.
- CDR boxes, per level (None / Questionable / Mild / Moderate / Severe) —
  MEMORY 111–115: 5852 / 4923 / 3715 / 2022 / 1001 · ORIENT 116–120: 7829 / 3290 / 2898 / 1688 / 997 ·
  JUDGMENT 121–125: 6977 / 4272 / 3528 / 1620 / 1155 · COMMUN 126–130: 7525 / 3151 / 2869 / 2010 / 908 ·
  HOMEHOBB 131–135: 7671 / 3209 / 2854 / 1853 / 1273 · PERSCARE 136–139 (no Questionable level):
  9822 / 2257 / 1357 / 985.
- FAQ domains, per level (Normal / Difficulty / Assistance / Dependent) —
  BILLS 8085/1838/1854/3466 · TAXES 8201/1669/1779/3665 · GAMES 8934/2530/1838/2141 ·
  STOVE 9413/2102/1360/2120 · MEALPREP 8844/1844/1678/2807 · EVENTS 8533/3013/2220/2200 ·
  PAYATTN 8751/3542/2107/1631 · REMDATES 7468/3597/3123/2992 · TRAVEL 7901/2553/1781/3683.
- NPI-Q symptoms, per level (Absent / Mild / Moderate / Severe) —
  DEL 10575/1087/654/269 · HALL 10559/762/347/158 · AGIT 10608/3194/1759/506 ·
  DEPD 10392/4601/2186/488 · ANX 10526/4051/2289/581 · ELAT 10708/614/294/57 ·
  APA 10130/3239/2137/915 · DISN 10627/1935/1068/417 · IRR 10556/4321/2142/582 ·
  MOT 10546/1596/1016/469 · NITE 10891/3365/2092/748 · APP 10949/2950/1551/508.

---

## 9. Model

A causally-masked transformer (`delphi/model.py`) with three departures from vanilla GPT.

### 9.1 Exact I/O interface

```
Delphi.forward(idx, age, targets=None, targets_age=None, validation_loss_mode=False)
    idx : LongTensor  (B, T)   model-space token ids (0..227)
    age : FloatTensor (B, T)   age in DAYS from birth (float)
    returns (logits, loss, att)
      logits : (B, T, vocab=228)   per-position next-event scores
      loss   : dict{loss_ce, loss_dt} if targets given, else None
      att    : stacked attention maps

Delphi.generate(idx, age, max_new_tokens, max_age, no_repeat=True, termination_tokens=[110])
    autoregressive sampler; sets logits[:, ignore_tokens] = -inf before sampling.
    Returns (idx, age, logits) grown with the rollout. Terminates on Death (110) or max_age.
    THIS IS THE TRAJECTORY SAMPLER used for every predicted quantity.
```

**Two heads share the one `logits` tensor:**

1. **Categorical next-state** = `softmax(logits)` over the 228 tokens.
2. **Time-to-next-event** = the *same* logits as **log-rates of competing exponentials**. Next
   `(token, Δt)` is drawn as `Δt_k = -exp(-logit_k)·log U`, take the min over k. Total intensity
   `λ = Σ_k exp(logit_k)`; `E[Δt] ≈ 1/λ`. **Units: days.**

Config baked into the **delivered** checkpoint's `model_args` (read back by `load_model`):
`block_size=96, vocab_size=111, n_layer=8, n_head=6, n_embd=120, t_min=30.4375 (≈1 month),
mask_ties=True, ignore_tokens=[0..21]`. **1,412,160 params.**

> ⚠️ **`ckpt.pt` predates the sum-score split.** It was trained at `vocab_size=111, block_size=96`,
> when 24–28 / 33–36 / 37–40 were the CDRSUM, FAQ and NPI-Q totals. The tokenizer now emits
> `vocab_size=228` with 30 per-item scales, so the shipped checkpoint **must not be run against
> the current data**: its embedding and tied head are 117 rows short and its context is 160
> positions short. `load_model` reads `vocab_size` back out of the checkpoint, so
> it will happily build the old 111-token model and only blow up later, as an `IndexError` the
> first time a CDR-box token is looked up — or, worse, score correctly on a slice that happens to
> contain none. Pass `expect_vocab_size=140` to `delphi/model.py::load_checkpoint` if you want that
> caught at load time. Retraining is required; §9.4 and §11 describe the old model until then.

### 9.2 Age encoding, and same-visit attention masking

```python
x = wte(idx) + wae(age.unsqueeze(-1))       # age encoding replaces positional encoding
attn_mask = (age[query_pos] != age[key_pos])  # mask_ties: same-visit tokens can't see each other
```

The model sees **real time** — a 10-year gap and a 1-month gap embed differently. Position index is
irrelevant; **age** orders and spaces events.

`mask_ties` blocks **within-visit leakage**: when predicting a visit's NACCUDSD token the model
cannot peek at the CDR/MoCA tokens recorded at that same visit. **It must be active at eval too** —
you must pass `targets`/`targets_age` (dummy values are fine) or the tie mask never activates and
staging accuracy is inflated.

### 9.3 `ignore_tokens` — the delivered model's loss mask

`ignore_tokens = 0..21` (padding, no-event, and every static/background token: sex, BMI, smoking,
alcohol, education, APOE) are **dropped from the loss** — not predicted, not scored — but stay in
the input stream and are still attended to.

This is a **"don't-predict" mask, NOT a feature ablation**: APOE remains visible context. It stops
the time-to-event loss being dominated by the no-event grid and by immutable statics.

Its consequence is the single most important thing to know about this model → **§10.**

### 9.4 Config — the delivered model (`config/train_delphi2m_mask_dedup.py`)

| Parameter | Value | |
|---|---|---|
| Vocabulary / block size | 111 / 96 | **as trained.** The tokenizer now emits 228 / 256 (§8.5) — retrain needed |
| Layers / heads / embedding | 8 / 6 / 120 | 1,412,160 params, head_dim 20 — chosen in `experiments/capacity`, **not** inherited |
| Dropout / token dropout | 0.0 / 0.0 | |
| `ignore_tokens` | `0..21` | §9.3 |
| `t_min` | `365.25/12` (~1 month) | **do not lower** — see below |
| max_iters / batch | 5000 / 128 | |
| LR / min LR / warmup | 2e-3 / 2e-4 / 500 | `weight_decay` 0.2, `beta2` 0.99, seed 42 |

> **`t_min` must stay > 0.** It caps the predicted intensity at λ ≤ 1/`t_min` so `loss_dt`
> (= λ·dt − log λ) cannot blow up. `t_min = 0` → `logsumexp(logits)` overflows in fp32 → `loss_dt`
> becomes NaN. This killed a 20k-iter run (weights went NaN between iters 10000–20000).

---

## 10. ⚠️ Content-renormalization — the gotcha that breaks everything

Because `ignore_tokens = 0..21` were **never trained as prediction targets**, their logits are
**unconstrained garbage — and they dominate the raw full-vocab softmax.** Probe output from real
patients on this checkpoint:

```
patient 0 next-event top5: Alcohol low .27, Alcohol high .23, Alcohol mid .21, Male .12, Female .05
patient 1 next-event top5: APOE e2e2 .61, APOE e3e3 .19, Alcohol mid .06, ...
E[time-to-next-event] via full-vocab logsumexp = 0.00 yr   ← garbage
```

**Consequences:**

- Any "next event probability", "top predicted event" or closed-form rate **must renormalize over
  the non-ignored tokens only (22–110)**. `delphi/predict_adapter.py` does this centrally — that is
  the reason the adapter exists. Skip it and AUCs collapse toward chance.
- **Per-scale restricted softmax is SAFE** (it softmaxes only over one scale's own tokens, all >21),
  so ordinal staging and transition matrices are unaffected.
- **Closed-form time-to-event is unreliable** (dominated by masked-token intensities and `t_min`
  flooring). **Get event times from Monte-Carlo trajectory sampling via `generate`**, which masks
  0–21 correctly — never from a closed-form rate. This is why every predicted quantity in Figure 2
  is an MC statistic.

A probe confirms `generate` itself is well-behaved: a patient-0 rollout produced a coherent future
(…MoCA Normal→MCI… NACCUDSD Normal @70.8 y … Dementia @91.4 y), realistic ages, terminating properly.

---

## 11. Objective — the two losses

```python
loss = loss_ce + loss_dt
```

**Cross-entropy (`loss_ce`)** — next-token prediction: at position *t*, predict token *t+1*.

**Time-to-next-event (`loss_dt`)** — modelled as an exponential:

```python
dt  = clamp(targets_age - age, min=1.0)   # days until next event
ldt = log(dt) - logits_dt                 # log of predicted rate
lse = log_sum_exp(logits_dt)
loss_dt = -(lse - exp(lse - ldt))         # negative exponential log-likelihood
```

This teaches the model not just *what* happens next but *when*.

**Why sequence modelling is meaningful here:** there is no grammar — but events are **age-ordered**
and clinical trajectories are strongly autocorrelated (prior state raises conversion risk;
comorbidities co-occur; age drives incidence).

---

## 12. Evaluation

Figure 2 — see §4 to run it, §5 to read it, and
[`Delphi-2M/figure2/FIGURE2.md`](Delphi-2M/figure2/FIGURE2.md) for panel-level method notes.

### 12.1 Why carry-forward cannot cheat the cognitive metrics

Because the cognitive scales are **keep-transitions**, a cognitive token in the stream is *always a
state change*. The target is literally "predict the conversion" (Normal→MCI, MCI→Dementia) — never
"repeat the current state". The naive cheat "next state = last state" scores ~0 **by construction**.
This is the direct payoff of the tokenizer design.

### 12.2 Validity concerns, ranked

1. **Direct label leakage** — ruled out by causal + same-visit masking (§9.2). *Fixed and verified.*
2. **Legitimate autocorrelation** — prior cognitive history raising risk is the *point*, not a
   cheat; quantify lift over a dumb "any decline yet? + age" baseline.
3. **Observation-pattern / informative-presence leakage — the one to chase.** Cognitive tokens
   appear only when a change is recorded, so their density correlates with monitoring intensity and
   cohort enrichment — *study-design signal, not physiology*.
4. **Age shortcut** — "old → Dementia" can inflate AUC without trajectory understanding; look for
   lift beyond age.

---

## 13. Decisions on record

Confirmed with the project owner; the evaluation implements these.

1. **"Worsening" = any ≥1-bin forward token transition.** As confirmed, this named
   *CDRSUM / MoCA / FAQ*. Two of those three no longer exist as tokens — the item split retired
   the CDRSUM and FAQ totals — so the sentence cannot stand as written. **⚠️ NOT RE-CONFIRMED:**
   the mechanical substitution is "the 6 CDR domains and the 9 FAQ domains in place of their
   totals", but whether the decision should now extend to the 12 NPI-Q symptoms and GDS — i.e.
   whether a single NPI symptom flickering counts as "worsening" — is a *new* question that was
   never put to the project owner. Do not read the broader version as agreed.
   NACCUDSD conversions are read as forward stage transitions, with **→MCI (108)** and
   **→Dementia (109)** called out separately; that part is untouched by the split, and it is what
   `figure2/` actually implements (it scores the 4 NACCUDSD states + Death, and never reads the
   other scales). Death is its own event. MoCA is handled so "worse" = higher token despite the
   inverted raw scale.
2. **Time-to-event via Monte-Carlo trajectory sampling, `n_mc = 100`** — not closed-form; forced by §10.
3. Conversion endpoints use **competing-risks CIF** (death competes), not naive Kaplan–Meier.
4. **`FASTOTAL` → renamed `FAQ`** everywhere (tokens 33–36 unchanged).

---

## 14. Known hazards

Two ways to lose work. Both are real; both were hit during development.

**`data_prep/tokenize_nacc_ad.py` and `make_split_ad.py` run on import.** Neither has an
`if __name__ == "__main__"` guard. Any `import make_split_ad`, IDE auto-import, doc tool or test
collector **regenerates `out_ad/`**, overwriting the tokenizer output and the splits. Both are
deterministic, so with the same CSV and seed the result is byte-identical — but on a machine where
the raw CSV differs or is missing this destroys the splits your checkpoint was trained on. Fixing it
means wrapping both scripts in a `main()`; that re-indents the whole file, so do it as its own
change and diff the regenerated `.bin` afterwards.

**`training/train.py --eval_only=True` overwrites `ckpt.pt`.** The save block runs on the first eval,
*before* the `if iter_num == 0 and eval_only: break`. `best_val_loss` starts at 1e9, so the first
eval always counts as "improved" and the checkpoint is written — even though you asked only to
evaluate. **Never point `--eval_only` at a directory holding a checkpoint you care about**; pass
`--out_dir=/tmp/scratch` instead.

---

## 15. Correctness audit

An internal audit swept the pipeline. All items below were **fixed**; recorded so nobody
re-introduces them.

**Training (`training/train.py`):**
- **fp16/bf16 AMP** — `torch.set_default_dtype` stays fp32 for `--dtype float16/bfloat16`, so params
  keep an fp32 master copy while autocast + `GradScaler` handle low precision (was: NaNs).
- **Resume** — the batch RNG is re-seeded `seed + iter_num`, so a resumed run continues a distinct
  batch stream instead of replaying iter-0's.
- **LR schedule** — `get_lr()` guards `lr_decay_iters == warmup_iters` (no div-by-zero).
- **Checkpoints** — `always_save_checkpoint` stores the true running `best_val_loss`.
- **Grad accumulation** — loss is divided by `gradient_accumulation_steps` (was: summed → effective
  LR ×steps).

**Clinical tokenization (`data_prep/tokenize_nacc_ad.py`):**
- **SMOKYRS** uses a numeric-missing set that excludes 8/9 (they are valid year counts).
- **VISITDAY** missing/blank falls back to mid-month (15) instead of dropping the whole visit.
- **FAST/FAQ** code **8** ("not applicable / never did") is recoded to 0 (no impairment) instead of
  dropping the visit. *Clinical convention — confirm with the data owner if the science depends on
  it; re-tokenizing with a different rule is a one-line change.*

**Generation (`model.generate`):** the rollout crops its conditioning to the last `block_size`
tokens; the dead `top_k` argument was removed.

Because the clinical fixes retain more tokens, event counts rose slightly (full `nacc_all.bin`
**1,142,310** vs the pre-fix 1,125,833); patient counts are unchanged.

---

## Citation

```bibtex
@software{delphi_gerstung,
  author  = {Gerstung Lab},
  title   = {Delphi: Medical Event Sequence Modeling},
  url     = {https://github.com/gerstung-lab/Delphi}
}
```
