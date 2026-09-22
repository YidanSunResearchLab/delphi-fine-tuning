# AD-projection — RADC/ROSMAP arm

A Delphi-2M-style generative event-sequence model for Alzheimer's disease trajectory
projection, trained on the Rush ROS / MAP / LATC cohorts.

The model reads a subject's clinical history as an age-ordered stream of tokens and learns two
things jointly: **which event comes next**, and **how long until it happens**. Forward-sampling
from a truncated history then gives a Monte-Carlo distribution over that subject's possible
futures, from which any horizon risk — P(dementia by 85), time to diagnosis, survival — falls
out as a statistic over sampled trajectories rather than a closed-form formula.

This is a RADC-only pipeline. The NACC arm that previously shared this repository has been
deleted; see the commit that introduced `radc_delphi/` for what went and why.

---

## 1. Layout

```
radc_delphi/
    vocab.py           the 50-token table — the single source of truth for token ids
    tokenizer.py       the three raw RADC files -> an age-ordered event stream
    splits.py          subject-level stratified train/val/test partitioning
    build_dataset.py   CLI: tokenize -> split -> data/radc-s<seed>/
    model.py           the Delphi transformer (vocabulary-agnostic)
    batching.py        subject indexing and batch assembly
    engine.py          inference: next-token probabilities, MC rollout, Aalen-Johansen
    decode_codebook.py makes the RADC codebook PDF readable
train.py               the training loop
sweep.py               capacity / regularisation sweep, scored by k-fold CV
configs/radc_base.py   the delivered configuration
eval/                  cohort definition, baselines, endpoint scoring, leakage audit
slurm/                 RIS submission scripts and the code-only deploy
tests/test_pipeline.py 46 regression checks
```

The four directories below are the ROSMAP arm: a separate line of work that fine-tunes the
*upstream* Delphi-2M implementation on ROS/MAP rather than using `radc_delphi/` above. It has its
own tokenizer, its own checkpoints and its own evaluation scripts, and shares nothing with the code
above except the modelling idea. Kept here so the two arms can be compared from one place.

```
tokenization/
    spec.py            the ROSMAP variable spec: which columns, which bins, what gets excluded
    build.py           the two gk spreadsheets + ROSMAP_clinical.csv -> data_rosmap*/{train,val}.bin
delphi/                vendored gerstung-lab/delphi @ fb72166 (2026-08-07), plus the ROSMAP arm:
    train.py, utils.py               carry a local diff vs fb72166 (+17/-3): ROSMAP data loading
    config/train_delphi_rosmap*.py   ROSMAP training configs (dedup and nodedup variants)
    evaluate_auc_rosmap.py           per-token AUC on the ROSMAP val split
    ris/                             RIS submission scripts
    Delphi-ROSMAP*/auc/              scored AUC tables for the three runs
figure2_eval/          Figure 2 for the ROSMAP arm, plus the diagnostic probes
    radc_delphi/       a fork of the engine above, drifted; do not import across arms
    probe_*.py         one probe per failure mode (calibration, shift, ties, perturbation, ...)
    next_visit_auc.py  next-visit AUC, the headline number
aladynoulli_rosmap/    Aladynoulli latent-signature baseline, k-fold CV against the Delphi states
```

Everything these scripts read lives under an ignored `data/` or `data_rosmap*/` directory and is
*not* in this repository — see `tokenization/README.md` and `figure2_eval/README.md` for how to
rebuild it from the RADC source files. `figure2_eval/data/` is two symlinks into
`delphi/data/`; recreate them after cloning.

Two token spaces, and confusing them is the classic bug in this family of code:

| space | where | 0 | 1 | content |
|---|---|---|---|---|
| **model** | what the model sees; the `labels.csv` row index | Padding | No event | 2–49 |
| **disk** | what is in the `.bin` files (`data[:, 2]`) | — | — | 1–48 |

`disk = model - 1`. The `+1` is applied in exactly one place, `batching.get_batch`.

## 2. Running it

```bash
conda env create -f environment.yml && conda activate ad-projection

python -m radc_delphi.build_dataset --seed 42      # -> data/radc-s42/
python train.py configs/radc_base.py --device=cuda # -> out-radc-base-s42/ckpt.pt
python tests/test_pipeline.py                      # 46 checks
```

On the RIS cluster:

```bash
./slurm/deploy.sh --submit            # sync code only, then sbatch
sbatch slurm/sweep_radc.sbatch        # the capacity sweep
```

`deploy.sh` refuses to send `data/`, `*.bin`, `*.pt`, `*.xlsx` or the pid map. The raw files
live on the cluster in a `0700` directory, uploaded once; nothing derived from them travels
back.

## 3. The data

| file | shape | what it is |
|---|---|---|
| `cross-sectional-data-gk.xlsx` | 4,428 × 24 | one row per subject: statics, baseline age, AD diagnosis age, autopsy pathology |
| `longitudinal_data_gk.xlsx` | 36,138 × 29 | one row per (subject, follow-up year) |
| `ROSMAP_clinical.csv` | 3,584 × 18 | last-visit summary; used for **one** column, `age_death` |
| `RADC_codebook_*.pdf` | 29 pages | authoritative variable definitions and value codings |

The codebook PDF embeds Type3 fonts with no `/ToUnicode` map and a separate glyph subset per
page — 29 different substitution ciphers. `radc_delphi/decode_codebook.py` hashes each glyph's
drawing procedure to unify them into one alphabet and solves it from known-plaintext cribs.

**Governance.** The raw files are controlled-access research data under a RADC data-use
agreement. `.gitignore` excludes `data/`, every `.bin`, and `radc_pidmap.csv` — the
`projid → integer pid` linkage, which carries the same re-identification risk as the raw files.

## 4. The tokenizer

50 tokens, **74,900 events over 4,428 subjects** (16.9 per subject; median 16, max 42).

### The age axis

The files carry no visit date and no per-visit age. The only anchors are `age_bl` and
`fu_year`, so

```
age_at_visit_days = round(age_bl * 365.25) + 365 * fu_year
```

Two details in that formula are load-bearing and both were measured, not assumed.

**`fu_year` is an elapsed-year index, not a visit counter.** 30.5% of subjects skip at least
one follow-up year, and those gaps are genuinely missing rows (only 8 of 36,138 rows are
entirely empty). For subjects with exactly one skipped year, the discrepancy against
ROSMAP_clinical's last-visit age has median **−0.007 y** using the index and **+0.993 y** using
the row count.

**The grid step is 365 days, not 365.25.** Rounding an integer-year index times 365.25 makes
the inter-visit delta cycle 365/365/365/366 with a phase fixed per subject — a perfectly
learnable arithmetic artifact sitting in the exact day-level channel the waiting-time head
reads. Only the two genuinely fractional ages (the diagnosis, and a numeric age at death) use
365.25.

*Accuracy ceiling:* residual error against the true visit age is within-year jitter, not drift
— median ≈21 days, ~8% of last visits slip past 6 months, ~2% past a year. **Every timing claim
this model makes inherits that floor.**

### Dedup — five regimes

| regime | tokens | rule |
|---|---|---|
| static | 2–22 | one per subject, at `age_bl − 1 day` |
| hysteresis keep-transitions | 23–35 | MMSE, global cognition, BMI |
| keep-first | 36–39, 48, 49 | monotone `_cum` histories, the diagnosis, death |
| onset-transitions | 40–43 | stroke, depression |
| start / durable-stop | 44–47 | antihypertensives, statins |

**Statics sit one day before baseline, and the offset is not cosmetic.** `mask_ties` blocks a
position from attending to any token sharing its target age, so statics placed *at* `age_bl`
would be invisible when predicting the baseline visit — roughly a fifth of all scored
positions.

**Asymmetric hysteresis.** Plain keep-transitions on the cognitive scales is mostly measurement
noise: measurement SD is ~1.2–1.6 MMSE points and ~0.18–0.21 z, 41% of MMSE and 54% of
cognition bin changes vanish under smoothing, and 35–51% of "recoveries" immediately
round-trip. So a **decline is emitted the moment the edge is crossed** (never delayed), while a
**recovery must clear the edge by one measurement SD** (2.0 MMSE points, 0.20 z, 1.0 kg/m²).

Round-trip churn in the delivered stream, measured as returns to an already-visited,
since-left level per emitted scale token: **MMSE 14.2%** (1,343 / 9,476) and **global
cognition 20.3%** (2,358 / 11,604), against ~29–31% without hysteresis. Those are the
reference rates a generated trajectory is compared against — a model producing substantially
more is oscillating dementia, which is a reportable failure and not a rounding issue.

Lookahead confirmation ("emit only if the change persists next visit") filters noise slightly
better and is deliberately **not** used: it places a token at age *t* using information from
*t+1*, which is future information flowing backwards in a causal model.

### What is excluded, and why

| excluded | measured reason |
|---|---|
| autopsy block (13 columns) | availability is a **perfect death oracle** — all 2,231 subjects with any autopsy value have `died == 1`; a missingness indicator alone reaches AUC **0.906** for death. The token's *existence* is the leak, so no placement rule rescues it. |
| `cogng_demog_slope` | AUC **0.775** for AD among subjects normal at baseline and diagnosed ≥10 y later. It is the outcome. |
| `ad_rx` | at a normal MMSE of 27–30 it lifts P(AD dx within 3 y) from **4.7% to 31.2%** — it encodes the clinician's diagnosis before `age_first_ad_dx` records it. Reserved behind a flag for the ablation. |
| `cogdx`, `dcfdx_lv`, `cts_mmse30_lv` | end-of-study columns; AUC 0.888 / 0.838 / 0.833 by construction. |
| hba1c, LDL, PSQI, Berlin, eGFR, HDL, glucose, BP | ~12 tokens per 17-token sequence for almost no AD signal (LR χ² ≤ 8.5 against BMI's 72.8), and largely a study indicator in disguise — PSQI is 99.7% present in MAP and 26.1% in ROS; hba1c is exactly 0% before ~2004. |
| inflammation trio | 1.2% coverage: a one-shot 426-subject MAP substudy. |

### Death

Emitted for **all 2,745 decedents**, not just the 975 with an uncensored `age_death`.

Those 975 are *exactly* the sub-90 deaths (max numeric value 89.993, zero at ≥90), so emitting
only them would teach the model that death after 90 is impossible — and would leave LATC with
zero death tokens out of 27 deaths. The 1,770 others are placed at
`max(90 if "90+", last visit + 0.68 y)`, where 0.68 y is the measured median lag between the
nominal last visit and the true death age (IQR 0.34–0.98). The resulting distribution has
median 89.7 y and 46.2% at 90+, against 47.4% observed among deaths carrying any age.

**Ignoring death is not an option here.** 53.3% of non-converters died, and treating death as
censoring inflates cumulative AD incidence by **+38.6% at age 85 and +59.4% at 90**.

## 5. The model

Delphi-2M: a causal transformer over event tokens with **no positional embedding** — a
continuous sinusoidal encoding of age in days replaces it — trained on two losses:

- `loss_ce` — cross-entropy over the next token
- `loss_dt` — exponential-waiting-time log-likelihood for when it happens

`mask_ties` blocks attention between tokens sharing an age, so same-visit tokens cannot leak
into each other's predictions.

**A caveat that must travel with every timing number.** RADC is an annual-protocol cohort:
93.9% of nominal inter-visit intervals are exactly one year. `loss_dt` therefore learns "the
next observation arrives in about a year" — the correct data-generating process for
*observations*, but almost no individual signal. Only two tokens sit on a real clock: the
diagnosis and death. `time_head = True` gives the timing objective its own scalar projection so
that degenerate target stops pulling on the token logits.

## 6. Evaluation

`eval/cohort.py` defines three populations, once, for everything:

| population | n | what it is for |
|---|---|---|
| all | 4,428 | every descriptive statistic |
| training | 4,029 | ≥2 visits — a single-visit subject cannot support a next-event objective |
| evaluation | 3,691 (1,051 converters) | training **and** not baseline-impaired |

**Baseline impairment** (`MMSE < 24` or `ad_rx == 1` at `fu_year 0`; 400 subjects) exists
because the codebook is explicit that `age_first_ad_dx` "is not available for participants that
were demented at baseline cycle". Left alone, several hundred prevalent-dementia subjects sit
in the negative class spending whole trajectories at MMSE 15–23 — actively teaching the model
that severe impairment is compatible with being AD-free. They are **excluded, not relabelled**:
entering them as baseline events would move the calibration target from 0.177/0.281 to
0.255/0.349.

Every incidence quantity is **cause-specific Aalen-Johansen with left truncation at `age_bl`**,
never 1 − Kaplan-Meier. Observed AD cumulative incidence on the evaluation cohort:

| age | CIF | risk set |
|---|---|---|
| 75 | 0.0360 | 1,008 |
| 80 | 0.0953 | 1,414 |
| 85 | 0.1774 | 1,531 |
| 90 | 0.2809 | 986 |
| 95 | 0.3611 | 361 |

### The bar

Measured by `eval/baselines.py` on the evaluation cohort, pooled train+val, 10 seeds × 5-fold
subject-level CV, with 1,000-draw subject-clustered bootstrap CIs. No model is involved.

| baseline | what it uses | outcome | AUC |
|---|---|---|---|
| B1 statics-only | the 21 static tokens + `age_bl` | ever-AD | 0.691 ± 0.002 |
| | | death | 0.869 ± 0.001 |
| B2 baseline prefix | every token at or before `age_bl` — the model's own prefix | ever-AD | 0.721 ± 0.003 |
| | | death | 0.878 ± 0.001 |
| **B3 concurrent cognition** | **current MMSE bin + cognition bin + age + sex + education, at the landmark** | **AD ≤ 5 y, 1 y lead** | **0.885** |
| | | AD ≤ 5 y, 0 y lead | 0.907 |
| B4 continuous ceiling | B3 with un-binned scores | AD ≤ 5 y, 1 y lead | 0.891 |

B1 and B2 are *ever*-AD from a fixed baseline prefix; B3 and B4 are horizon risk from a moving
landmark. They are not comparable to each other — **B2 is the ceiling the leakage audit checks
against, B3 is the bar the model has to clear.**

Two things that change these numbers and must therefore be quoted with them:

- **Lead time.** 40.6% of converters have their last legal landmark within three months of the
  diagnosis (median lead 0.51 y), because RADC records the diagnosis age at a visit. At zero
  lead a "5-year" AUC is partly a *concurrent-diagnosis* AUC. B3 runs 0.907 → 0.895 → 0.892 →
  0.885 at leads of 0 / 0.25 / 0.5 / 1 y.
- **Cohort.** Excluding the baseline-impaired raises ever-AD and lowers the death AUCs by
  0.006–0.009, because it removes a small, sick, high-mortality group. On the unfiltered
  cohort the same code gives 0.684 / 0.877 / 0.718 / 0.883.

`B4 − B3 = +0.006 to +0.007` at every horizon and lead: that is the information cost of
binning the cognitive scores, and it is the ceiling the sequence model should approach rather
than exceed — it sees the bins, not the raw values.

**B3 is the real bar.** If the sequence model does not beat a logistic regression on the two
cognitive tokens it can already see at the landmark, then the transformer, the age axis and the
whole event stream are adding nothing, and the honest deliverable is the logistic.

### Leakage audit

`eval/leakage.py` runs on every checkpoint before any number is believed:

1. **Baseline-prefix ceiling** — AUC from an `age_bl`-truncated prefix, against B2's legal
   ceiling. Triggers above 0.76 (AD) or 0.92 (death). Known signatures: ~0.87 for AD is
   `cogng_demog_slope`; ~0.97 for death is autopsy availability.
2. **Horizon curve** — model minus a legally-refit logistic at T−1, −3, −5, −8, −12 years.
   Genuine antecedent signal grows as the horizon shortens; a static leak is flat and already
   elevated.
3. **Canary** — a tokenizer flag that deliberately leaks eventual AD status for a random 5% of
   subjects. Tests 1 and 2 must fire on those subjects and not on the rest. *A null result from
   an untested detector is worthless.* The production build asserts the flag off.

## 7. Results — the delivered model

`out-radc-final/ckpt.pt`. 4L / 4H / 64d, **204,545 parameters**, block 64, `time_head=True`,
`dt_target="next_event"`. Architecture chosen by a 75-run 5-fold-CV sweep (§ `configs/`).
Trained on 2,801 subjects, best validation 8.6699 at iteration 5,400 of 6,000.

**The test split was opened once**, after the architecture and every hyperparameter were frozen.

### Leakage audit: PASS

| test | model | legal ceiling | trigger | verdict |
|---|---|---|---|---|
| baseline-prefix ever-AD | 0.6441 [0.599, 0.686] | 0.7260 (B2) | 0.76 | PASS |
| baseline-prefix death | 0.6500 [0.613, 0.693] | 0.8853 (B2) | 0.92 | PASS |
| horizon curve | long-lead gap −0.194 [−0.270, −0.113] | +0.10 | | PASS |

The model sits **below** the legal ceiling on both prefix readouts, and its horizon curve
*gains* 0.176 AUC between the longest lead and one year — the signature of genuine antecedent
signal rather than a static leak. 22/22 detector self-checks pass, including a planted-oracle
canary. The tokenizer is clean.

### Primary endpoint: incident AD, death competing

| H | lead | subjects scored | positives | model AUC | B3 baseline | Δ |
|---|---|---|---|---|---|---|
| 1 y | 0 y | 702 | 164 | 0.9082 [0.886, 0.928] | 0.9356 | −0.027 |
| 3 y | 0 y | 623 | 416 | 0.8885 [0.865, 0.910] | 0.9129 | −0.024 |
| 5 y | 0 y | 552 | 597 | 0.8737 [0.850, 0.893] | 0.8932 | −0.020 |
| 3 y | 2 y | 593 | 115 | 0.8217 [0.780, 0.861] | 0.8525 | −0.031 |
| **5 y** | **2 y** | **522** | **296** | **0.8108 [0.776, 0.844]** | **0.8293 [0.799, 0.858]** | **−0.019** |

**The model does not clear its bar.** It loses to the concurrent-cognition logistic by
0.019–0.031 AUC at every horizon and every lead. The confidence intervals overlap heavily, so
this is not a decisive defeat — but the criterion set before the run was that the transformer
must *beat* B3, and it does not. On this evidence the honest deliverable for a pure
AD-within-H-years risk score is the logistic regression, not this model.

Calibration at H=5 is monotone across all ten deciles (mean predicted 0.0004 → 0.6251 against
Aalen-Johansen observed 0.0233 → 0.6424) but **under-predicts overall**: mean predicted 0.1394
against 0.1997 observed, mean absolute decile gap 0.0627.

Next-token cross-entropy on test: `loss_ce` **2.1596**, `loss_dt` 6.5205 (not a
timing-calibration result — see §5).

### The generative side is broken, and this is the main finding

| from age-80 prefixes | simulated | observed (same subjects, left-truncated at 80) |
|---|---|---|
| P(AD by 85) | 0.0918 | 0.1269 |
| P(AD by 90) | 0.1797 | 0.2133 |
| **P(death by 85)** | **0.0083** | **0.2463** |
| **P(death by 90)** | **0.0110** | **0.5338** |

**98.6% of sampled trajectories never die.** They run to the age-110 cap, median simulated age
108.8 y. Diagnosed directly: one-step `P(Death next)` has median **0.0003** across
under-85 prefixes while Death is **8.4%** of observed post-85 tokens — the mass is not merely
low, it is concentrated on a handful of prefixes (max 0.48) and absent everywhere else.

The mechanism is a design consequence, not a bug. Death is the **last** token of every stream
that has one, so the model learned it from a position cue — "the record is about to stop" —
and free generation supplies no such cue. The specification explicitly rejected an
end-of-follow-up token on the grounds that `dt_target="next_event"` already right-censors
correctly. That reasoning is right for *training* and wrong for *generation*: without a token
that marks approaching end of observation, the simulated cohort is immortal, and every
survival and cumulative-incidence number derived from rollouts is unusable.

This does not invalidate the primary AUC, which is a *ranking* over a 5-year horizon where
immortality bites least, and which the competing-risk labelling handles on the observed side.
It does invalidate Secondary 1 and 2 as reported.

Trajectory realism is directionally right but too noisy: simulated round-trip rates are
**0.215 (MMSE)** and **0.282 (cognition)** against observed 0.143 and 0.204 — the model
oscillates about 50% more than the data does.

### What to do next, in priority order

1. **Give death an observable antecedent.** Either a typed end-of-follow-up token (the option
   the spec rejected) or an explicit per-step hazard head. Until then, do not quote any
   simulated survival number.
2. **Close the 0.02 AUC gap to B3, or stop.** The obvious candidates are the ones this build
   deliberately cut: the biomarker block, and `min_lead`-aware training. If a fair rematch
   still loses, the logistic is the answer and the interesting result is *that* — a 4,428-subject
   cohort with two cognitive scales does not support a sequence model over a tabular one.
3. Run the pre-registered ablations: death token on/off, the 0.68 y vs 1.5 y imputation,
   hysteresis on/off, study token on/off, BMI on/off, biomarkers back in.

## 8. Honest limitations

- **The cohort is not the population.** ROS/MAP/LATC are volunteer cohorts: 73.2% female,
  median 16 years of education, ~83% non-Hispanic white, all consented to autopsy. Risk
  estimates from this model should not be applied to general or under-educated populations.
- **Timing is grid-limited.** Visit ages are nominal; ~21-day median error, ~2% of last visits
  off by more than a year.
- **The AD label is incident-only.** Prevalent-dementia subjects have no `age_first_ad_dx` by
  construction and are excluded from scoring rather than relabelled.
- **Death ages are partly imputed.** 1,770 of 2,745 death ages come from the +0.68 y rule.
  A pre-registered ablation at 1.5 y exists to show the AD calibration is insensitive to it.
- **`loss_dt` is not a timing-calibration result** on this cohort. See §5.
- **Simulated survival is unusable** as delivered — see §7. Rollouts almost never emit death.
- **The sweep predates the cohort-filter fix.** All 75 runs saw 3,544 pooled subjects rather
  than the 3,199 the fixed filter keeps; the filter was counting the static block's own age
  and so passed everybody. Every cell saw the same population, so the ranking stands, but the
  absolute CV losses are not comparable to the delivered run's.
- **The three sub-studies are different hazard regimes.** LATC has 8.8% mortality against ROS's
  72.3%; a pooled number is a weighted average of two populations. Report stratified by study.
