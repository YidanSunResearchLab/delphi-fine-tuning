# Figure 2 — transitions, timing, individual trajectories, learned representation

Four-panel evaluation of **`out-delphi2m-dedup-mask-s42`** (dedup / keep-transitions + `ignore_tokens`
0–21), scored on the held-out **test** split of `nacc-dedup-s42`.

## Which cohort — read this first

`train.py` does not fit on every patient in `train.bin`. `filter_cohort` (train.py:106, defaults
`cohort_min_visits=4` / `cohort_short_min_visits=2`, not overridden by the config) keeps a subject
only if it has **≥4 distinct visits, OR ≥2 visits and at least one NACCUDSD transition**. The
training log confirms it: `train 38687 -> 16262 | val 5526 -> 2349`. The model was fitted on 42% of
the train split.

So the figures come in two versions, from the **same** Monte-Carlo cache:

| version | files | n | what it answers |
|---|---|---|---|
| **matched** (the paper figure) | `*_matched.*` | **4,589** | how the model does on the population it was actually trained for |
| full cohort | `fig2*_allcohort.png`, `metrics_allcohort.json`, `SUMMARY_allcohort.md` | 7,989 | generalisation — 43% of these patients (2 visits, median 1.95 y follow-up, 45% already demented) are outside the training regime |

Neither has leakage: the split is by patient and test never touched training. The difference is
purely *which* test patients get scored. Reading the two side by side is informative in itself —
see "What changes between the two" below.

| file | what it is |
|---|---|
| `figure2_combined_matched.{png,pdf}` | the assembled four-panel figure (`_allcohort` = the other cohort) |
| `fig2a_matched…fig2d_matched.{png,pdf}` | the same panels standalone (identical code path, larger) |
| `fig2*_data.csv` | the exact numbers behind every mark in that panel |
| `fig2a_transition_time_data.csv` | per-patient observed vs predicted transition time (panel a3) |
| `fig2a_supp_transition_matrix.*` | companion one-step transition matrix — see the caveat below |
| `metrics_matched.json` | every reported statistic, machine-readable |
| `SUMMARY_matched.md` | narrative version, auto-generated |
| `_cache/` | Monte-Carlo cache keyed by (checkpoint, split, n_mc) — delete to force a rebuild |

Reproduce (RIS compute2, from `AD0718/Delphi-2M/`):

```bash
sbatch slurm/figure2.sbatch                    # MC cache + BOTH cohort versions (~25 min, 32 cores)
sbatch slurm/figure2.sbatch --limit 300        # smoke run
python figure2_panels.py --cohort matched      # re-plot only, cache reused (~70 s)
python figure2_panels.py --cohort all
```

## How every predicted number is produced

The model is autoregressive, so there is **no closed-form risk at a horizon**. Every predicted
quantity here is a Monte-Carlo statistic over **100 sampled trajectories per patient**, seeded on
that patient's **first visit** (birth-time statics + the whole baseline visit; nothing after it).
That prompt is the entire input — no future leakage — and it is also what makes the panels
comparable to each other.

Two consequences worth stating out loud:

* **Competing risks are handled by construction.** A sampled trajectory that dies stops there, so
  "reaches Dementia within 5 y" is counted only when the sampled dementia precedes the sampled
  death. On the observed side the matching estimator is Aalen–Johansen, not the naive event
  proportion.
* **Right-censoring is explicit.** A patient whose follow-up ends before the horizon with no event
  has an *unknown* label and is dropped from AUCs (`labels_at_h` returns −1), rather than being
  silently counted as a non-event — which is what would bias the rates upward.

## The panels

**a — Transition-prediction accuracy.** Three cells.
*a1* AUC for reaching each target state within 5 y, one row per (baseline state → state reached),
patient-level bootstrap CI, shown only where at least 30 events and 30 non-events exist.
"from → to" means *reaching* `to` by any path, so Normal→Dementia includes patients who passed
through MCI; the strictly one-step version is the supplementary transition matrix.
*a2* mean predicted risk vs the Aalen–Johansen cumulative incidence on the same at-risk cohort.
*a3* predicted vs observed **time** to the first transition out of the baseline state, among
patients who actually transitioned before dying, with R², MAE, RMSE and the 2-year binned median
trend. Note a3 conditions on a transition having happened on *both* sides — it asks only whether
the timing is right, never whether the transition occurs (that is a1/a2).

**b — Timing accuracy by interval.** Error (predicted − observed) for three endpoints, boxed by how
far ahead the event actually was. The grey dash in each box is the MAE of a *constant* predictor
(the cohort median time) — the honest baseline. A model that beats it in a bin is adding
information there; one that does not is not. The right cell is AUC vs horizon for the same
endpoints.

**c — Per-individual trajectory comparison.** Observed vs MC-modal state on a yearly grid,
years 1–15 after baseline (year 0 is excluded: both series equal the baseline state there by
construction and would inflate every score). A grid point counts only while the patient is under
observation, or once they are known dead. Reported: Jaccard per patient against the observed
series, the same Jaccard for a *carry-baseline-forward* reference (the trivial "nothing changes"
predictor), per-state IoU, and agreement by year. The **alive-years-only** Jaccard is reported
alongside because once a patient dies every remaining grid year is "Death", so one correct death
call can carry an otherwise poor cognitive trajectory. Four illustrative patients are chosen
deterministically — two the model matches, one it over-predicts, one it under-predicts.

**d — Patient-embedding structure.** UMAP of the final-block hidden state (post `ln_f`) for the
baseline prompt and for the full observed history, coloured by observed trajectory class.
The headline number is **10-NN label purity in the original 120-d space**, not in the 2-d
projection — the projection is for looking at, the purity is the claim. Chance (the label prior)
is drawn as a dashed line; a bar near that line means the embedding does not encode that label.

## What changes between the two cohort versions

| | matched (4,589) | full (7,989) |
|---|---|---|
| median AUC across transition types | **0.696** | 0.678 |
| Dementia AUC @5 y / @10 y | **0.871 / 0.832** | 0.852 / 0.791 |
| Death AUC @2 y / @5 y | 0.679 / 0.779 | **0.776 / 0.803** |
| trajectory Jaccard (model) | 0.626 | 0.625 |
| trajectory Jaccard (carry-baseline reference) | **0.510** | 0.461 |
| 10-NN trajectory purity / chance | 0.606 / 0.215 | 0.651 / 0.288 |

Three things this comparison actually tells you:

1. **Cognitive-transition discrimination genuinely improves** on the matched cohort, and the event
   counts for those rows are *unchanged* (MCI→Dementia 620 in both, Normal→MCI 432 in both). That is
   not a coincidence: the filter's second clause is "≥2 visits AND a transition", so every patient
   with an observed cognitive transition is already inside the training cohort. The matched cohort
   removes almost only non-transitioners.
2. **Death prediction looks worse when matched** (AUC@2y 0.776 → 0.679), and that is the honest
   number. The full cohort is padded with short-follow-up, already-demented patients who die soon —
   easy calls that flatter the model. Death event counts collapse accordingly (Dementia→Death
   1,058 → 292). `Impaired→Death` falls below the 30-event threshold and disappears from panel a.
3. **The trajectory panel's headline margin shrinks.** Model Jaccard is flat (0.625 → 0.626) but the
   trivial carry-baseline-forward reference rises 0.461 → 0.510, so the model's advantage over
   "assume nothing changes" goes from +0.164 to **+0.116**. Quote the margin, not the raw Jaccard.

Panel a3 (transition timing) is **identical** in both versions — its cohort is "patients who
transitioned before dying", all of whom are inside the training cohort by construction. A useful
check that the filter does what it claims.

## Caveats to carry with these numbers

1. **Dedup makes the observed series sparse.** The data is keep-transitions: a recorded state
   sequence is a sequence of *changes*, and NACCUDSD is recorded at only ~35% of visits. The
   observed "stage-at-age" grid in panel c is therefore a carry-forward reconstruction, not a
   per-visit measurement, and the diagonal of the transition matrix is 0 by construction.
2. **Discrimination is much better than time-calibration.** This matched the earlier 5-figure evaluation sweep
   (removed 2026-07-31), where event times also came out biased late by several years. Read
   panel a1/b-right (ranking) and panel a3/b-left (absolute timing) as two separate claims.
3. **Panel a3 and panel b-left both condition on the event being observed.** They are silent about
   patients who never had the event, by design.
4. **The cohort is memory-clinic-enriched**, not a population sample. Absolute rates here should not
   be read as incidence.
5. **The supplementary transition matrix is not cohort-matched.** It is summed over patients when
   the cache is built, so it cannot be subset afterwards and shows the full evaluable test set in
   both versions. The figure says so in its title. Fixing it properly means storing per-patient
   matrices and re-running the 22-minute MC pass — not worth it for one supplementary panel.
