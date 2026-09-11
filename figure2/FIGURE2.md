# Figure 2 — transitions, timing, individual trajectories, learned representation

Four-panel evaluation of **`out-radc-v3-final`** (204,929 params, L4/H4/E64, d0.1), scored on the
held-out **test** split of `radc-v3-s42`.

This is the NACC Figure 2 ported to RADC. Every panel asks the same question it asked there; what
changed is the instrument behind the question, and **`figure2/radc_states.py` is the single place
that mapping lives** — including the paragraph on where the mapping is exact and where it is a
substitution. Nothing else under `figure2/` hard-codes a token id.

## The mapping, in one table

| NACC | RADC | exact? |
|---|---|---|
| NACCUDSD staging: Normal / Impaired-not-MCI / MCI / Dementia | MMSE collapsed to the four **Folstein** stages: Normal (≥27) / Mild (24–26) / Moderate (18–23) / Severe (<18) | **substitution** — see below |
| Death, absorbing, competing risk | Death, same role | exact |
| panel b endpoint "Dementia" | the **AD diagnosis** token | *better* — RADC has a real incident diagnosis |
| panel b endpoint "Reach ≥MCI" | "Reach ≥Mild (MMSE ≤26)" | exact in role |
| trajectory class "Progressed to Dementia" | "Progressed to AD" | *better*, same reason |
| the 30 CDR/FAQ/NPI-Q/GDS/MoCA sub-scales in `perdomain.py` | RADC's 3 ordinal scales + 4 cumulative histories + 2 graded onset families + 4 medication events (28 tokens, 13 outcomes) | exact in role |
| `perdomain.py`'s reconstructed CDRSUM / FAQ / NPI-Q totals | **deleted, not replaced** | RADC never had an item split, so there is no retired token to reconstruct |
| `training_cohort_mask` replicating train.py's rule | calls `radc_delphi.batching.filter_cohort`, the real thing | *better* — the NACC copy had already drifted by one id |

**Where it is not exact, stated plainly.** NACCUDSD is a clinician's diagnosis; MMSE is a test
score. Panel a's row `Normal → Severe` reads "reached an MMSE below 18", not "was diagnosed as
demented". The choice is not free-hand: `vocab.STAGE_TOKENS_DISK` already declares MMSE the
staging analogue for this cohort, because RADC's per-visit diagnosis columns (`cogdx`,
`dcfdx_lv`) are last-visit only and both leak. The Folstein cuts land on edges the tokenizer
already has (`SCALE_EDGES["MMSE"]` contains 18.0, 24.0, 26.5), so a stage is a pure **grouping of
emitted tokens** computed at evaluation time — no retokenization, no new model, the `.bin`
untouched.

`radc_states.STAGE_MODE = "mmse7"` keeps all seven emitted levels instead. It is the honest
high-resolution view and it is what the per-token AUC work asks for, but panel a becomes a 7×8
grid in which almost every cell falls below the 30-event reporting floor.

## Which cohort — read this first

`filter_cohort` keeps a subject with ≥ `cohort_min_visits` distinct PREDICTED-EVENT ages, OR
≥ `cohort_short_min_visits` of them AND ≥2 staging tokens. `configs/radc_v3.py` sets both to 2.

**This matters much less here than it did on NACC.** There the filter cut train from 38,687 to
16,262 (42%) and the matched/full distinction moved the headline numbers by 2 AUC points. Here it
cuts 3,101 → 2,801 (90.3% kept), and on the test split it drops almost nobody. Both versions are
still produced, from the same Monte-Carlo cache, so the figure can say which population it
describes:

| version | files | what it answers |
|---|---|---|
| **matched** (the paper figure) | `*_matched.*` | the population train.py actually fitted on |
| full cohort | `*_allcohort.*` | the whole test split |

Reproduce (RIS compute2):

```bash
sbatch slurm/figure2.sbatch                    # MC cache + BOTH cohort versions (~2 min, 30 cores)
sbatch slurm/figure2.sbatch --limit 300        # smoke run
python figure2/figure2_panels.py --cohort matched      # re-plot only, cache reused
```

## How every predicted number is produced

Unchanged from the NACC version, and it is the part that makes the panels comparable to each
other. The model is autoregressive, so there is **no closed-form risk at a horizon**. Every
predicted quantity is a Monte-Carlo statistic over **100 sampled trajectories per subject**,
seeded on that subject's **first visit** (statics + the whole baseline visit; nothing after it).
Model access goes through `radc_delphi.engine`, which owns the three things that are easy to get
silently wrong: Death terminates the rollout, the No-event marker can never be sampled, and only
`REPEATABLE_TOKENS` may recur.

* **Competing risks are handled by construction.** A sampled trajectory that dies stops there, so
  "reaches Severe within 5 y" counts only when the sampled stage change precedes the sampled
  death. On the observed side the matching estimator is Aalen–Johansen, never 1−KM.
* **Right-censoring is explicit.** A subject whose follow-up ends before the horizon with no event
  has an *unknown* label and is dropped from AUCs (`labels_at_h` returns −1) rather than being
  silently counted as a non-event.
* **"Distinct visit age" excludes the statics.** They sit one day before the baseline visit, so
  counting every distinct age would make a one-visit subject look like it has two and would put
  the baseline prompt a day before any clinical content — the same off-by-one that made
  `filter_cohort` a no-op until it was fixed.

## The panels, and what they say on this model

**a — Transition-prediction accuracy.** Three cells. *a1* 5-year AUC per (baseline stage →
reached stage), subject-level bootstrap CI, shown only where ≥30 events and ≥30 non-events exist.
*a2* mean predicted risk vs the Aalen–Johansen cumulative incidence on the same at-risk cohort.
*a3* predicted vs observed **time** to the first transition out of the baseline stage, among
subjects who actually transitioned before dying.

> **5 of 20 transition types clear the 30-event floor** (NACC: 13 of 20). Median AUC **0.627**.
> `Mild→Normal` is the best row at **0.733** — the model's strongest cognitive call is a
> *recovery*. `Normal→Mild` **0.560** [0.497, 0.615] and `Normal→Death` **0.553** [0.505, 0.606]
> are barely above chance.
>
> **Death calibration is catastrophic and now quantified per transition:** `Normal→Death`
> observed 0.185 vs predicted **0.0035** (53×), `Mild→Death` observed 0.371 vs predicted
> **0.0115** (32×). This is the immortality failure, not a new finding, but a1/a2 localise it.
>
> **Timing: R² = −0.89, MAE 6.10 y, bias +5.28 y** — the model puts transitions five years too
> late on average. Spearman 0.490, so the ORDER is partly right and the absolute time is not.
> (NACC was R² −0.28, MAE 2.51 y, bias +0.47 y.)

**b — Timing accuracy by interval.** Error (predicted − observed) for the three endpoints, boxed
by how far ahead the event actually was. The grey dash in each box is the MAE of a **constant**
predictor (the cohort median time) — the honest baseline. The right cell is AUC vs horizon.

> **AD diagnosis is the only endpoint with real discrimination:** 0.849 / 0.865 / 0.838 / 0.777 /
> 0.712 at 1 / 2 / 3 / 5 / 10 y. Death is flat at 0.59–0.63. Reach ≥Mild is 0.65–0.72.
>
> **The constant predictor wins in every single timing bin.** Reach ≥Mild, 0–2 y: model MAE
> 9.65 y against a naive 4.49 y; 2–5 y: 8.73 vs 2.26. The model is adding no timing information
> anywhere on this cohort.

**c — Per-individual trajectory comparison.** Observed vs MC-modal stage on a yearly grid, years
1–15 after baseline (year 0 excluded: both series equal the baseline stage there by construction).
A grid point counts only while the subject is under observation, or once they are known dead.
Reported: Jaccard per subject, the same Jaccard for a **carry-baseline-forward** reference (the
trivial "nothing changes" predictor), per-stage IoU, and agreement by year.

> **This is the headline result, and it is bad.** Model Jaccard **0.4244** [0.396, 0.453] against
> carry-baseline-forward **0.4230** — a lift of **+0.0014**. On NACC the same comparison was
> 0.618 vs 0.523, a lift of +0.095. The agreement-by-year series confirms it is not a
> ceiling artefact: the two curves are within 0.006 of each other at every one of the 15 years.
>
> **On this cohort the trajectory panel cannot distinguish the model from "assume nothing
> changes."** Per-stage IoU: Normal 0.508, Mild 0.184, Moderate 0.072, Severe 0.055, Death
> **0.001** (macro 0.164).

**d — Patient-embedding structure.** UMAP of the final-block hidden state (post `ln_f`) for the
baseline prompt, coloured by observed trajectory class. The headline number is **10-NN label
purity in the original 64-d space**, not in the 2-d projection.

> Trajectory purity **0.326** against chance 0.253. But baseline-STAGE purity is **0.844**
> against chance 0.651, and restricting neighbours to the same baseline stage leaves purity
> **0.315** against chance 0.264 — a lift of **+0.051**, almost exactly NACC's +0.050. The
> embedding encodes where you are now; it barely encodes where you are going.
>
> RADC adds a confounder NACC did not have: the `Dementia at entry` static (6.5% of subjects) is
> in the baseline prompt, so the embedding can read prevalent disease directly. That is
> background, not leakage, but it is another reason the raw purity overstates prognosis.
>
> `full_history_LEAKY` purity 0.734 is recorded and **not plotted**: `traj_class` is defined by
> tokens that are in that embedding's own input, so separation there measures memory.

**The projection is optional.** `umap-learn` is declared in `environment.yml`; when it is absent
panel d falls back to a deterministic 2-component PCA and says so in the title, the axis labels,
the CSV column names and `metrics.json`. The claim is the purity number, the scatter is for
looking at. A figure that silently relabelled PCA as UMAP would be worse than one without a
scatter.

## Caveats to carry with these numbers

1. **Death is not predictable from this stream and every survival-derived number is unusable.**
   Not a caveat about precision — the one-step P(Death next) has median 0.0003 against death being
   8.4% of observed post-85 tokens, and a2's 53× under-prediction is the same defect measured a
   different way.
2. **Discrimination is much better than time-calibration.** Panels a1 / b-right (ranking) and
   a3 / b-left (absolute timing) are two separate claims, and only the first survives.
3. **Panel a3 and panel b-left both condition on the event being observed.** They are silent
   about subjects who never had the event, by design.
4. **The staging is an MMSE score, not a diagnosis** (see the mapping table). Panel b's AD
   endpoint is the only diagnosis in the figure.
5. **The cohort is a volunteer ageing cohort**, not a population sample; absolute rates here are
   not incidence.
6. **The supplementary transition matrix is not cohort-matched.** It is summed over subjects when
   the cache is built, so it cannot be subset afterwards and shows the full evaluable test set in
   both versions. The figure says so in its title.
7. **Self-transitions are dropped from the transition matrix.** A RADC stage is a GROUP of MMSE
   levels, so a 30 → 28 move is a real token pair inside one stage; counting it would put mass on
   the diagonal that no state change produced. The NACC stream could not contain such a pair.
