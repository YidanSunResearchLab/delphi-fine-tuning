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

**a — Per-state prediction accuracy.** Three cells. *a1* 5-year AUC for REACHING each state or
scale level, one number per row, subject bootstrap CI, shown only where ≥30 events and ≥30 non-events
exist. At risk: subjects not already in that state at baseline (Death and AD cannot be present
at baseline in this stream — prevalent AD is a STATIC, not an event — so everyone is at risk of
the incident version). *a2* mean predicted risk vs the Aalen–Johansen cumulative incidence on
the same at-risk cohort. *a3* predicted vs observed **time** to the first transition out of the
baseline stage, among subjects who actually transitioned before dying.

> **WHY THIS IS NOT A TRANSITION GRID ANY MORE.** a1 used to report a (baseline stage → reached
> state) grid: 20 cells of which 5 cleared the event floor. Two things were wrong with it as a
> headline. Its row labels read as token adjacencies — "Mild→Death" looks like "the Death token
> follows the Mild token", when the row actually meant *among subjects whose BASELINE stage was
> Mild, reaching Death by any path within the horizon*. And splitting one question by where the
> subject started turned a single adequately-powered number into three underpowered ones. The
> stratified version is still computed (`compute_A_strat`) and written to
> `fig2a_stratified_*_data.csv` and `metrics.json` under `per_transition_stratified`, because it
> is the only place the figure says whether discrimination depends on the starting stage.

| row | at risk | events | **AUC** | 95% CI | observed | predicted |
|---|---|---|---|---|---|---|
| **cogn_global >0.8** | 728 | 71 | **0.823** | [0.782, 0.866] | 0.103 | 0.096 |
| MMSE Severe (<18) | 780 | 43 | **0.809** | [0.741, 0.873] | 0.060 | 0.026 |
| cogn_global ≤−2.0 | 782 | 30 | **0.806** | [0.724, 0.892] | 0.042 | 0.013 |
| AD diagnosis | 793 | 95 | **0.777** | [0.725, 0.826] | 0.128 | 0.089 |
| cogn_global −2.0 to −1.0 | 746 | 73 | **0.770** | [0.711, 0.823] | 0.105 | 0.035 |
| MMSE Normal (back to ≥27) | 165 | 47 | **0.758** | [0.671, 0.838] | 0.296 | 0.275 |
| cogn_global 0.2 to 0.8 | 503 | 97 | **0.758** | [0.709, 0.807] | 0.201 | 0.163 |
| MMSE Moderate (18–23) | 758 | 83 | **0.729** | [0.674, 0.786] | 0.117 | 0.064 |
| cogn_global −1.0 to −0.3 | 651 | 115 | **0.710** | [0.662, 0.758] | 0.187 | 0.080 |
| cogn_global −0.3 to 0.2 | 550 | 105 | **0.627** | [0.573, 0.681] | 0.201 | 0.118 |
| Death | 793 | 176 | **0.607** | [0.561, 0.649] | 0.237 | **0.006** |
| MMSE Mild (24–26) | 676 | 111 | **0.562** | [0.505, 0.622] | 0.173 | 0.104 |

**TWO INSTRUMENTS, AND THEY MUST NOT BE SUMMED.** The four MMSE stages are a partition — exactly
one holds at any time, and panel c's stage-at-age grid walks it. The six `cogn_global` levels are
a SECOND, independent partition, scored identically but deliberately kept out of the state space:
a cognition token inside `GRID_TOKENS` would overwrite the MMSE stage on panel c's carry-forward
grid and make the stage unreadable for every later grid year. The panel groups the three families
by POSITION (a blank row between blocks) rather than by hue, because twelve identities is more
than colour can carry — every row is named on the axis and carries its AUC at the bar end, so
colour only has to say which family a row belongs to. The two ordinal ramps were checked with the
dataviz validator on their FAMILY identities (GnBu mid, RdPu mid, `#D55E00`, `#4d4d4d`): CVD worst
ΔE 8.8 deutan / 7.2 tritan, normal-vision worst 20.8, both PASS. Within a family, order is carried
by lightness (monotonic, asserted in the test suite).

**`cogn_global` is the better instrument, and the shape is the same on both.** Its floor (0.627 at
−0.3 to 0.2) sits well above MMSE's (0.562 at Mild), and its top row (0.823 at >0.8) is the best
row in the whole panel — above AD. Both instruments show the same U: the extremes are predictable,
the middle of the distribution is not. That is the third independent line pointing at the same
fact — a bin in the middle of a noisy scale is the hardest thing here to predict, which is what
the σ measurement says from the other side and what `vocab.py` predicted when it recorded that
`cogn_global` is complementary to MMSE rather than redundant and starts moving 3–4 years earlier.

`cogn_global ≤−2.0` sits exactly ON the 30-event floor and its CI is correspondingly wide
[0.724, 0.892]. Read it as "high, imprecise", not as 0.806.

> Median across the 12 rows 0.758. Every rate is UNDER-predicted (mean −0.065), death by a
> factor of 40 and `cogn_global ≤−2.0` by a factor of 3. The one nearly-calibrated row is
> `cogn_global >0.8` (0.103 observed against 0.096 predicted) — which is IMPROVEMENT.
>
> **`Mild` is the one row that fails, and the reason is measured rather than guessed.** Among
> baseline-Normal subjects the observed 5-year rate of reaching 24–26 is 0.211 / 0.218 / 0.215 /
> 0.120 for a baseline MMSE of 27 / 28 / 29 / 30 — i.e. someone sitting one point from the 26.5
> boundary has the SAME risk as someone at 29, and only the ceiling (30) differs. Spearman
> between baseline MMSE and the outcome is **−0.091**. The strongest feature available carries
> almost no information about this endpoint, so an AUC of 0.56 is close to what is achievable,
> not a failure to use signal. It is the same fact the σ measurement reports from the other
> side: at 27.5 the measurement SD is 1.21 points, so "27 vs 29" is within noise.
>
> Two secondary contributors, both quantified. (i) The label is NON-MONOTONE: of baseline-Normal
> subjects who ended up at Moderate or Severe, **52.6% were never observed in 24–26** — they are
> true decliners (mean predicted 0.118 against 0.104 for non-decliners) labelled NEGATIVE for
> this endpoint. Restricting to subjects whose decline stopped at Mild lifts the AUC 0.560 →
> 0.605; using the monotone composite "reach Mild OR WORSE" gives 0.654. (ii) The model's output
> is nearly constant: across four groups with wildly different true outcomes the predicted
> Mild-risk spans only 0.089–0.141.
>
> A mechanism that was tested and REFUTED: death competition. `composite` requires the stage
> emission to precede the sampled death, so an aggressively-dying model could mechanically
> suppress cognitive endpoints in the frailest subjects. Removing the death term changes the
> Mild AUC by **0.000** on both the delivered model and the short-recipe one, and the two risks
> are positively (not negatively) correlated. Not the cause.
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

**c — Per-individual trajectory comparison.** Observed vs predicted stage-at-age on a yearly
grid, years 1–15 after baseline (year 0 excluded: both series equal the baseline stage there by
construction). A grid point counts only while the subject is under observation, or once they are
known dead. Scored against a **carry-baseline-forward** reference — the trivial "nothing changes"
predictor that repeats the subject's baseline stage for 15 years, using no model, no age, nothing.
It is not a weak reference: 79% of this cohort is Normal at baseline, so "nothing changes" is
right 87% of the time at year 1.

> **THIS PANEL USED TO REPORT A MODAL STATISTIC, AND THAT WAS A MEASUREMENT ERROR ON MY PART.**
> The old version compared `argmax` over the predicted state distribution against the observed
> stage, and reported model Jaccard 0.4244 against the reference's 0.4230 — a lift of +0.0014 —
> which I wrote up as "the trajectory panel cannot distinguish the model from assume-nothing-
> changes". That conclusion does not follow. `argmax` is a hard 0.5 threshold, and measured on
> this model P(the subject has left their baseline stage) has mean 0.045 and **maximum 0.430** at
> year 1: not one subject crosses 0.5, so the modal prediction is "unchanged" for everybody, and
> the reference becomes identical to it by arithmetic. Only 5.6% of subjects cross 0.5 by year 5.
> The modal statistic was measuring the model's CALIBRATION and reporting it as an absence of
> information.
>
> The same rollouts discriminate who actually changes at **AUC 0.72–0.88** at every year.

The panel now scores the DISTRIBUTION, with two statistics and the distinction between them
stated because it is the whole point:

| | model | carry-baseline | who wins |
|---|---|---|---|
| **Brier** (strictly proper, lower better) | **0.766** [0.724, 0.808] | 1.007 | **model, by 24%** |
| P(truth) (linear, NOT strictly proper) | 0.471 [0.447, 0.497] | **0.497** | reference |
| modal Jaccard (0.5 threshold) | 0.424 [0.396, 0.453] | 0.423 | tie |

`P(truth)` is the probability placed on the state that actually occurred. It reduces to plain
accuracy for a deterministic forecaster, which makes it intuitive and directly comparable — but
the linear score is **not strictly proper**: it is maximised by betting everything on the mode,
which is exactly what the reference does. So the reference winning it is expected and means
nothing. **Brier** (`sum_k (p_k − y_k)^2`, the standard multiclass form) is strictly proper: it
cannot be improved by misreporting, and it charges the reference 2.0 every time its point mass
lands on the wrong state. That is the number to quote.

> The model is ahead on Brier **from year 1**, and the gap widens monotonically: 0.240 vs 0.261
> at year 1, 0.675 vs 0.865 at year 5, 1.224 vs 1.715 at year 15. The per-subject histogram shows
> the mechanism directly — the reference is bimodal at 0 and 2 (confidently right or confidently
> wrong), the model is spread and almost never lands above 1.75.
>
> **BUT READ WHAT THE BRIER GAIN IS.** The model wins because it does not commit, not because it
> is right. Hedging genuinely beats being confidently wrong, and that is worth 24% here — but the
> underlying rates are still ~2.5× too low (predicted P(changed) 0.172 against an observed 0.432
> at 5 y, 0.381 against 0.858 at 15 y, and the deficit COMPOUNDS rather than being a fixed delay).
> A well-calibrated model would beat the reference on all three statistics. This one beats it on
> the one that rewards uncertainty.

Per-state IoU (still modal, which is inherent to IoU): Normal 0.508, Mild 0.184, Moderate 0.072,
Severe 0.055, Death **0.001** (macro 0.164).

The **alive-years-only** Jaccard (0.697, n=627) is reported alongside because once a subject dies
every remaining grid year is "Death", so one correct death call can carry an otherwise poor
cognitive trajectory. Four illustrative subjects are chosen deterministically — two the model
matches, one it over-predicts, one it under-predicts. They illustrate; the aggregate statistics
on the right are the evidence.

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

0. **The modal trajectory statistic was wrong and is fixed.** See panel c. Any earlier text of
   mine claiming the model is indistinguishable from "nothing changes" was an artifact of
   thresholding a forecast at 0.5 that never reaches 0.5; the real defect is calibration, and
   the discrimination is AUC 0.72–0.88.
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
