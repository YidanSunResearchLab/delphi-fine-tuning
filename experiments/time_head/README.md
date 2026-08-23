# Time head — does the timing objective get better if you give it its own parameters?

Follow-up to `experiments/capacity/`, and specifically to its recommendation 4:

> **The real target is `loss_dt`, not the architecture.** 73% of the loss, immovable across a
> 6.9× capacity range. Whatever is limiting it — the `t_min` floor, the competing-exponential
> assumption, the no-event token rate — will matter more than any reshaping of the trunk.

This experiment tests the fourth candidate, the one that was implicit in the model rather
than in the config: **that "when" had no parameters of its own.**

## The model as it stands

![the current architecture](results/architecture.png)

Regenerate with `python experiments/time_head/figure_architecture.py` (PNG at 300 dpi +
vector PDF). It is drawn from the code rather than photographed, so if `model.py` changes,
that script is the thing to update — and a reader can check every label against the source.

## The question

Delphi has **one** output projection. `lm_head` is vocab-wide and weight-tied to the token
embedding, and both objectives are read off it:

```python
logits  = lm_head(x)                                  # (b, t, 111)
loss_ce = cross_entropy(logits, next_token)           # WHICH event
lambda  = logsumexp(logits), floored via t_min        # WHEN — one scalar per position
loss_dt = lambda*t - log(lambda)                      # exponential NLL
```

So "when" is a single number per position **and** a deterministic function of the same logits
that decide "which". Adding capacity to the trunk gives the timing objective nowhere to put
it, which is a plausible mechanical explanation for the capacity sweep's finding 2: `loss_dt`
is 73% of the reported validation loss and a 6.9× parameter range moved it by 0.0071 nats,
0.10% of itself, with every shape inside 2–3 sd of the others.

**The arm:** `time_head=True` adds `nn.Linear(n_embd, 1)` and reads the intensity off *that*.
121 parameters at `n_embd=120` — 0.006% of the delivered model.

> Does `loss_dt` move when it has its own head, and does downstream **timing** get better?

Either answer closes a branch. If 121 dedicated parameters move a term that 1.8M shared ones
could not, parameterisation was the binding constraint and the next question is how much head
to give it. If it still does not move, the limit is the likelihood itself — the
competing-exponential assumption, the `t_min` floor, the synthetic no-event grid — and those
are what to attack instead.

## What changed in the model

`delphi/model.py`, opt-in, default off. `time_head=True` makes `logits` the **per-token
log-rates** of the same competing-exponential model:

```python
log_lambda = time_head(x)                                   # (b, t)
logits     = log_softmax(lm_head(x)) + log_lambda           # (b, t, 111)
```

Two exact identities make that a drop-in, and `verify.py` checks all three:

| identity | why it matters | measured |
|---|---|---|
| `softmax(logits)` unchanged | a per-position constant cancels in softmax, so `loss_ce`, `ad_engine.next_event_probs`, the panels' state probabilities and `generate()`'s sampled token are the same functions of `lm_head` as before | max abs diff **4.7e-09** |
| `logsumexp(logits) == log_lambda` | `logsumexp(log_softmax(·)) == 0`, so `generate()`'s inverse-CDF sampling and `predict_adapter`'s closed-form rate pick up the new head with **no code change** and the right semantics: `rate_k = lambda * p_k`, total rate `lambda` | max abs diff **2.2e-07** |
| `loss_ce` identical at identical weights | "which event" is untouched | diff **exactly 0** |

And the gradients route the way the experiment needs them to: `d loss_ce / d(time head)` is
**2e-07** (analytically zero — the constant drops out of softmax) while
`d loss_dt / d(time head)` is **O(20)**. The head is trained by the timing objective alone.

One deliberate asymmetry, documented at the line: the loss reads `log_lambda` **directly**
rather than as `logsumexp` of the masked rate-logits. `validation_loss_mode` writes `-inf`
into the ignored columns, so `logsumexp` there would return `log(lambda · P(content))` —
lambda deflated by the content probability mass, a quantity the head never sees in training
(training mode applies no such mask). Reading the head directly scores it on the definition
it is trained on. The single-head path keeps `logsumexp` exactly as delivered.

### The default-off path is bit-identical

`time_head=False` is not "equivalent", it is the same numbers. `tests/golden.py` — the repo's
own regression harness, run against `ckpt.pt` — passes **29/29 arrays at `atol=0`**, including
the sampled `generate()` trajectories, whose RNG call order would expose any change to the
forward pass. Every existing checkpoint keeps loading and every capacity-sweep run stays
reproducible.

Two incidental fixes were needed to get there, both flagged in the code:

- `model.py`'s `mask_ties` gather used `.squeeze((1, 2))`, which also dropped the *time* axis
  when `t == 1` and crashed with "Index tensor must have the same number of dimensions as
  input tensor". Now `.squeeze(1)`. Identical for every `t > 1`, so no trained model is
  affected; it only stops 1-token sequences from crashing — which is what `golden.py`'s first
  case does, and why `golden.npz` had never been recorded.
- `golden.py` compared NaN with `max|a-b|`, so `single_token`'s two losses (NaN by
  construction: its only target is padding) reported a permanent FAIL. Both-NaN now counts as
  unchanged; NaN on one side still fails.

## The matrix — 6 new runs, not 12

| arm | shape | params | control | why |
|---|---|---:|---|---|
| `TH_L12E120H12` | 12L/12H/120d | 2,104,441 | `L12E120H12` | the delivered architecture — reads straight against the capacity table's reference arm |
| `TH_L8E120H6` | 8L/6H/120d | 1,412,281 | `L8E120H6` | the shape the sweep recommends (best on both downstream metrics) |

× seeds 42/43/44. Two shapes, not six: the capacity sweep already established that the
trunk's size is not what moves `loss_dt`, so re-sweeping shape would spend CPU-hours
re-answering a settled question. These are the two shapes anyone would actually ship.

**The controls are the capacity sweep's own runs**, reused rather than re-run, and the pairing
is exact rather than approximate:

- the model change consumes no RNG and adds no parameter when `time_head=False` (proved at
  `atol=0` above), so those runs are still reproducible as-is;
- `Delphi.__init__` builds the head **after** the shared init has drawn its numbers, so a
  `time_head=True` model is **bit-identical to its control on every shared parameter** at the
  same seed — `verify.py` check [3];
- the head's bias is **warm-started to `log(vocab_size)`**, so both members of a pair start at
  the *same* intensity and therefore in the same `t_min`-floor regime — `verify.py` check [5];
- `time_head.weight` is kept out of the weight-decay group, matching the control, whose
  intensity comes from the tied (and blacklisted) `wte`/`lm_head` tensor — `verify.py` check [5];
- `train.py` re-seeds the batch RNG (`manual_seed(seed + iter_num)`) before the training loop,
  so both members of a pair also see the **same batch stream**.

A pair therefore differs in exactly one thing: the presence of the head. That is worth more
than three extra seeds — the per-seed *difference* removes initialisation and batch-order
variance instead of averaging over it, which matters when the effect under test is smaller
than the capacity sweep's own seed sd (~0.005 on total val loss).

> The ordering that makes this work is load-bearing and was got wrong first: building the head
> before `self.apply(_init_weights)` shifts the RNG state that sets *every* weight, and then
> **no** tensor matched the control. `verify.py` check [3] exists because that happened.

> The bias warm start is load-bearing for the same reason. `loss_dt` only reaches the trunk
> through the floor, whose derivative is `exp(-lse) / (exp(-lse) + t_min)`. The control starts
> at `lse = log(vocab_size)`, i.e. `exp(-lse) ≈ 1/111` against `t_min = 30.44` — deeply
> saturated, `loss_dt` almost invisible to it. A zero-biased head starts at `exp(-lse) = 1`,
> ~111× further out. Measured at the 8L/6H/120d arm config, at init:
>
> | arm | floor attenuation | `loss_dt` | ‖g_dt‖/‖g_ce‖ on the trunk |
> |---|---|---|---|
> | control | 2.79e-04 | 11.9101 | 0.0034 — dt contributes ~nothing |
> | `time_head`, bias 0 | 3.71e-02 | 11.6347 | 3.8149 — dt dominates |
> | `time_head`, bias `log V` | 3.47e-04 | 11.9096 | 0.0373 |
>
> With a zero bias the arms are not running the same optimisation at step 0 — one effectively
> CE-only, the other dt-dominated, a ~1100× gap — and a `loss_dt` win would be confounded with
> simply having started outside the floor. The residual ~10× is structural: the control's dt
> gradient reaches `x` as a softmax-weighted average over `vocab_size` `lm_head` rows, which
> largely cancels, while the head's goes through one row. That difference *is* the treatment.

## What is held constant

Everything. Each config is `experiments/capacity/configs/cap_<control>.py` read off disk
verbatim, plus one appended line, `time_head = True`. Nothing is retyped — `gen.py` builds it
by reading that file, so regenerating the capacity configs and re-running `gen.py` keeps them
in lockstep. `verify.py` check [6] fails if the diff is ever anything but `{time_head}`.

## Run it

```bash
python experiments/time_head/gen.py            # configs/ + manifest.tsv from arms.py
python experiments/time_head/verify.py         # 9 checks — four of them are the premise
```

Preflight must pass before anything is submitted. Then:

```bash
# phase 1 — train the 6 arms (~40 min each, 16 cores; all 6 fit the 128-core quota at once)
sbatch --array=1-6 experiments/time_head/slurm/train_time_head.sbatch

# phase 2 — the Figure-2 metrics, all 6 (the headline is TIMING, which is Monte-Carlo)
sbatch --array=1-6%3 experiments/time_head/slurm/eval_time_head.sbatch

# analysis — decompose val loss into its two terms, then pair against the controls
python experiments/capacity/decompose.py --runs experiments/time_head/runs \
       --out experiments/time_head/results/decomposition.csv
python experiments/time_head/analyze.py        # -> results/paired.csv, results/PAIRED.md
```

`analyze.py` degrades cleanly: with only the training logs it reports paired val loss and
best-iter, and lists what is still missing. On a laptop, `run_local.sh` runs the same configs
— read its header first, because a local arm paired against a cluster control is not a
controlled comparison.

## How to read the result

`analyze.py` reports **paired** deltas — `d` and `sd(d)` over the seeds — in this order:

1. **`loss_dt`.** Yardstick: **0.0071** nats is what 6.9× of trunk capacity bought. A `d` at
   or beyond that, consistent in sign across 3 exact pairs, is the result this arm was built
   to find.
2. **Downstream timing.** `a/transition_time/mae` (years, ~2.4k transitions) and
   `b/timing_error/*/mae` against their `naive_mae` baseline. This is what a clinician would
   feel, and the capacity sweep never moved it. Note the bar: the delivered model's near-term
   timing MAE is ~4.2 y for Dementia 0–2 y against a naive 1.76 y — **worse than naive**. Watch
   whether the head narrows that gap, not just whether it beats the control.
3. **Guards.** `median_auc`, `mean_jaccard`. The head cannot change `softmax(logits)` at fixed
   weights, but these runs *train* differently, so the categorical side can still drift. If it
   degrades, the head is not free.

Three seeds is three seeds: `sd(d)` over 3 pairs is a crude estimate and nothing here prints a
p-value. The per-pair table is printed so sign consistency is visible, which on an exact
pairing carries more than the sd suggests.

## If it works, the obvious next arms

Not built, deliberately — one variable at a time:

- **more head.** `nn.Linear(n_embd, 1)` is the smallest possible intervention. A 2-layer MLP
  head answers "how much" once "whether" is settled.
- **decouple the trunk.** The head still reads the same `ln_f(x)` the token logits read, so
  "when" and "which" share every representation. A separate final LayerNorm, or a dedicated
  block, is the next place capacity could go.
- **drop the exponential.** `loss_dt` assumes a constant hazard between events. A Weibull or
  a discrete-time hazard changes the likelihood, not the parameterisation — the other branch
  of recommendation 4, and the one to take if this arm comes back flat.
