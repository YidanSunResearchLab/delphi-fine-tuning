# Capacity sweep — what we ran and what came out

48 runs on RIS: 18 arms at the 5000-iter schedule (6 shapes × 3 seeds), 12 at a 15000-iter
schedule, 18 downstream Figure-2 evaluations (6 shapes × 3 seeds). Nothing under `Delphi-2M/`
was modified to produce them.

> Since then `Delphi-2M/` has gained one opt-in flag — `time_head`, default `False` — for the
> follow-up to recommendation 4 below. At the default it is bit-identical to the code these
> runs used (`tests/golden.py`: 29/29 arrays at `atol=0`); these numbers stand as measured.

## The question

The delivered model `out-delphi2m-dedup-mask-s42` is 12L/12H/120d = **2,104,320 parameters**,
trained on 16,262 patients / ~500k events. Its best checkpoint lands at iter 2750 of 5000 —
validation loss bottoms at 55% of the schedule and rises for the rest. The architecture was
copied verbatim from the upstream Delphi-2M demo config (vocab 1270, a cohort orders of
magnitude larger); it was never chosen for this dataset.

**Is 2.1M the right size, and does capacity change anything that matters?**

## Design

Everything except `(n_layer, n_head, n_embd)` and the seed is held constant, and that is
*enforced* — `verify.py` check [2] diffs every generated config against
`config/train_delphi2m_mask_dedup.py` and fails on any unexpected key.

One deliberate deviation: `eval_interval` 250 → 100, uniformly. `estimate_loss()` draws
batches from the **global** torch RNG (400 `randint` calls per eval), so the eval cadence
shifts the training batch stream. It is not a free knob, and the delivered checkpoint
(eval_interval 250) is therefore **not** an arm — the `L12E120H12` arm is the reference.

**Reproduction check:** that arm lands at best_iter 2733 ± 153 / best_val 10.0981 ± 0.0054
against the delivered checkpoint's 2750 / 10.1035. The chain is faithful.

## Everything in one table

| shape | L/H/d | params | ms/iter | val loss | best iter | loss_ce | median AUC | Dem@10y |
|---|---|---:|---:|---|---|---:|---|---|
| **L12E120H12** *(delivered)* | 12/12/120 | 2,104,320 | 217 | 10.0981 ± 0.0054 | 2733 ± 153 | 2.7249 | **0.6532** ± 0.0117 | **0.8433** ± 0.0059 |
| L12E120H6 | 12/6/120 | 2,104,320 | 202 | 10.0938 ± 0.0036 | 2367 ± 153 | 2.7169 | 0.6697 ± 0.0067 | 0.8597 ± 0.0061 |
| **L8E120H6** | 8/6/120 | 1,412,160 | 135 | 10.0932 ± 0.0011 | 2533 ± 58 | 2.7178 | **0.6942** ± 0.0098 | **0.8677** ± 0.0021 |
| L8E96H6 | 8/6/96 | 906,240 | 116 | 10.0865 ± 0.0014 | 3900 ± 265 | 2.7116 | 0.6853 ± 0.0195 | 0.8648 ± 0.0032 |
| L6E96H6 | 6/6/96 | 684,672 | 85 | 10.0857 ± 0.0111 | 4167 ± 208 | 2.7108 | 0.6781 ± 0.0053 | 0.8598 ± 0.0034 |
| L6E64H4 | 6/4/64 | 306,944 | 53 | **10.0690** ± 0.0032 | 4667 ± 321 | 2.7030 | 0.6893 ± 0.0048 | 0.8565 ± 0.0068 |

Bold = best/worst in that column. Note they do not agree.

## Four findings

### 1. Head geometry matters, at identical parameter count — and val loss cannot see it

`L12E120H6` is the delivered architecture with **only** `n_head` changed, 12 → 6 (head_dim
10 → 20). `n_head` does not appear in the parameter formula: both are 2,104,320 params, same
speed.

| metric | head_dim 10 | head_dim 20 | Δ | pooled sd |
|---|---|---|---:|---:|
| median AUC | 0.6532 ± 0.0117 | 0.6697 ± 0.0067 | **+0.0165** | 0.0095 |
| MCI→Dementia AUC | 0.7113 ± 0.0156 | 0.7309 ± 0.0071 | **+0.0196** | 0.0121 |
| Dementia @5y | 0.8720 ± 0.0081 | 0.8833 ± 0.0022 | **+0.0113** | 0.0059 |
| Dementia @10y | 0.8433 ± 0.0059 | 0.8597 ± 0.0061 | **+0.0164** | 0.0060 |
| Death @5y | 0.7695 ± 0.0321 | 0.7957 ± 0.0008 | **+0.0262** | 0.0227 |

**5/5 better, every one outside seed noise.** The same comparison on validation loss is
10.0981 vs 10.0938 — Δ −0.0043 against a pooled sd of 0.0046, i.e. **inside** noise. Judged
on val loss alone this change looks like nothing.

head_dim 10 was inherited from the upstream config, not chosen. Changing it is one line, zero
parameters, zero slowdown.

### 2. Validation loss must not be used to rank architectures here

The two rankings disagree outright: val loss picks the 0.31M arm, both downstream metrics
pick the 1.41M arm, and val loss is blind to the head-geometry effect in finding 1.

The reason is in the decomposition (`decompose.py`, measured on each arm's own best
checkpoint):

| | loss_ce (*which* event) | loss_dt (*when*) |
|---|---|---|
| spread across all 6 shapes | 0.0219 | **0.0071** |
| share of total val loss | 27% | **73%** |

`loss_dt` is 73% of the reported number and is **inert** — a 6.9× parameter range moves it by
0.10% of itself, with every shape inside 2–3 sd of the others. It dilutes the one term that
does respond. And even `loss_ce`'s full spread is small: against a uniform baseline of
ln(89) = 4.4886 nats, the models sit at ~2.71, so 6.9× of capacity buys **1.2% of the learned
signal**.

### 3. The delivered architecture is the worst arm downstream

Against the other five pooled: median AUC +0.0301 (pooled sd 0.0122), Dementia@5y +0.0139
(0.0065), Dementia@10y +0.0184 (0.0058) — all outside noise. MCI→Dementia and Death@5y are
inside noise.

But downstream performance is **not monotonic in capacity**: the best arm is 1.41M, not the
smallest. So this is not "smaller is better"; it is "the delivered point is a bad one", and
capacity is the wrong axis to describe why (see finding 1).

### 4. The "small models just hadn't converged" objection is refuted

In the 5000-iter sweep the bottom moves monotonically later as capacity falls — to iter
4667 ± 321 for the 0.31M arm, 93% of the schedule. That confounds "smaller is better" with
"smaller has not finished". Re-running with `max_iters` and `lr_decay_iters` both at 15000:

| shape | 5000-iter | 15000-iter | Δ |
|---|---|---|---:|
| L12E120H12 | 10.0981 | 10.1105 | +0.0125 |
| L8E96H6 | 10.0865 | 10.1040 | +0.0175 |
| L6E96H6 | 10.0857 | 10.1044 | +0.0187 |
| L6E64H4 | 10.0690 | 10.0919 | +0.0230 |

**Every shape got worse**, and the bottoms are now interior (2200–5767 of 15000) rather than
pinned to the end. The two schedules draw *identical batches at identical iterations* — same
seed, same eval cadence, so the same RNG call sequence — and differ only in the cosine length.
The 5000-iter ranking stands; a longer schedule just holds the LR high for longer.

## Recommendations

1. **Set `n_head = 6`.** Same parameters, same speed, 5/5 downstream metrics better and
   outside noise. Highest value-per-risk change in the whole sweep.
2. **If shrinking, go to 8L/6H/120d (1.41M), not smaller.** Best on both downstream metrics
   and 1.6× faster than the delivered model. The 0.31M arm wins on val loss and does *not*
   carry that win downstream.
3. **Score architecture choices on Figure-2 metrics, never on val loss.** Budget ~40 min of
   CPU per arm per seed (`slurm/eval_sweep.sbatch`) and use ≥3 seeds; single-seed bootstrap
   CIs on MCI→Dementia are ±0.03 and overlap for every pair of shapes.
4. **The real target is `loss_dt`, not the architecture.** 73% of the loss, immovable across
   a 6.9× capacity range. Whatever is limiting it — the `t_min` floor, the competing-exponential
   assumption, the no-event token rate — will matter more than any reshaping of the trunk.
   *Taken up in `experiments/time_head/`, which tests the fourth candidate this sweep could not
   see: that "when" had no parameters of its own at all. `time_head_probe.py` here is the
   evidence that motivated it.*

## Caveats

- Downstream metrics are the **cohort-matched** test set (n = 4,589), scored with the same
  Monte-Carlo, competing-risk and censoring-aware machinery as the paper figure.
- 3 seeds per arm. Differences quoted as "outside noise" exceed the pooled seed sd; that is a
  weak criterion, not a hypothesis test. Treat rankings among the five non-reference arms as
  provisional — only the reference-vs-rest and head-geometry contrasts are clean.
- Everything ran on `general-cpu`: the `general-gpu` partition refuses this account's jobs
  entirely (see README). CPU means plain fp32 with no autocast and no TF32; all arms identical,
  so comparisons are unaffected.
