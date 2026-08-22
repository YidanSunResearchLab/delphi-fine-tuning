# Capacity sweep — is 2.1M parameters the right size for this dataset?

Self-contained experiment. It does not modify anything under `Delphi-2M/`; it drives the
existing `training/train.py` and `figure2/` with generated configs and reads their stdout.

## The question

The delivered model `out-delphi2m-dedup-mask-s42` is **12L / 12H / 120d = 2,104,320
parameters**, trained on **16,262 patients / ~500k events** (after `filter_cohort` keeps
16,262 of 38,687). Its best checkpoint lands at **iter 2750 of 5000** — validation loss
bottoms at 55% of the schedule and rises for the remaining 45%.

That architecture was never chosen for this dataset. It is copied verbatim from the
upstream Delphi-2M demo config, whose vocabulary is 1270 (ours is 111) and whose cohort is
orders of magnitude larger. The vocabulary difference barely matters for size — the token
embedding is only 0.6% of the parameters, so an 11× smaller vocabulary buys a 6% smaller
model — but the **data** difference is not 11×, it is far more, and the model did not
shrink to match.

**Does lower capacity reach a lower validation loss, and does the bottom move later?**

## The matrix

6 shapes × 3 seeds (42/43/44) = 18 runs.

| tag | shape | head_dim | params | why it is in the sweep |
|---|---|---:|---:|---|
| `L12E120H12` | 12L/12H/120d | 10 | 2,104,320 | delivered architecture — the reference arm |
| `L12E120H6` | 12L/6H/120d | 20 | 2,104,320 | **same params**, different head split |
| `L8E120H6` | 8L/6H/120d | 20 | 1,412,160 | depth down, width held |
| `L8E96H6` | 8L/6H/96d | 16 | 906,240 | the shape proposed in review |
| `L6E96H6` | 6L/6H/96d | 16 | 684,672 | depth down again |
| `L6E64H4` | 6L/4H/64d | 16 | 306,944 | small end — is the knee already behind us? |

`L12E120H6` is the one arm that is not about capacity. It has the **identical parameter
count** to the reference and changes only how 120 dimensions are split into heads (20-d
heads instead of 10-d). `n_head` does not appear in the parameter formula at all, so this
isolates head geometry from size — a free control worth having, because head_dim 10 is
unusually narrow and was inherited, not chosen.

## What is held constant

Everything else: dataset `nacc-dedup-s42`, cohort filter 4/2, `ignore_tokens=0..21`,
block_size 96, vocab 111, lr 2e-3 with 500-iter warmup and cosine decay to 2e-4, 5000
iters, batch 128, weight_decay 0.2, beta2 0.99, dropout 0, grad_clip 1.0.

This is enforced, not just intended: `verify.py` check **[2]** diffs every generated config
against `config/train_delphi2m_mask_dedup.py` and fails if anything outside
`{n_layer, n_head, n_embd, eval_interval, out_dir, seed, wandb_run_name}` differs.

### The one deliberate deviation: `eval_interval` 250 → 100

`estimate_loss()` in `train.py` draws its batches with the **global** torch RNG
(`eval_iters=200` × 2 splits = 400 `randint` calls per eval), so **the eval cadence shifts
the training batch stream**. `eval_interval` is therefore not a free knob. It is pinned to
100 for every arm — 250 is too coarse to locate a bottom that sits at 2750.

The consequence: **the delivered checkpoint is not an arm of this sweep.** It ran at 250
and saw a different batch stream. The `L12E120H12` arm is the like-for-like reference, and
the delivered numbers appear in the outputs as context only, marked as such.

## Layout

```
shapes.py          single source of truth: the 6 shapes, the seeds, the param formula
gen.py             regenerates configs/ + manifest.tsv from shapes.py (idempotent)
configs/           GENERATED — never hand-edit
manifest.tsv       array-task table: idx -> (tag, seed, shape, params)
verify.py          preflight: 6 checks, gates submission
deploy.sh          rsync code to RIS, symlink data, run preflight there
slurm/
  train_sweep.sbatch   phase 1 — GPU array, one task per (shape, seed)
  eval_sweep.sbatch    phase 2 — CPU array, Figure-2 metrics per shape
collect.py         runs/*/train.log -> results/{curves,summary}.csv
analyze.py         -> capacity_table.md, findings.md, capacity_{curves,knee}.png
```

## Running it

```bash
python experiments/capacity/verify.py          # must pass first
./experiments/capacity/deploy.sh               # -> RIS, re-runs preflight against real data

ssh compute2
cd /storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0820_capacity
SMOKE=200 sbatch --array=10 experiments/capacity/slurm/train_sweep.sbatch   # validate first
sbatch --array=1-18%8 experiments/capacity/slurm/train_sweep.sbatch          # then the sweep
sbatch --array=1-6%3  experiments/capacity/slurm/eval_sweep.sbatch           # after phase 1
```

### This runs on CPU, and that is not a compromise

`general-gpu` does not schedule this account's jobs. Measured 2026-08-20 with a probe matrix
in which the **only** thing that varied was the partition:

| submission | outcome |
|---|---|
| `general-gpu --gres=gpu:1 -c 4 --mem=32G` | stuck, `PartitionConfig` |
| `general-gpu --gres=gpu:1 -c 8 --mem=32G` | stuck (and this is the exact shape that ran here historically) |
| `general-gpu` **no GPU at all**, `-c 4 --mem=32G` | stuck |
| `general-cpu` no GPU, `-c 4 --mem=32G` | **COMPLETED in 1 s** |

So it is the partition, not the GPU request, not `--exclude`, not group membership
(`id -Gn` contains `compute2`, which is in the partition's `AllowGroups`). There are free
GPUs and no reservations. This needs an RIS admin; nothing in this repo can fix it.

It costs the sweep very little. The model is 0.3–2.1M parameters, and on a general-cpu node
the largest shape runs at **170 ms/iter on 32 cores, 226 ms on 16, 418 ms on 8**. The
`general-cpu-qos` caps a user at `cpu=128`, so cores-per-job trades against concurrency;
16 cores × 8 concurrent jobs was chosen because it is within ~8% of the best aggregate
throughput while halving per-run latency, and because the 16-core figure is measured rather
than extrapolated. All 18 arms complete in roughly 70 minutes.

Numerically, `device=cpu` means `train.py` uses `nullcontext()` rather than autocast and
never touches CUDA's TF32 path, so these runs are plain fp32. Every arm runs identically, so
the comparison is unaffected.

If GPU access is restored, set `--partition=general-gpu --gres=gpu:1` in the sbatch and pass
`DEVICE=cuda`; nothing else changes.

Then, back on the laptop:

```bash
rsync -az compute2:.../AD0820_capacity/runs ./experiments/capacity/
python experiments/capacity/collect.py
python experiments/capacity/analyze.py
```

### Deployment note

`deploy.sh` writes to a **new** cluster directory, `AD0820_capacity`, beside the existing
`AD0718`. The existing checkout is the old flat layout (`Delphi-2M/model.py`,
`Delphi-2M/eval_ad.py`) from before the restructure into packages — this code cannot run
there, and overwriting it would destroy the provenance of the delivered model.

**No PHI moves.** `data/` is a symlink to the split already on the cluster; the raw CSV and
the `.bin` files are never transferred in either direction, and the rsync excludes
`*.bin`, `*.csv`, and `investigator_nacc*`.

## Reading the results

Two different things are measured and they are not interchangeable:

- **`best_val`** — how good the model gets. This is the model-selection criterion, because
  `train.py` only writes `ckpt.pt` when val loss improves. **Only a lower `best_val` is a
  win.**
- **`best_iter`** — when it peaks. Later means the schedule is better matched to the
  capacity. A shape can move the bottom later without getting any better — it just
  overfits more slowly to the same place.

With 3 seeds the spread on `best_iter` is wide. `analyze.py` prints the seed sd next to
every mean, and states explicitly whether a difference exceeds it. **Do not quote a
difference smaller than the seed spread.**

## The caveat that belongs with every number here

Validation loss is `loss_ce + loss_dt`, not a clinical endpoint. A shape that wins on val
loss has **not** been shown to win on the things the model is for: transition AUC, timing
MAE, trajectory Jaccard against the carry-baseline-forward reference. That is what phase 2
(`eval_sweep.sbatch`) measures, and it is the one that decides anything.

## If you change the sweep

Edit `shapes.py`, re-run `gen.py`, re-run `verify.py`, redeploy. Never edit a file under
`configs/` — it will be silently overwritten on the next `gen.py`.
