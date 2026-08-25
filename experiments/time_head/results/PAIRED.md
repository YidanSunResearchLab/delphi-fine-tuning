# Paired against the capacity sweep's own runs

9 pairs across 3 arm(s). Every row is one arm run minus its control at the SAME seed; the two members of a pair start from bit-identical trunk weights and see the same batch stream, so `d` is a within-pair difference, not a difference of means. Arms are reported separately.

---

## DT_L12E120H12  (control `L12E120H12`, 3 seeds)

Sets `dt_target = 'next_event'`.

> **Validation loss is NOT comparable for this arm.** It changes `dt_target`, i.e. WHAT `loss_dt` is measured against. `loss_dt` is minimised at `1 + log E[dt]`, and `dt_ablation.py` measured the target's mean falling from 2.648 y to 1.488 y, so the *achievable* `loss_dt` drops by `log(2.648/1.488)` = **0.58 nats** with no change in model quality. A val-loss delta smaller than that says nothing about quality — if anything, less than 0.58 means the model fits its own (easier) target WORSE than the control fit its harder one, which is expected once the 73.6% of positions that were being handed a subtraction they could read off their own input stop being free. Read section 2, not section 1.

### 1. The objective terms

Bookkeeping only for this arm — see the warning above.

| metric | arm mean | control mean | d (mean) | sd(d) | pairs better |
|---|---:|---:|---:|---:|---|
| total val loss | 10.0454 | 10.0981 | **-0.0526** | 0.0025 | 3/3 |
| iter of the best checkpoint | 2400.0000 | 2733.3333 | **-333.3333** | 115.4701 | 3/3 |

### 2. Downstream timing — THE headline

`a/transition_time` over ~2.4k observed transitions, from sampled trajectories — objective-independent, so comparable for every arm. The control's bias is **+2.6 y** (events predicted too late) and its R2 is negative, i.e. worse than predicting the mean. Lower MAE is better; bias closer to 0 is better; Spearman higher is better (read its sign column inverted).

| metric | arm mean | control mean | d (mean) | sd(d) | pairs better |
|---|---:|---:|---:|---:|---|
| transition-time MAE (years) | 4.1286 | 3.7785 | **+0.3501** | 0.3143 | 0/3 |
| transition-time bias (years) | 3.5353 | 3.0380 | **+0.4974** | 0.4462 | 1/3 |
| transition-time Spearman | 0.2741 | 0.2815 | **-0.0075** | 0.0376 | 2/3 |
| mean calibration error | -0.1336 | -0.1193 | **-0.0142** | 0.0043 | 3/3 |

### 2b. Timing MAE by outcome and horizon

`naive` = predict one constant (the median observed time) for everyone. Per-bucket it is rigged in the constant's favour, since the buckets are cut on the observed time; the aggregate is the fair comparison.

| outcome / horizon | control MAE | arm MAE | d | naive |
|---|---:|---:|---:|---:|
| Death 0–2 y | 9.730 | 10.431 | **+0.701** | 5.422 |
| Death 2–5 y | 5.766 | 5.600 | **-0.166** | 3.324 |
| Death 5–10 y | 4.728 | 4.488 | **-0.239** | 1.215 |
| Death >10 y | 4.494 | 4.114 | **-0.381** | 5.998 |
| Dementia 0–2 y | 4.613 | 5.311 | **+0.698** | 1.760 |
| Dementia 2–5 y | 4.593 | 5.144 | **+0.550** | 0.736 |
| Dementia 5–10 y | 4.332 | 4.577 | **+0.245** | 3.960 |
| Dementia >10 y | 4.475 | 4.243 | **-0.232** | 10.043 |
| Reach ≥MCI 0–2 y | 7.067 | 6.887 | **-0.180** | 1.911 |
| Reach ≥MCI 2–5 y | 5.842 | 5.873 | **+0.031** | 0.701 |
| Reach ≥MCI 5–10 y | 3.976 | 3.821 | **-0.155** | 4.041 |
| Reach ≥MCI >10 y | 4.076 | 3.615 | **-0.461** | 9.602 |

### 3. Guards — did the categorical side pay for it?

HIGHER is better here — read the last column as pairs worse.

| metric | arm mean | control mean | d (mean) | sd(d) | pairs better |
|---|---:|---:|---:|---:|---|
| median transition AUC | 0.6642 | 0.6532 | **+0.0110** | 0.0152 | 3/3 |
| trajectory Jaccard | 0.6295 | 0.6333 | **-0.0037** | 0.0085 | 1/3 |
| auc[Death @10y] | 0.8426 | 0.8289 | **+0.0137** | 0.0101 | 3/3 |
| auc[Death @2y] | 0.6496 | 0.6795 | **-0.0299** | 0.0825 | 1/3 |
| auc[Death @3y] | 0.7348 | 0.7411 | **-0.0063** | 0.0199 | 2/3 |
| auc[Death @5y] | 0.7755 | 0.7695 | **+0.0060** | 0.0169 | 1/3 |
| auc[Dementia @10y] | 0.8591 | 0.8433 | **+0.0158** | 0.0058 | 3/3 |
| auc[Dementia @1y] | 0.8011 | 0.8048 | **-0.0037** | 0.0309 | 2/3 |
| auc[Dementia @2y] | 0.8679 | 0.8694 | **-0.0015** | 0.0254 | 1/3 |
| auc[Dementia @3y] | 0.8747 | 0.8717 | **+0.0030** | 0.0133 | 1/3 |
| auc[Dementia @5y] | 0.8788 | 0.8720 | **+0.0069** | 0.0102 | 2/3 |
| auc[Reach ≥MCI @10y] | 0.6872 | 0.6539 | **+0.0333** | 0.0369 | 3/3 |
| auc[Reach ≥MCI @1y] | 0.6170 | 0.6222 | **-0.0052** | 0.0192 | 1/3 |
| auc[Reach ≥MCI @2y] | 0.6824 | 0.6718 | **+0.0105** | 0.0082 | 3/3 |
| auc[Reach ≥MCI @3y] | 0.6876 | 0.6849 | **+0.0027** | 0.0055 | 2/3 |
| auc[Reach ≥MCI @5y] | 0.7050 | 0.6960 | **+0.0089** | 0.0107 | 3/3 |

### Per-pair detail

| pair | val loss | d | best iter | d | timing bias (y) | d |
|---|---:|---:|---:|---:|---:|---:|
| DT_L12E120H12_s42 vs L12E120H12_s42 | 10.0512 | -0.0529 | 2500 | -400 | 3.371 | +0.660 |
| DT_L12E120H12_s43 vs L12E120H12_s43 | 10.0437 | -0.0500 | 2500 | -200 | 3.540 | -0.007 |
| DT_L12E120H12_s44 vs L12E120H12_s44 | 10.0414 | -0.0550 | 2200 | -400 | 3.695 | +0.839 |

---

## DT_L8E120H6  (control `L8E120H6`, 3 seeds)

Sets `dt_target = 'next_event'`.

> **Validation loss is NOT comparable for this arm.** It changes `dt_target`, i.e. WHAT `loss_dt` is measured against. `loss_dt` is minimised at `1 + log E[dt]`, and `dt_ablation.py` measured the target's mean falling from 2.648 y to 1.488 y, so the *achievable* `loss_dt` drops by `log(2.648/1.488)` = **0.58 nats** with no change in model quality. A val-loss delta smaller than that says nothing about quality — if anything, less than 0.58 means the model fits its own (easier) target WORSE than the control fit its harder one, which is expected once the 73.6% of positions that were being handed a subtraction they could read off their own input stop being free. Read section 2, not section 1.

### 1. The objective terms

Bookkeeping only for this arm — see the warning above.

| metric | arm mean | control mean | d (mean) | sd(d) | pairs better |
|---|---:|---:|---:|---:|---|
| total val loss | 10.0427 | 10.0932 | **-0.0505** | 0.0029 | 3/3 |
| iter of the best checkpoint | 2766.6667 | 2533.3333 | **+233.3333** | 321.4550 | 0/3 |

### 2. Downstream timing — THE headline

`a/transition_time` over ~2.4k observed transitions, from sampled trajectories — objective-independent, so comparable for every arm. The control's bias is **+2.6 y** (events predicted too late) and its R2 is negative, i.e. worse than predicting the mean. Lower MAE is better; bias closer to 0 is better; Spearman higher is better (read its sign column inverted).

| metric | arm mean | control mean | d (mean) | sd(d) | pairs better |
|---|---:|---:|---:|---:|---|
| transition-time MAE (years) | 3.3546 | 3.3671 | **-0.0125** | 0.2621 | 2/3 |
| transition-time bias (years) | 2.5552 | 2.5703 | **-0.0151** | 0.3670 | 2/3 |
| transition-time Spearman | 0.3135 | 0.3171 | **-0.0036** | 0.0103 | 2/3 |
| mean calibration error | -0.1097 | -0.1082 | **-0.0014** | 0.0143 | 1/3 |

### 2b. Timing MAE by outcome and horizon

`naive` = predict one constant (the median observed time) for everyone. Per-bucket it is rigged in the constant's favour, since the buckets are cut on the observed time; the aggregate is the fair comparison.

| outcome / horizon | control MAE | arm MAE | d | naive |
|---|---:|---:|---:|---:|
| Death 0–2 y | 8.944 | 8.899 | **-0.045** | 5.422 |
| Death 2–5 y | 5.048 | 4.724 | **-0.324** | 3.324 |
| Death 5–10 y | 3.977 | 3.705 | **-0.272** | 1.215 |
| Death >10 y | 4.153 | 4.159 | **+0.005** | 5.998 |
| Dementia 0–2 y | 3.871 | 4.133 | **+0.262** | 1.760 |
| Dementia 2–5 y | 3.736 | 4.294 | **+0.558** | 0.736 |
| Dementia 5–10 y | 3.950 | 4.308 | **+0.358** | 3.960 |
| Dementia >10 y | 4.503 | 4.624 | **+0.121** | 10.043 |
| Reach ≥MCI 0–2 y | 6.038 | 6.013 | **-0.024** | 1.911 |
| Reach ≥MCI 2–5 y | 5.351 | 5.137 | **-0.214** | 0.701 |
| Reach ≥MCI 5–10 y | 3.855 | 3.490 | **-0.365** | 4.041 |
| Reach ≥MCI >10 y | 4.113 | 3.854 | **-0.259** | 9.602 |

### 3. Guards — did the categorical side pay for it?

HIGHER is better here — read the last column as pairs worse.

| metric | arm mean | control mean | d (mean) | sd(d) | pairs better |
|---|---:|---:|---:|---:|---|
| median transition AUC | 0.6790 | 0.6942 | **-0.0152** | 0.0108 | 0/3 |
| trajectory Jaccard | 0.6479 | 0.6464 | **+0.0015** | 0.0132 | 2/3 |
| auc[Death @10y] | 0.8467 | 0.8443 | **+0.0024** | 0.0164 | 2/3 |
| auc[Death @2y] | 0.6516 | 0.6637 | **-0.0120** | 0.0447 | 1/3 |
| auc[Death @3y] | 0.7531 | 0.7558 | **-0.0027** | 0.0050 | 1/3 |
| auc[Death @5y] | 0.7850 | 0.7872 | **-0.0022** | 0.0185 | 1/3 |
| auc[Dementia @10y] | 0.8635 | 0.8677 | **-0.0042** | 0.0125 | 1/3 |
| auc[Dementia @1y] | 0.8126 | 0.8115 | **+0.0011** | 0.0226 | 2/3 |
| auc[Dementia @2y] | 0.8834 | 0.8841 | **-0.0007** | 0.0104 | 2/3 |
| auc[Dementia @3y] | 0.8821 | 0.8877 | **-0.0056** | 0.0114 | 1/3 |
| auc[Dementia @5y] | 0.8837 | 0.8884 | **-0.0047** | 0.0087 | 1/3 |
| auc[Reach ≥MCI @10y] | 0.6975 | 0.7081 | **-0.0106** | 0.0219 | 2/3 |
| auc[Reach ≥MCI @1y] | 0.6512 | 0.6414 | **+0.0098** | 0.0254 | 2/3 |
| auc[Reach ≥MCI @2y] | 0.7097 | 0.7096 | **+0.0002** | 0.0134 | 2/3 |
| auc[Reach ≥MCI @3y] | 0.7090 | 0.7136 | **-0.0046** | 0.0094 | 1/3 |
| auc[Reach ≥MCI @5y] | 0.7166 | 0.7286 | **-0.0119** | 0.0118 | 1/3 |

### Per-pair detail

| pair | val loss | d | best iter | d | timing bias (y) | d |
|---|---:|---:|---:|---:|---:|---:|
| DT_L8E120H6_s42 vs L8E120H6_s42 | 10.0455 | -0.0482 | 2600 | +100 | 2.468 | -0.228 |
| DT_L8E120H6_s43 vs L8E120H6_s43 | 10.0403 | -0.0537 | 3100 | +600 | 2.266 | -0.226 |
| DT_L8E120H6_s44 vs L8E120H6_s44 | 10.0423 | -0.0496 | 2600 | +0 | 2.932 | +0.409 |

---

## TH_L8E120H6  (control `L8E120H6`, 3 seeds)

Sets `time_head = True`.

### 1. The objective terms

Yardstick: the capacity sweep moved `loss_dt` by **0.0071** nats across a 6.9x parameter range (its finding 2). A `d` at or beyond that, consistent in sign across seeds, is a result.

| metric | arm mean | control mean | d (mean) | sd(d) | pairs better |
|---|---:|---:|---:|---:|---|
| total val loss | 10.0760 | 10.0932 | **-0.0172** | 0.0119 | 3/3 |
| iter of the best checkpoint | 3833.3333 | 2533.3333 | **+1300.0000** | 264.5751 | 0/3 |

### 2. Downstream timing — THE headline

`a/transition_time` over ~2.4k observed transitions, from sampled trajectories — objective-independent, so comparable for every arm. The control's bias is **+2.6 y** (events predicted too late) and its R2 is negative, i.e. worse than predicting the mean. Lower MAE is better; bias closer to 0 is better; Spearman higher is better (read its sign column inverted).

| metric | arm mean | control mean | d (mean) | sd(d) | pairs better |
|---|---:|---:|---:|---:|---|
| transition-time MAE (years) | 2.9617 | 3.3671 | **-0.4054** | 0.3230 | 3/3 |
| transition-time bias (years) | 1.8212 | 2.5703 | **-0.7491** | 0.5770 | 3/3 |
| transition-time Spearman | 0.2842 | 0.3171 | **-0.0329** | 0.0498 | 2/3 |
| mean calibration error | -0.0849 | -0.1082 | **+0.0234** | 0.0212 | 0/3 |

### 2b. Timing MAE by outcome and horizon

`naive` = predict one constant (the median observed time) for everyone. Per-bucket it is rigged in the constant's favour, since the buckets are cut on the observed time; the aggregate is the fair comparison.

| outcome / horizon | control MAE | arm MAE | d | naive |
|---|---:|---:|---:|---:|
| Death 0–2 y | 8.944 | 8.400 | **-0.544** | 5.422 |
| Death 2–5 y | 5.048 | 4.148 | **-0.900** | 3.324 |
| Death 5–10 y | 3.977 | 3.656 | **-0.321** | 1.215 |
| Death >10 y | 4.153 | 4.373 | **+0.220** | 5.998 |
| Dementia 0–2 y | 3.871 | 3.353 | **-0.518** | 1.759 |
| Dementia 2–5 y | 3.736 | 3.351 | **-0.385** | 0.735 |
| Dementia 5–10 y | 3.950 | 3.820 | **-0.130** | 3.961 |
| Dementia >10 y | 4.503 | 4.583 | **+0.081** | 10.044 |
| Reach ≥MCI 0–2 y | 6.038 | 5.642 | **-0.396** | 1.911 |
| Reach ≥MCI 2–5 y | 5.351 | 4.737 | **-0.614** | 0.701 |
| Reach ≥MCI 5–10 y | 3.855 | 3.420 | **-0.436** | 4.041 |
| Reach ≥MCI >10 y | 4.113 | 4.551 | **+0.438** | 9.602 |

### 3. Guards — did the categorical side pay for it?

HIGHER is better here — read the last column as pairs worse.

| metric | arm mean | control mean | d (mean) | sd(d) | pairs better |
|---|---:|---:|---:|---:|---|
| median transition AUC | 0.6599 | 0.6942 | **-0.0343** | 0.0050 | 0/3 |
| trajectory Jaccard | 0.6475 | 0.6464 | **+0.0011** | 0.0127 | 2/3 |
| auc[Death @10y] | 0.8335 | 0.8443 | **-0.0108** | 0.0350 | 1/3 |
| auc[Death @2y] | 0.6835 | 0.6637 | **+0.0199** | 0.0652 | 2/3 |
| auc[Death @3y] | 0.7535 | 0.7558 | **-0.0023** | 0.0354 | 1/3 |
| auc[Death @5y] | 0.7831 | 0.7872 | **-0.0041** | 0.0394 | 2/3 |
| auc[Dementia @10y] | 0.8502 | 0.8677 | **-0.0175** | 0.0031 | 0/3 |
| auc[Dementia @1y] | 0.8122 | 0.8115 | **+0.0007** | 0.0349 | 2/3 |
| auc[Dementia @2y] | 0.8823 | 0.8841 | **-0.0018** | 0.0167 | 2/3 |
| auc[Dementia @3y] | 0.8844 | 0.8877 | **-0.0033** | 0.0145 | 2/3 |
| auc[Dementia @5y] | 0.8844 | 0.8884 | **-0.0040** | 0.0108 | 2/3 |
| auc[Reach ≥MCI @10y] | 0.6619 | 0.7081 | **-0.0461** | 0.0198 | 0/3 |
| auc[Reach ≥MCI @1y] | 0.6371 | 0.6414 | **-0.0043** | 0.0288 | 1/3 |
| auc[Reach ≥MCI @2y] | 0.6888 | 0.7096 | **-0.0208** | 0.0080 | 0/3 |
| auc[Reach ≥MCI @3y] | 0.6878 | 0.7136 | **-0.0258** | 0.0124 | 0/3 |
| auc[Reach ≥MCI @5y] | 0.6987 | 0.7286 | **-0.0299** | 0.0047 | 0/3 |

### Per-pair detail

| pair | val loss | d | best iter | d | timing bias (y) | d |
|---|---:|---:|---:|---:|---:|---:|
| TH_L8E120H6_s42 vs L8E120H6_s42 | 10.0863 | -0.0074 | 3600 | +1100 | 1.469 | -1.228 |
| TH_L8E120H6_s43 vs L8E120H6_s43 | 10.0635 | -0.0305 | 4100 | +1600 | 1.581 | -0.911 |
| TH_L8E120H6_s44 vs L8E120H6_s44 | 10.0782 | -0.0137 | 3800 | +1200 | 2.414 | -0.109 |

---

## Pairs not analysed

- TH_L12E120H12_s42 vs L12E120H12_s42: arm log unusable
- TH_L12E120H12_s43 vs L12E120H12_s43: arm log unusable
- TH_L12E120H12_s44 vs L12E120H12_s44: arm log unusable

## What is not in here yet

- decomposition (loss_ce / loss_dt split): present
- downstream eval: present
