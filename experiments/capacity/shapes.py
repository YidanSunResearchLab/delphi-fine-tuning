"""
shapes.py -- the single source of truth for the capacity sweep.

Every other file in this folder (configs, manifest, sbatch array, collect, analyze)
derives from SHAPES below. Change a shape here and regenerate; never edit a generated
config by hand.

WHY THIS SWEEP EXISTS
---------------------
The delivered model (out-delphi2m-dedup-mask-s42) is 12L/12H/120d = 2,104,320 params,
trained on 16,262 patients / ~500k events after `filter_cohort`. Its best checkpoint
lands at iter 2750 of 5000 -- val loss bottoms at 55% of the schedule and rises for the
rest. That is a capacity/data mismatch, and the architecture was never chosen for this
dataset: it was copied verbatim from the upstream Delphi-2M demo config, whose vocab is
1270 and whose cohort is ~100x larger.

The sweep asks one question: **does the val-loss bottom move later (and lower) as capacity
comes down, and where is the knee?**

WHAT IS HELD CONSTANT
---------------------
Everything except (n_layer, n_head, n_embd) and the seed. Same dataset, same cohort
filter, same ignore_tokens, same LR schedule, same block_size, same eval cadence.

NOTE ON eval_interval: `estimate_loss()` in train.py draws batches with the GLOBAL torch
RNG (200 iters x 2 splits = 400 randint calls per eval), so the eval cadence PERTURBS the
training batch stream. eval_interval is therefore NOT a free variable -- it is pinned to
100 for every arm of the sweep, including the re-run of the delivered architecture. That
re-run, not the delivered checkpoint (which used 250), is the sweep's reference arm.
"""

# tag -> (n_layer, n_head, n_embd, why this point is in the sweep)
SHAPES = {
    "L12E120H12": (12, 12, 120, "delivered architecture, re-run as the reference arm"),
    "L12E120H6":  (12,  6, 120, "head-geometry control: SAME params, head_dim 20 not 10"),
    "L8E120H6":   ( 8,  6, 120, "depth down, width held"),
    "L8E96H6":    ( 8,  6,  96, "the shape proposed in review (~0.9M)"),
    "L6E96H6":    ( 6,  6,  96, "depth down again"),
    "L6E64H4":    ( 6,  4,  64, "small end -- is the knee already behind us?"),
}

SEEDS = (42, 43, 44)          # 3 seeds/shape: a moved bottom must beat seed noise
VOCAB_SIZE = 111
BLOCK_SIZE = 96


def n_params(n_layer, n_head, n_embd, vocab_size=VOCAB_SIZE, bias=False):
    """Exact parameter count for delphi.model.Delphi, tied wte/lm_head counted once.

    Verified against the live model in verify.py. n_head does NOT appear -- c_attn is
    n_embd x 3*n_embd regardless of how the heads are split.
    """
    per_block = 12 * n_embd ** 2 + (2 * n_embd if not bias else 2 * 2 * n_embd)
    if bias:
        per_block += 3 * n_embd + n_embd + 4 * n_embd + n_embd   # c_attn,c_proj,c_fc,c_proj
    return (vocab_size * n_embd            # wte (tied with lm_head)
            + n_embd ** 2                  # AgeEncoding.linear (always bias=False)
            + per_block * n_layer
            + n_embd)                      # ln_f weight


def head_dim(tag):
    nl, nh, ne, _ = SHAPES[tag]
    return ne // nh


def manifest():
    """[(idx, tag, seed, n_layer, n_head, n_embd, n_params)] -- array-task order."""
    rows = []
    i = 1
    for tag, (nl, nh, ne, _) in SHAPES.items():
        for s in SEEDS:
            rows.append((i, tag, s, nl, nh, ne, n_params(nl, nh, ne)))
            i += 1
    return rows


if __name__ == "__main__":
    print(f"{'tag':<12}{'L':>3}{'H':>3}{'d':>5}{'head_dim':>10}{'params':>12}{'  vs baseline':>14}")
    base = n_params(*SHAPES["L12E120H12"][:3])
    for tag, (nl, nh, ne, why) in SHAPES.items():
        p = n_params(nl, nh, ne)
        print(f"{tag:<12}{nl:>3}{nh:>3}{ne:>5}{ne//nh:>10}{p:>12,}{p/base:>13.3f}x")
        print(f"{'':<12}{why}")
    print(f"\n{len(SHAPES)} shapes x {len(SEEDS)} seeds = {len(manifest())} runs")
