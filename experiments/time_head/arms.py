"""
arms.py -- the single source of truth for the time-head experiment.

Everything else here (configs, manifest, sbatch array, analysis) derives from ARMS below.
Change an arm here and regenerate; never edit a generated config by hand.

WHY THIS EXPERIMENT EXISTS
--------------------------
The capacity sweep's finding 2, and its recommendation 4: `loss_dt` is 73% of the reported
validation loss and is INERT -- a 6.9x parameter range moved it by 0.0071 nats, 0.10% of
itself, with every shape inside 2-3 sd of the others.

`time_head_probe.py` says why that is plausible. Delphi has ONE output projection
(`lm_head`, vocab-wide, weight-tied to the token embedding) and both objectives are read
off it:

    loss_ce = cross_entropy(logits, next_token)         -- WHICH event
    lambda  = logsumexp(logits), floored via t_min      -- WHEN, as one scalar
    loss_dt = lambda*t - log(lambda)                       (exponential NLL)

So "when" is a single number per position AND a deterministic function of the same logits
that decide "which". It has no parameters of its own. Adding capacity to the trunk cannot
give the timing objective anywhere to put it.

THE ARM: give it somewhere. `time_head=True` (delphi/model.py) adds an independent scalar
projection, log lambda = time_head(ln_f(x)), and reads the intensity off that instead of off
the token logits. 121 parameters at n_embd=120 -- 0.006% of the delivered model.

    Does loss_dt move when it has its own head, and does the timing get better downstream?

If it does not move even then, the limit is not parameterisation -- it is the
competing-exponential likelihood, the t_min floor, or the no-event token rate, and those are
the things to attack next. Either answer closes off a branch.

WHAT IS HELD CONSTANT
---------------------
Everything. Each arm's config is the capacity sweep's config for the same architecture,
byte-for-byte, plus the single line `time_head = True` -- gen.py builds it by reading that
file, and verify.py check [6] fails if the diff is anything other than that one key.

THE CONTROLS ARE NOT RE-RUN. They are the existing capacity-sweep runs for the same shape
and seed, and this is exact rather than approximate:

  * the model change adds no parameter and consumes no RNG when time_head=False (proved by
    tests/golden.py at atol=0, 29 arrays), so those runs are still reproducible as-is;
  * `Delphi.__init__` builds the time head AFTER the shared init has drawn its numbers, so a
    time_head=True model is bit-identical to its control on every shared parameter at the
    same seed (verify.py check [3]);
  * train.py re-seeds the batch RNG (`manual_seed(seed + iter_num)`) before the training
    loop, so both members of a pair also see the SAME batch stream.

A pair therefore differs in exactly one thing: the presence of the head. 6 new runs, not 12.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(os.path.dirname(HERE), "capacity")
sys.path.insert(0, CAP)

from shapes import (SHAPES as CAP_SHAPES, SEEDS as CAP_SEEDS,       # noqa: E402
                    VOCAB_SIZE, BLOCK_SIZE, n_params as trunk_params)

# arm tag -> (its control = a capacity-sweep tag, {config overrides}, why this arm is here)
#
# Every arm is "the capacity sweep's run for the same shape and seed, plus these config keys".
# The overrides dict IS the experiment: gen.py writes exactly those lines onto the control's
# config, and verify.py check [6] fails if the resulting diff is anything else.
#
# Two architectures per question, not six. The capacity sweep already established that the
# trunk's size is not what moves loss_dt, so re-sweeping shape would spend CPU-hours
# re-answering a settled question. These are the two shapes anyone would actually ship: the
# delivered one (so the result reads straight against the capacity table) and the one the
# sweep recommends.
ARMS = {
    # ---- question 1: does the timing objective improve if given its own parameters? -------
    "TH_L12E120H12": ("L12E120H12", {"time_head": True},
                      "time head on the DELIVERED architecture -- reads straight against the "
                      "capacity table's reference arm"),
    "TH_L8E120H6":   ("L8E120H6", {"time_head": True},
                      "time head on the shape the sweep recommends (1.41M; best on both "
                      "downstream metrics)"),

    # ---- question 2: how much of the +2.6 y downstream timing bias was a WRONG TARGET? ----
    # dt_ablation.py measured, on the real split, that the delivered `gather` target runs
    # 1.16x (median) / 1.72x (mean) longer than the true visit gaps -- 51.8x with the two data
    # augmentations off -- because 73.6% of scored positions are handed another position's dt,
    # which for same-visit tokens is the gap BEFORE their own visit. `next_event` computes the
    # real forward gap and lands at 0.99x / 0.98x, insensitive to both augmentations.
    # lambda = 1/E[dt], so a target 1.7x too long makes every predicted time 1.7x too long:
    # this is the arm that says how much of the observed bias that accounts for.
    "DT_L12E120H12": ("L12E120H12", {"dt_target": "'next_event'"},
                      "fixed timing target on the DELIVERED architecture"),
    "DT_L8E120H6":   ("L8E120H6", {"dt_target": "'next_event'"},
                      "fixed timing target on the shape the sweep recommends"),
}

# Must match the capacity sweep's seeds: the controls ARE those runs.
SEEDS = CAP_SEEDS


def control(arm):
    return ARMS[arm][0]


def overrides(arm):
    """The config keys this arm sets on top of its control's config. Values are written into
    the generated config verbatim, so strings must carry their own quotes."""
    return ARMS[arm][1]


def why(arm):
    return ARMS[arm][2]


def override_keys():
    """Every key any arm touches -- what verify.py is allowed to see differ."""
    return {k for a in ARMS for k in overrides(a)}


def shape(arm):
    """(n_layer, n_head, n_embd) -- taken from the control, never restated here. Restating it
    is how an arm silently stops being a control led comparison."""
    return CAP_SHAPES[control(arm)][:3]


def head_params(arm):
    """Parameters this arm ADDS to the trunk. Only the time head adds any: nn.Linear(d, 1).
    A dt_target arm changes the objective, not the model, so it adds exactly 0."""
    if overrides(arm).get("time_head"):
        return shape(arm)[2] + 1
    return 0


def n_params(arm):
    nl, nh, ne = shape(arm)
    return trunk_params(nl, nh, ne) + head_params(arm)


def manifest():
    """[(idx, arm, control, seed, n_layer, n_head, n_embd, n_params)] -- array-task order."""
    rows = []
    i = 1
    for arm in ARMS:
        nl, nh, ne = shape(arm)
        for s in SEEDS:
            rows.append((i, arm, control(arm), s, nl, nh, ne, n_params(arm)))
            i += 1
    return rows


if __name__ == "__main__":
    print(f"{'arm':<16}{'control':<13}{'L':>3}{'H':>3}{'d':>5}{'params':>12}{'+params':>9}"
          f"   overrides")
    for arm in ARMS:
        nl, nh, ne = shape(arm)
        p, hp = n_params(arm), head_params(arm)
        ov = ", ".join(f"{k}={v}" for k, v in overrides(arm).items())
        print(f"{arm:<16}{control(arm):<13}{nl:>3}{nh:>3}{ne:>5}{p:>12,}{hp:>9}   {ov}")
        print(f"{'':<16}{why(arm)}")
    print(f"\n{len(ARMS)} arms x {len(SEEDS)} seeds = {len(manifest())} new runs "
          f"({len(manifest())} controls reused from experiments/capacity/runs)")
