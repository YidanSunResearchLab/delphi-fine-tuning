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

# arm tag -> (its control = a capacity-sweep tag, why this arm is here)
#
# Two architectures, not six. The question is whether the timing objective responds to being
# given its own head, and the capacity sweep already showed that the trunk's size is not what
# moves loss_dt -- so sweeping shape again would spend GPU-hours re-answering a settled
# question. These two are the shapes anyone would actually ship: the delivered one (so the
# result reads directly against the whole capacity table) and the one the sweep recommends.
ARMS = {
    "TH_L12E120H12": ("L12E120H12",
                      "time head on the DELIVERED architecture -- reads straight against the "
                      "capacity table's reference arm"),
    "TH_L8E120H6":   ("L8E120H6",
                      "time head on the shape the sweep recommends (1.41M; best on both "
                      "downstream metrics)"),
}

# Must match the capacity sweep's seeds: the controls ARE those runs.
SEEDS = CAP_SEEDS


def control(arm):
    return ARMS[arm][0]


def shape(arm):
    """(n_layer, n_head, n_embd) -- taken from the control, never restated here. Restating it
    is how an arm silently stops being a control led comparison."""
    return CAP_SHAPES[control(arm)][:3]


def head_params(n_embd):
    """The added head: nn.Linear(n_embd, 1, bias=True)."""
    return n_embd + 1


def n_params(arm):
    nl, nh, ne = shape(arm)
    return trunk_params(nl, nh, ne) + head_params(ne)


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
    print(f"{'arm':<16}{'control':<13}{'L':>3}{'H':>3}{'d':>5}{'params':>12}{'head':>7}"
          f"{'  overhead':>12}")
    for arm in ARMS:
        nl, nh, ne = shape(arm)
        p, hp = n_params(arm), head_params(ne)
        print(f"{arm:<16}{control(arm):<13}{nl:>3}{nh:>3}{ne:>5}{p:>12,}{hp:>7}"
              f"{100 * hp / p:>11.3f}%")
        print(f"{'':<16}{ARMS[arm][1]}")
    print(f"\n{len(ARMS)} arms x {len(SEEDS)} seeds = {len(manifest())} new runs "
          f"({len(manifest())} controls reused from experiments/capacity/runs)")
