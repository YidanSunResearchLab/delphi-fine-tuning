"""
rollout_probe.py -- free-running death behaviour, cheap enough to measure at every evaluation.

WHY THIS EXISTS, with the numbers that forced it. Two checkpoints, same data and same
vocabulary (identical data_sig and vocab_sig), configs four values apart:

    out-radc-testckpt   3L/48d, 800 iters,  dropout 0.2    best_val 9.154
    out-radc-final      4L/64d, 6000 iters, dropout 0.1    best_val 8.670   <- BETTER val loss

Scored through engine.simulate on identical age-80 prefixes, 50 draws each:

    testckpt    2.8% of draws never emit Death, median end age 85.2
    final      89.0% of draws never emit Death, median end age 108.1

The damage is already sitting in the reported results: eval/results_test.json (out-radc-final)
puts the simulated all-cause death CIF at age 85 at 0.0083 against an observed 0.246 -- a
factor of 30 -- and at age 90 at 0.0110 against 0.534, a factor of 49. eval/results_val.json
(out-radc-testckpt) has 0.274 against 0.193 and 0.519 against 0.487, which is the right order.

VALIDATION LOSS CANNOT SEE THIS, and no reweighting of it will. loss_ce is teacher-forced:
every prediction is scored one step ahead of a REAL prefix. A rollout is free-running -- the
model conditions on its own draws for tens of steps, and drifts off the data manifold if the
hazards are wrong. out-radc-final is better on BOTH components (val_ce 2.140 against 2.552,
val_dt 6.536 against 6.602) while being the broken one, so `select_on` over those two terms
will keep choosing it no matter which one is picked. The free-running behaviour has to be
measured as its own quantity, and measured DURING training if it is to select anything.

WHAT IS MEASURED, cheapest first:

  p_death_next   One forward pass per prefix: the renormalised P(Death) for the next token,
                 masked exactly the way engine.simulate masks it (padding, the 21 statics, and
                 No event). No sampling at all, so it carries zero Monte-Carlo noise and its
                 trajectory over training is readable step by step. This is the primary signal:
                 at the two checkpoints above it is 0.0090 against 0.0723 at age 80, and the
                 gap holds at every age from 75 (0.0080/0.0517) to 95 (0.0495/0.1526).

  frac_no_death  Fraction of draws reaching the token or age cap without ever emitting Death.
                 The symptom as it is actually experienced downstream.

  death_cif@A    Simulated all-cause mortality CIF at each of `cif_ages`, from the landmark
                 prefix, against the observed value ON THE SAME SUBJECTS. The RATIO is the
                 gate-able quantity; the raw simulated number is not, because it depends on
                 which subjects the probe happened to select.

RNG DISCIPLINE, and this is not optional. The probe samples, and sampling consumes the same
global generator that drives dropout and the training batch stream. Every call saves and
restores the RNG state of every device it touches and re-seeds itself from a fixed `seed`. So
the probe is BOTH invisible to training -- a run with it on follows the same trajectory as one
with it off, which is what makes a controlled experiment over probe-on runs meaningful -- AND
comparable across evaluations, since the only thing differing between two calls is the weights.
tests/test_rollout_probe.py asserts both properties.

THE OBSERVED REFERENCE is computed once, at construction, on the probe's own subject set, with
the risk set opened at the LANDMARK age rather than at birth: these subjects were selected for
being alive and endpoint-free at the landmark, so that is where they become at risk. All-cause
death has no competing event, so the Kaplan-Meier complement is the cumulative incidence here
and engine.aalen_johansen is not needed -- unlike the AD endpoint, where it is mandatory.

PREFIX SELECTION mirrors the a80 arm of eval/evaluate.py so the numbers are comparable to the
`secondary.cif_calibration` / `secondary.mortality` blocks of results_*.json: the prefix is the
subject's stream truncated at the landmark age, it must carry at least two distinct ages (the
static block sits one day before the baseline visit and owns its own age, so one distinct age
means no real visit at all), and it must contain neither endpoint.
"""
import os

import numpy as np
import torch

from . import vocab as V

DAYS_PER_YEAR = 365.25
PAD_AGE = -10000.0


def km_death_cif(exit_age, died, entry_age, grid_ages):
    """1 - KM for all-cause mortality, left-truncated at `entry_age`.

    All-cause death is the terminal event, so nothing competes with it and the Kaplan-Meier
    complement IS the cumulative incidence. This is exactly the case where the Aalen-Johansen
    machinery in engine.py is NOT required, and saying so here stops someone reaching for it.
    """
    exit_age = np.asarray(exit_age, float)
    died = np.asarray(died, bool)
    entry_age = np.asarray(entry_age, float)
    surv, curve = 1.0, []
    for t in np.unique(exit_age[died]):
        at_risk = int(((entry_age <= t) & (exit_age >= t)).sum())
        if at_risk == 0:
            continue
        surv *= 1.0 - int(((exit_age == t) & died).sum()) / at_risk
        curve.append((t, 1.0 - surv))
    if not curve:
        return np.zeros(len(grid_ages))
    ct = np.array([c[0] for c in curve])
    cv = np.array([c[1] for c in curve])
    return np.array([cv[ct <= g][-1] if (ct <= g).any() else 0.0 for g in grid_ages])


class RolloutProbe:
    """Free-running death behaviour on a fixed set of landmark-truncated prefixes.

    Construct once per run (it selects its subjects and computes the observed reference), then
    call `run(model)` at every evaluation. Returns a flat dict of floats, ready to drop into
    history.json and the wandb metrics.
    """

    def __init__(self, val_data, data_dir, ignore_tokens, landmark_age=80.0, n_subjects=24,
                 n_mc=32, max_new_tokens=128, until_age=110.0, cif_ages=(85.0, 90.0),
                 seed=0, device='cpu'):
        self.landmark_age = float(landmark_age)
        self.n_mc = int(n_mc)
        self.max_new_tokens = int(max_new_tokens)
        self.until_age = float(until_age)
        self.cif_ages = tuple(float(a) for a in cif_ages)
        self.seed = int(seed)
        self.device = device
        # engine.simulate masks padding + the statics + No event. The statics are taken from
        # the TRAINING config rather than from vocab, so a config that changes what it holds
        # out of the loss keeps the probe consistent with itself.
        self.ignore = sorted(set(int(t) for t in ignore_tokens) | {int(V.NO_EVENT)})

        vs_path = os.path.join(data_dir, 'visit_sizes.npy')
        self.visit_sizes = np.load(vs_path) if os.path.exists(vs_path) else None

        t0_days = self.landmark_age * DAYS_PER_YEAR
        data = np.asarray(val_data)
        self.prefixes, pids = [], []
        for pid in np.unique(data[:, 0]):
            rows = data[data[:, 0] == pid]
            rows = rows[rows[:, 1].astype(float) <= t0_days]
            if len(rows) == 0:
                continue
            toks = rows[:, 2].astype(np.int64) + 1                 # DISK -> MODEL
            if len(np.unique(rows[:, 1])) < 2:                     # statics only: no real visit
                continue
            if np.isin(toks, [V.AD_DX, V.DEATH]).any():            # an endpoint is not a prefix
                continue
            self.prefixes.append((toks, rows[:, 1].astype(np.float32)))
            pids.append(int(pid))
            if len(self.prefixes) >= n_subjects:
                break
        self.pids = pids

        self.observed = {}
        subj_csv = os.path.join(data_dir, 'subjects.csv')
        if self.prefixes and os.path.exists(subj_csv):
            import pandas as pd
            sub = pd.read_csv(subj_csv).set_index('projid').reindex(pids)
            died = sub['died'].fillna(0).to_numpy().astype(bool)
            exit_age = np.where(died, sub['age_death'].to_numpy(float),
                                sub['age_last_obs'].to_numpy(float))
            # Risk opens at the landmark, not at age_bl: selection already conditioned on being
            # alive and endpoint-free there.
            entry = np.full(len(pids), self.landmark_age)
            keep = np.isfinite(exit_age) & (exit_age >= self.landmark_age)
            cif = km_death_cif(exit_age[keep], died[keep], entry[keep], self.cif_ages)
            self.observed = {a: float(c) for a, c in zip(self.cif_ages, cif)}
            self.n_observed = int(keep.sum())
        else:
            self.n_observed = 0

    # ------------------------------------------------------------------ rng isolation
    def _save_rng(self):
        st = {'cpu': torch.get_rng_state()}
        if torch.cuda.is_available():
            st['cuda'] = torch.cuda.get_rng_state_all()
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            st['mps'] = torch.mps.get_rng_state()
        return st

    def _restore_rng(self, st):
        torch.set_rng_state(st['cpu'])
        if 'cuda' in st:
            torch.cuda.set_rng_state_all(st['cuda'])
        if 'mps' in st:
            torch.mps.set_rng_state(st['mps'])

    # ------------------------------------------------------------------ the measurement
    @torch.no_grad()
    def run(self, model):
        if not self.prefixes:
            return {}
        was_training = model.training
        model.eval()
        st = self._save_rng()
        try:
            torch.manual_seed(self.seed)
            out = self._measure(model)
        finally:
            self._restore_rng(st)
            if was_training:
                model.train()
        return out

    def _measure(self, model):
        dev = self.device
        p_next, no_death, end_ages = [], [], []
        death_ages, truncated = [], []

        for toks, ages in self.prefixes:
            x1 = torch.as_tensor(toks, dtype=torch.long, device=dev)[None]
            a1 = torch.as_tensor(ages, dtype=torch.float32, device=dev)[None]

            # --- deterministic: P(Death) for the very next token, renormalised like engine ---
            logits, _, _ = model(x1, a1)
            lg = logits[0, -1].float().clone()
            lg[self.ignore] = -float('inf')
            p_next.append(torch.softmax(lg, -1)[V.DEATH].item())

            # --- sampled: the free-running rollout ---
            gi, ga, _ = model.generate(
                x1.repeat(self.n_mc, 1), a1.repeat(self.n_mc, 1),
                max_new_tokens=self.max_new_tokens,
                max_age=self.until_age * DAYS_PER_YEAR,
                termination_tokens=list(V.TERMINATION_TOKENS),
                extra_ignore=[V.NO_EVENT],
                repeatable_tokens=list(V.REPEATABLE_TOKENS),
                visit_sizes=self.visit_sizes,
            )
            gi, ga = gi.cpu().numpy(), ga.cpu().numpy()
            real = ga > PAD_AGE + 1
            hit = (gi == V.DEATH) & real
            no_death.append(~hit.any(1))
            da = np.where(hit, ga, np.inf).min(1) / DAYS_PER_YEAR
            death_ages.append(da)
            ea = np.where(real, ga, -np.inf).max(1) / DAYS_PER_YEAR
            end_ages.append(ea)
            truncated.append(np.isinf(da) & (ea < self.until_age - 0.5))

        nd = np.concatenate(no_death)
        da = np.concatenate(death_ages)
        ea = np.concatenate(end_ages)
        tr = np.concatenate(truncated)

        out = {
            'probe/p_death_next': float(np.mean(p_next)),
            'probe/frac_no_death': float(nd.mean()),
            'probe/median_end_age': float(np.median(ea)),
            # A draw that stopped early without dying cannot say whether death happened by age
            # A, so report the rate rather than silently counting it as a survivor.
            'probe/frac_truncated': float(tr.mean()),
            'probe/n_draws': int(da.size),
        }
        for a in self.cif_ages:
            sim = float((da <= a).mean())
            out[f'probe/death_cif_{a:g}'] = sim
            obs = self.observed.get(a)
            if obs:
                out[f'probe/death_cif_ratio_{a:g}'] = sim / obs
        return out

    def describe(self):
        obs = "  ".join(f"{a:g}y {self.observed.get(a, float('nan')):.3f}" for a in self.cif_ages)
        return (f"[probe] {len(self.prefixes)} age-{self.landmark_age:g} prefixes x {self.n_mc} draws "
                f"| observed death CIF on these {self.n_observed} subjects: {obs}")
