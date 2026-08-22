"""
ad_engine.py -- thin wrapper around the trained Delphi-AD model for evaluation/figures.

IMPORTANT conventions for the REBUILT data (one consistent space):
  * Tokens you feed the model at inference are MODEL-space = labels.csv index.
    (The +1 padding shift only happens inside get_batch during TRAINING.)
  * So: Male=2, Female=3, ... cognitive Normal=106 / Impaired=107 / MCI=108 /
    Dementia=109, Death=110.  P(Dementia next) = softmax(logits)[109].
  * model.generate now defaults termination_tokens=[110]=Death (model space); we
    still pass [DEATH]=[110] explicitly for clarity.
"""
import torch
from delphi.model import Delphi, DelphiConfig

# model-space token ids (== labels.csv row index)
MALE, FEMALE = 2, 3
NORMAL, IMPAIRED, MCI, DEMENTIA, DEATH = 106, 107, 108, 109, 110
COGNITIVE = {"Normal": NORMAL, "Impaired": IMPAIRED, "MCI": MCI, "Dementia": DEMENTIA}


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["model_args"]
    model = Delphi(DelphiConfig(**args)).to(device).eval()
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}  # strip torch.compile prefix
    model.load_state_dict(sd)
    return model, args


@torch.no_grad()
def next_event_probs(model, tokens, ages, device="cpu", exclude_nonevent=True):
    """Softmax over the vocabulary for the NEXT event after the given history.
    If exclude_nonevent=True (default), zeroes out Padding (0) and No-event (1)
    and renormalizes so the distribution answers 'given a real event happens next,
    what is it?' — without this, the No-event token absorbs ~75% of probability."""
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    age = torch.tensor([ages], dtype=torch.float32, device=device)
    logits, _, _ = model(idx, age)
    p = torch.softmax(logits[0, -1], -1).cpu().numpy()
    if exclude_nonevent:
        p[0] = 0.0  # Padding
        p[1] = 0.0  # No event
        total = p.sum()
        if total > 0:
            p = p / total
    return p


@torch.no_grad()
def p_event_by_age(model, tokens, ages, target=DEMENTIA, by_age_years=85,
                   n_samples=128, device="cpu", seed=0):
    """Forward-sample n trajectories; return fraction that reach `target` (e.g. Dementia)
    at or before `by_age_years`. This is the robust progression metric (see project notes)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    idx = torch.tensor([tokens] * n_samples, dtype=torch.long, device=device)
    age = torch.tensor([ages] * n_samples, dtype=torch.float32, device=device)
    gi, ga, _ = model.generate(idx, age, max_new_tokens=64, max_age=by_age_years * 365.25,
                               termination_tokens=[DEATH])
    hit = ((gi == target) & (ga <= by_age_years * 365.25)).any(1)
    return float(hit.float().mean())


@torch.no_grad()
def p_dementia_given_alive(model, tokens, ages, by_age_years=90, n_samples=128, device="cpu", seed=0):
    """Competing-risk-SAFE progression metric: P(Dementia by age | ALIVE at that age).
    Conditioning on survival means a lethal comorbidity that kills before dementia can't
    masquerade as 'protective' (the flip you get from the naive P(Dementia by age))."""
    torch.manual_seed(seed)
    X = by_age_years * 365.25
    idx = torch.tensor([tokens] * n_samples, dtype=torch.long, device=device)
    age = torch.tensor([ages] * n_samples, dtype=torch.float32, device=device)
    gi, ga, _ = model.generate(idx, age, max_new_tokens=64, max_age=X, termination_tokens=[DEATH])
    def first(tok):
        m = (gi == tok) & (ga >= 0); big = torch.full_like(ga, 1e18)
        return torch.where(m, ga, big).min(1).values
    dem, dth = first(DEMENTIA), first(DEATH)
    alive = dth > X
    n_alive = int(alive.sum().item())
    return float(((dem <= X) & alive).sum().item()) / max(n_alive, 1)
