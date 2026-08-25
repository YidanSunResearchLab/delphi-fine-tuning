"""
mc_matched.py -- run the Figure-2 Monte-Carlo pass over the TRAINING-COHORT-MATCHED rows only.

Why not `figure2_core.py --build`: that walks every evaluable patient in the split (11,055 in
test), and panel A then throws 59% of them away via `load_all(cohort="matched")`. Scoring only
the rows that survive that filter gives byte-identical panel-A inputs for ~2.4x less compute --
which is what makes the run feasible on a laptop.

Writes the same frame/grids artefacts figure2_core writes, under a `_matched` cache key, so
figure2_panels.compute_A can consume them unchanged.

`--cap-years C` truncates each trajectory at baseline + C years instead of the delivered
`max(105, baseline + 16)`. In `model.generate`, `max_age` gates ONLY the early-stop test and the
final padding -- the sampling loop draws the same `torch.rand(logits.shape)` per step either way,
so a shorter cap yields the SAME trajectory prefix for the same seed, just fewer steps. Risk at
horizon h is therefore bit-identical for any h <= C (verified: --verify-cap below), while the
per-patient cost drops with the number of steps saved.

    !! A capped artefact is ONLY valid for horizons <= cap. The 10-year risk column, the
    !! state-occupancy grids and the transition matrices in it are TRUNCATED, not wrong-by-a-little.
    !! Panel A at H=5 is what it is built for; do not feed it to panels b/c/d.

    python experiments/lr_baseline/mc_matched.py --split test --n-mc 100 --workers 9 --cap-years 6
    python experiments/lr_baseline/mc_matched.py --verify-cap 6     # prove the identity first
"""
import os, sys, time, argparse, logging
import numpy as np
import pandas as pd

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "Delphi-2M")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "figure2"))
import figure2_core as F2                                    # noqa: E402
from delphi import predict_adapter as PA                     # noqa: E402

log = logging.getLogger("mc_matched")
CAP_ENV = "AD_CAP_YEARS"          # read by the worker; see _init_worker_capped


def cache_key(ckpt_sig, n_mc, split, dataset, cap=0):
    tag = "_matched" if not cap else f"_matched_cap{int(cap)}"
    return f"{ckpt_sig}_{dataset}_{split}_n{n_mc}{tag}"


def _init_worker_capped(*a):
    """figure2_core's worker init, plus the trajectory-span cap.

    Wrapping `simulate_trajectory` (rather than editing `_process`) keeps the delivered code path
    untouched: with AD_CAP_YEARS unset this module runs figure2_core exactly as shipped."""
    F2._init_worker(*a)
    cap = float(os.environ.get(CAP_ENV, "0") or 0)
    if cap <= 0:
        return
    orig = PA.simulate_trajectory

    def capped(ad, tokens, ages, until_age_years=100.0, **kw):
        base_y = float(np.max(ages)) / F2.D          # prompt ends at the baseline visit
        return orig(ad, tokens, ages,
                    until_age_years=min(until_age_years, base_y + cap), **kw)
    PA.simulate_trajectory = capped
    F2.PA.simulate_trajectory = capped


def build(ckpt=F2.CKPT, dataset=F2.DATASET, split="test", device="cpu", n_mc=F2.N_MC,
          seed=F2.SEED, workers=9, force=False, cap=0, limit=0, matched=True):
    os.makedirs(F2.CACHE_DIR, exist_ok=True)
    ad = PA.load_model(dict(checkpoint=os.path.join(HERE, ckpt), device=device, seed=seed))
    key = cache_key(ad.ckpt_sig, n_mc, split, dataset, cap)
    if limit:
        key += f"_lim{limit}" + ("" if matched else "_all")
    fpath = os.path.join(F2.CACHE_DIR, f"frame_{key}.pkl")
    gpath = os.path.join(F2.CACHE_DIR, f"grids_{key}.npz")
    if not force and os.path.exists(fpath) and os.path.exists(gpath):
        log.info("cache hit (%s)", key)
        return key

    if matched:
        keep = F2.training_cohort_mask(dataset=dataset, split=split)
        ks = np.where(keep)[0].tolist()
    else:
        _, p2i = PA.load_split(dataset, split)
        ks = list(range(len(p2i)))
    if limit:
        ks = ks[:limit]
    os.environ[CAP_ENV] = str(cap)          # inherited by the spawned workers
    log.info("MC pass: %d patients (%s, %s), n_mc=%d, %d workers, cap=%s y",
             len(ks), split, "matched cohort" if matched else "all", n_mc, workers,
             cap or "none")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    t0 = time.time(); res = []
    payload = [(k, n_mc, seed, F2.HORIZONS) for k in ks]
    with ctx.Pool(workers, initializer=_init_worker_capped,
                  initargs=(ckpt, dataset, split, device, seed)) as pool:
        for i, r in enumerate(pool.imap(F2._process, payload, chunksize=8)):
            if r is not None:
                res.append(r)
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                log.info("  %d/%d  (%d kept)  %.0fs elapsed, ~%.0f min left",
                         i + 1, len(ks), len(res), el, el / (i + 1) * (len(ks) - i - 1) / 60)
    log.info("MC pass done in %.1f min (%d evaluable)", (time.time() - t0) / 60, len(res))

    df = pd.DataFrame([r["row"] for r in res])
    df.to_pickle(fpath)
    np.savez_compressed(gpath, pid=df["pid"].to_numpy(), grid_years=F2.GRID_YEARS,
                        obs=np.stack([r["obs_grid"] for r in res]),
                        known=np.stack([r["known"] for r in res]).astype(bool),
                        occ=np.stack([r["occ"] for r in res]).astype(np.float32),
                        sim=np.stack([r["sim_grid"] for r in res]).astype(np.int8),
                        fp=np.stack([r["fp"] for r in res]).astype(np.float32))
    log.info("cached -> %s", fpath)
    return key


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-mc", type=int, default=F2.N_MC)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cap-years", type=float, default=0,
                    help="truncate trajectories at baseline + C years (valid for horizons <= C)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="every patient, not just the training cohort")
    a = ap.parse_args()
    build(split=a.split, n_mc=a.n_mc, workers=a.workers, device=a.device, force=a.force,
          cap=a.cap_years, limit=a.limit, matched=not a.all)
    print("done.")
