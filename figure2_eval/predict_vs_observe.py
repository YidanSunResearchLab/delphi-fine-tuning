"""
predict_vs_observe.py -- 预测发生率 vs 实际发生率，不做任何 at-risk 条件化。

WHY THIS EXISTS, AND HOW IT DIFFERS FROM figure2 panel a2.

panel a2 沿用上游的框架：每一行只计入"基线时还不在该状态"的人。那个条件化有它的道理（一个已经
在某状态里的人无法"到达"它），但它有两个代价，而第二个代价是这个文件存在的原因：

  1. 每一行的分母都不同（80 / 362 / 390 / 416 ...），不能横向比较；
  2. 它会制造**方向相反**的终点。`MMSE Normal` 那一行的 at-risk 全是基线低于 27 分的人，
     所以那一行问的是"会不会**好转**"，而同一张图上其它行问的是"会不会**恶化**"。整张图里
     只有它（和 `cogn_global ≥0`）是这样，而那恰好就是塌得最狠的两行。

这里换成一个**没有条件化**的量，对每个 token 都问同一个问题：

    在基线之后的 H 年窗口里，这个 token 到底出现没出现（实际），
    以及模型预测它出现的概率是多少（100 条 MC 轨迹里出现的比例）。

所有 416 个可评估患者都计入每一个 token，不做任何筛选。两边读的是同一个窗口、同一条规则。

WHAT IS *NOT* REMOVED, AND CANNOT BE. 预测必须有一个起点 —— 模型得站在某个时刻往前看。这里
仍然是每个患者的**首次访视**（statics + 基线那一访），和 figure2 一致，因为那是"只看入组时的
信息能预测多远"这个预后问题的正确起点。要换成"站在每一次访视往前看 1 年"是另一个量
（内置 evaluate_auc 测的就是那个，见 README §4），不是这个文件。

CENSORING. 随访不足 H 年、且窗口内没发生、也没死的人，对这个 token 是**未知**而不是"没发生"，
按 figure2_core.labels_at_h 的同一条规则**丢弃**。把他们算成阴性会系统性压低实测率，正好朝着
让模型好看的方向。每个 token 的有效 n 都写进 CSV。

病理 token 被排除（不是被算成 0）。它们在数据里只出现在死亡当天或之后，而 engine 禁止采样它们
（vocab.py 的 SAMPLE_BLOCKED），所以预测值会是结构性的 0 —— 那是我们的设计选择，不是模型的性质，
画上去只会误导。

    python predict_vs_observe.py                 # 全量，约 6 分钟（8 workers）
    python predict_vs_observe.py --limit 40      # 冒烟
    python predict_vs_observe.py --horizon 2     # 换窗口
"""
import argparse
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from radc_delphi import engine as EN  # noqa: E402
from figure2 import figure2_core as F2  # noqa: E402
from figure2 import plotting_style as ps  # noqa: E402


def _use_cjk_font():
    """图上有中文标签，DejaVu Sans 没有 CJK 字形（会静默画成方框并刷一屏 warning）。

    按可得性挑一个系统中文字体，找不到就退回默认并**明确提示**——一张标签变方框的图比一张
    英文标签的图更糟，所以失败要出声，不能静默。
    """
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("Hiragino Sans GB", "PingFang HK", "Songti SC", "STHeiti",
                 "Arial Unicode MS", "Heiti TC", "Noto Sans CJK SC"):
        if name in have:
            matplotlib.rcParams["font.sans-serif"] = [name] + \
                list(matplotlib.rcParams["font.sans-serif"])
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    print("WARNING: 没找到中文字体，图上的中文会变成方框。装一个 Noto Sans CJK 或改用英文标签。",
          file=sys.stderr)
    return None

D = EN.DAYS_PER_YEAR
CKPT = "../delphi/Delphi-ROSMAP/ckpt.pt"
DATASET = "rosmap"
OUT = os.path.join(HERE, "results", "predict_vs_observe")

# 家族着色。三个 categorical 色相用 dataviz 的 validate_palette.js 验过（light 模式）：
#   node scripts/validate_palette.js "#7B3294,#56B4E9,#009E73" --mode light --pairs all
#   -> 亮度带 PASS，色度下限 PASS，CVD 全对最差 ΔE 16.6 (deutan) / 11.2 (tritan) PASS，
#      常视觉最差 18.4 PASS，#56B4E9 对底色 2.25:1 触发 contrast WARN。
# 那条 WARN 是**义务**而非可忽略项，用两种冗余编码兑付：偏差最大的点直接标注 token 名，
# 且每张图都写一份 source-data CSV。dark 模式没有验过也不是目标（这套图只出白底 PDF/PNG）。
# Death 用中性灰而**不是**第四个色相：它是吸收态、是参照，不是并列的一个序列
# （沿用 plotting_style 的同一处理），并且总是直接标注。
FAMILY_COLORS = {
    "Clinical measures": "#7B3294",
    "Events": "#56B4E9",
    "Medications": "#009E73",
    "Death": "#4d4d4d",
}
FAMILY_ORDER = ["Clinical measures", "Events", "Medications", "Death"]


def family_of(res, tok):
    if tok == res.DEATH:
        return "Death"
    if tok in res.SCALE_OF:
        return "Clinical measures"
    if tok in res.MED_IDS:
        return "Medications"
    return "Events"          # AD_DX / STROKE / DEPRESSION / *_ONSET


# ---------------------------------------------------------------- worker
_G = {}


def _init(ckpt, dataset, device):
    import torch
    torch.set_num_threads(1)
    eng = F2._engine(os.path.join(HERE, ckpt), dataset, device)
    data, p2i, _ = eng.load_split(F2.SPLIT)
    _G.update(eng=eng, data=data, p2i=p2i)


def _one(args):
    """一个患者：每个 token 的 (实际是否发生, 预测概率, 是否已知)。"""
    k, n_mc, seed, horizon, toks_scored = args
    eng, data, p2i = _G["eng"], _G["data"], _G["p2i"]
    ages, toks, _pid = eng.stream(data, p2i, k)
    bp = F2._baseline(ages, toks)
    if bp is None:
        return None
    base, pmask, _b = bp
    a = np.asarray(ages, float)
    t = np.asarray(toks)
    last = float(a[a > 0].max())
    fu = (last - base) / D
    hd = base + horizon * D

    T, A = eng.simulate(t[pmask], a[pmask], n_mc=n_mc, seed=seed + k,
                        until_age_years=max(105.0, base / D + horizon + 1.0))
    real = A > EN.PAD_AGE + 1
    win_sim = real & (A > base) & (A <= hd)
    win_obs = (a > base) & (a <= hd)

    died_by_h = bool((t[win_obs] == eng.res.DEATH).any())
    obs = np.zeros(len(toks_scored), dtype=np.int8)
    pred = np.zeros(len(toks_scored), dtype=np.float32)
    known = np.zeros(len(toks_scored), dtype=bool)
    for i, tok in enumerate(toks_scored):
        o = bool((t[win_obs] == tok).any())
        obs[i] = int(o)
        pred[i] = float(((T == tok) & win_sim).any(1).mean())
        # 同 labels_at_h：发生了、或随访够长、或已死，才算"已知"
        known[i] = o or (fu >= horizon - 1e-6) or died_by_h
    return obs, pred, known


# ---------------------------------------------------------------- 口径扫描
# 一次 MC pass 出全部 horizon + "下一次访视"窗口 + "每次发射的组成"三种口径。
# rollout 是同一批，只有窗口/归一化不同，所以没必要为每个 horizon 重跑一遍。
SWEEP_H = [1, 2, 3, 5, 10]


def _one_sweep(args):
    """一个患者：多个窗口下的 (实际计数, 预测计数, 是否已知)，外加临床 token 总数。"""
    k, n_mc, seed, toks_scored = args
    eng, data, p2i = _G["eng"], _G["data"], _G["p2i"]
    ages, toks, _pid = eng.stream(data, p2i, k)
    bp = F2._baseline(ages, toks)
    if bp is None:
        return None
    base, pmask, _b = bp
    a = np.asarray(ages, float); t = np.asarray(toks)
    last = float(a[a > 0].max()); fu = (last - base) / D
    skip = set(eng.ignore_tokens) | {eng.res.NO_EVENT}

    # "下一次访视"窗口：到下一个**真实**访视年龄为止，没有下一访就排除
    va = np.unique(a[(a > base) & ~np.isin(t, list(skip))])
    nxt = float(va[0]) if len(va) else np.nan

    T, A = eng.simulate(t[pmask], a[pmask], n_mc=n_mc, seed=seed + k,
                        until_age_years=max(105.0, base / D + max(SWEEP_H) + 1.0))
    real = A > EN.PAD_AGE + 1
    clin_sim = real & ~np.isin(T, list(skip))
    clin_obs = ~np.isin(t, list(skip))

    out = {}
    windows = [(f"{h}y", base + h * D, h) for h in SWEEP_H] + [("nextvisit", nxt, None)]
    for lab, hd, h in windows:
        if not np.isfinite(hd):
            continue
        wo = (a > base) & (a <= hd)
        ws = real & (A > base) & (A <= hd)
        died = bool((t[wo] == eng.res.DEATH).any())
        hy = (hd - base) / D
        known_all = (fu >= hy - 1e-6) or died
        obs = np.array([bool((t[wo] == tok).any()) for tok in toks_scored], dtype=np.int8)
        pred = np.array([float(((T == tok) & ws).any(1).mean()) for tok in toks_scored],
                        dtype=np.float32)
        kn = np.array([bool(o) or known_all for o in obs], dtype=bool)
        out[lab] = (obs, pred, kn,
                    int((clin_obs & wo).sum()),                 # 实际临床 token 数
                    float((clin_sim & ws).sum(1).mean()))       # 预测临床 token 数
    return out


def sweep(n_mc=100, seed=42, limit=0, workers=8, device="cpu", min_events=10):
    """三种口径的对照，这是 A3（"改报告口径"）的实际检验。"""
    eng = F2._engine(os.path.join(HERE, CKPT), DATASET, device)
    res = eng.res
    scored = [t for t in range(res.VOCAB_SIZE)
              if t not in set(res.IGNORE_TOKENS) and t != res.NO_EVENT
              and t not in set(res.PATHOLOGY_IDS)]
    data, p2i, _ = eng.load_split(F2.SPLIT)
    N = len(p2i) if not limit else min(len(p2i), limit)
    payload = [(k, n_mc, seed, scored) for k in range(N)]
    ctx = mp.get_context("spawn")
    t0 = time.time(); rows = []
    with ctx.Pool(workers, initializer=_init, initargs=(CKPT, DATASET, device)) as pool:
        for i, r in enumerate(pool.imap(_one_sweep, payload, chunksize=8)):
            if r is not None:
                rows.append(r)
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{N}  {el:.0f}s，约剩 {el/(i+1)*(N-i-1):.0f}s")
    print(f"完成 {len(rows)} 人，{(time.time()-t0)/60:.1f} 分钟")

    recs = []
    for lab in [f"{h}y" for h in SWEEP_H] + ["nextvisit"]:
        have = [r[lab] for r in rows if lab in r]
        if not have:
            continue
        OBS = np.stack([x[0] for x in have]); PRED = np.stack([x[1] for x in have])
        KN = np.stack([x[2] for x in have])
        nobs = np.array([x[3] for x in have], float)
        npred = np.array([x[4] for x in have], float)
        for i, tok in enumerate(scored):
            m = KN[:, i]
            if m.sum() == 0:
                continue
            recs.append(dict(window=lab, token=int(tok), display=res.display(tok),
                             family=family_of(res, tok), n_known=int(m.sum()),
                             n_events=int(OBS[m, i].sum()),
                             observed=float(OBS[m, i].mean()),
                             predicted=float(PRED[m, i].mean()),
                             # 组成口径：该 token 占"被发射的临床 token"的份额
                             share_obs=float(OBS[:, i].sum() / max(nobs.sum(), 1e-9)),
                             share_pred=float(PRED[:, i].sum() / max(npred.sum(), 1e-9)),
                             n_clin_obs=float(nobs.mean()), n_clin_pred=float(npred.mean())))
    df = pd.DataFrame(recs)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "reporting_frames_data.csv"), index=False)

    print("\n=== A3 的检验：换报告口径，低估会消失吗 ===")
    print(f"{'窗口':12s}{'临床token数 实/预':>22s}{'速率比':>9s}"
          f"{'  每token低估倍数(中位)':>24s}{'组成口径低估(中位)':>22s}")
    for lab in [f"{h}y" for h in SWEEP_H] + ["nextvisit"]:
        d = df[(df.window == lab) & (df.n_events >= min_events)]
        if not len(d):
            continue
        o, pr = d.n_clin_obs.iloc[0], d.n_clin_pred.iloc[0]
        u = np.nanmedian(d.observed / d.predicted.replace(0, np.nan))
        us = np.nanmedian(d.share_obs / d.share_pred.replace(0, np.nan))
        nm2 = "下一次访视" if lab == "nextvisit" else lab
        print(f"{nm2:12s}{o:10.2f} /{pr:7.2f}{pr/o:9.2f}x{u:20.1f}x{us:20.2f}x")
    return df, len(rows)


# ---------------------------------------------------------------- build
def build(n_mc=100, seed=42, limit=0, horizon=5, workers=8, device="cpu"):
    eng = F2._engine(os.path.join(HERE, CKPT), DATASET, device)
    res = eng.res
    scored = [t for t in range(res.VOCAB_SIZE)
              if t not in set(res.IGNORE_TOKENS) and t != res.NO_EVENT
              and t not in set(res.PATHOLOGY_IDS)]
    data, p2i, _ = eng.load_split(F2.SPLIT)
    N = len(p2i) if not limit else min(len(p2i), limit)
    print(f"打分 {len(scored)} 个 token × {N} 名患者 × {n_mc} 条轨迹，窗口 {horizon} 年")

    payload = [(k, n_mc, seed, horizon, scored) for k in range(N)]
    ctx = mp.get_context("spawn")
    t0 = time.time()
    rows = []
    with ctx.Pool(workers, initializer=_init, initargs=(CKPT, DATASET, device)) as pool:
        for i, r in enumerate(pool.imap(_one, payload, chunksize=8)):
            if r is not None:
                rows.append(r)
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{N}  {el:.0f}s 已用，约剩 {el/(i+1)*(N-i-1):.0f}s")
    print(f"完成：{len(rows)} 名可评估患者，{(time.time()-t0)/60:.1f} 分钟")

    OBS = np.stack([r[0] for r in rows])
    PRED = np.stack([r[1] for r in rows])
    KN = np.stack([r[2] for r in rows])

    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    out = []
    for i, tok in enumerate(scored):
        m = KN[:, i]
        if m.sum() == 0:
            continue
        y, p = OBS[m, i].astype(int), PRED[m, i].astype(float)
        # AUC 只在两类都有的时候有定义；受试者级 bootstrap 给 CI（MC 预测的不确定性不在其中）
        auc = lo = hi = np.nan
        if 0 < y.sum() < len(y):
            auc = float(roc_auc_score(y, p))
            bs = []
            for _ in range(400):
                idx = rng.integers(0, len(y), len(y))
                if 0 < y[idx].sum() < len(idx):
                    bs.append(roc_auc_score(y[idx], p[idx]))
            if bs:
                lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
        out.append(dict(token=int(tok), name=res.NAMES[tok], display=res.display(tok),
                        family=family_of(res, tok), n_known=int(m.sum()),
                        n_events=int(OBS[m, i].sum()),
                        observed=float(OBS[m, i].mean()),
                        predicted=float(PRED[m, i].mean()),
                        auc=auc, auc_lo=lo, auc_hi=hi))
    df = pd.DataFrame(out).sort_values("observed", ascending=False).reset_index(drop=True)
    df["ratio"] = df["predicted"] / df["observed"].replace(0, np.nan)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"predict_vs_observe_h{horizon}_data.csv"), index=False)
    return df, len(rows)


# ---------------------------------------------------------------- plot
# 两张独立的图，不是一张双栏图。它们的自然几何差太远：校准散点要正方（恒等线必须是 45 度，
# 否则"偏离多远"读不出来），成对横条要随 token 数长高。挤进一个 gridspec 的结果是散点被压成
# 一个小方块——第一版就是这样，渲染出来才看见。
def _ascii(x):
    """中文字体（Hiragino Sans GB 等）没有 U+2212 减号的字形，会画成方框。

    matplotlib 不做逐字形回退，所以这里把 U+2212 换成 ASCII 连字符，而不是指望字体。
    """
    return (str(x).replace("\u2212", "-").replace("\u2264", "<=")
            .replace("\u2265", ">=").replace("\u2013", "-"))


def _logfmt(v, _pos):
    """log 轴刻度用纯小数文本，绕开 mathtext 的负号（同一个字形问题），并且全轴格式一致
    ——第一版混用了 "0.001" 和 "1e-4" 两种写法。"""
    if v <= 0:
        return ""
    if v >= 1:
        return f"{v:g}"
    return f"{v:.{max(0, int(-np.floor(np.log10(v))))}f}"


def plot_scatter(df, n_pat, horizon, min_events=10):
    """两栏：经典校准散点 + 比值图。

    比值图（右）不是装饰，它回答经典散点回答不了的那个问题：低估是**一个公共常数因子**，还是
    **特定 token 特有**的？前者保序、只是定标问题、原则上一个重标定就能修（README 3.2）；后者是
    模型对那个 token 没学会（README 3.1）。在 y=实际/预测 的图上，前者是一条水平带，后者是
    离群点。经典散点把这两种情形画成同一个样子。
    """
    from matplotlib.ticker import FuncFormatter
    d = df[df.n_events >= min_events].copy()
    d["under"] = d.observed / d.predicted.replace(0, np.nan)
    ps.setup_style(); _use_cjk_font()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.2, 6.9))

    # ---------------- 左：校准散点，等比例、恒等线 45 度
    # 轴范围取自数据本身，不用固定的 1e-4 下限——第一版那样做在左下角留了两个空白数量级。
    allv = np.concatenate([d.observed.values, d.predicted[d.predicted > 0].values])
    lo = 10 ** (np.floor(np.log10(allv.min()) * 2) / 2)
    hi = 10 ** (np.ceil(np.log10(allv.max()) * 2) / 2)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.6, color="#333333", zorder=2,
            label="完美校准 (y = x)")
    for f, ls, lab in [(0.5, ":", "低估 2 倍"), (0.1, "-.", "低估 10 倍")]:
        ax.plot([lo, hi], [lo * f, hi * f], ls=ls, lw=1.0, color="#9a9a9a", zorder=1)
        # 标签放在线上、轴内。第一版用 rotation_mode="anchor" 贴在线的左端，结果被挤到
        # 坐标轴外面去了（渲染出来才看见）——这里锚在线的中段，并显式 clip_on。
        xa = 10 ** (np.log10(lo) + 0.70 * (np.log10(hi) - np.log10(lo)))
        ax.text(xa, xa * f, lab, fontsize=8.5, color="#8a8a8a",
                ha="center", va="center", rotation=45, rotation_mode="anchor",
                clip_on=True,
                bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.9))
    for fam in FAMILY_ORDER:
        sub = d[d.family == fam]
        if not len(sub):
            continue
        ax.scatter(sub.observed, np.maximum(sub.predicted, lo), s=70,
                   color=FAMILY_COLORS[fam], edgecolor="white", linewidth=1.8,
                   zorder=4, label=fam)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    for a_ in (ax,):
        a_.xaxis.set_major_formatter(FuncFormatter(_logfmt))
        a_.yaxis.set_major_formatter(FuncFormatter(_logfmt))
    ax.set_xlabel(f"实际发生率（基线后 {horizon:g} 年内）")
    ax.set_ylabel("模型预测发生率（同一窗口）")
    ax.set_title("每个 token 的校准\n点落在虚线下方 = 低估", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left", frameon=True, framealpha=0.94)
    ax.grid(alpha=0.22)

    # ---------------- 右：低估倍数 vs 实际发生率
    med = float(np.nanmedian(d.under))
    ax2.axhline(1, ls="--", lw=1.6, color="#333333", zorder=2, label="完美校准")
    ax2.axhline(med, ls="-", lw=1.4, color="#8a8a8a", zorder=2,
                label=f"中位低估 {med:.1f} 倍")
    for fam in FAMILY_ORDER:
        sub = d[d.family == fam]
        if not len(sub):
            continue
        ax2.scatter(sub.observed, sub.under, s=70, color=FAMILY_COLORS[fam],
                    edgecolor="white", linewidth=1.8, zorder=4, label=fam)
    # 直接标注离群点（也兑付 #56B4E9 的 contrast WARN 义务）+ Death 作为参照
    tag = pd.concat([d.nlargest(5, "under"), d[d.family == "Death"]]) \
        .drop_duplicates("token").reset_index(drop=True)
    for i, r in tag.iterrows():
        ax2.annotate(_ascii(r["display"]), (r["observed"], r["under"]),
                     textcoords="offset points",
                     xytext=(-11 if i % 2 else 11, 7 if i % 2 else -11),
                     ha="right" if i % 2 else "left", fontsize=8.5, color="#1a1a1a",
                     arrowprops=dict(arrowstyle="-", lw=0.7, color="#aaaaaa",
                                     shrinkA=0, shrinkB=3))
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.xaxis.set_major_formatter(FuncFormatter(_logfmt))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}x"))
    ax2.set_xlabel(f"实际发生率（基线后 {horizon:g} 年内）")
    ax2.set_ylabel("低估倍数（实际 / 预测）")
    ax2.set_title("低估是常数因子还是 token 特有\n水平带 = 常数（可重标定）；离群点 = 没学会",
                  fontsize=10.5)
    ax2.legend(fontsize=8.5, loc="upper right", frameon=True, framealpha=0.94)
    ax2.grid(alpha=0.22)

    fig.suptitle(f"预测 vs 实际 · 无 at-risk 条件化 · {len(d)} 个 token · 全部 {n_pat} 名患者 · "
                 f"{int((d.predicted < d.observed).sum())}/{len(d)} 个被低估",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save_fig(fig, OUT, f"calibration_h{horizon:g}")


def plot_frames(df, n_pat, min_events=10):
    """A3 的检验图：低估倍数随窗口怎么变，以及换成"组成"口径会怎样。

    左panel 回答"缩短窗口有没有用"。它不会有用，而且**方向和直觉相反**，原因是饱和：
    P(窗口内出现) = 1 - exp(-R·h)，两边的 R 差一个常数因子 f。h 小时两边都近似线性，
    比值 -> f（真实的速率比）；h 大时实测那边先趋近 1，比值被压回来。所以**长窗口让低估
    看起来更轻，短窗口让它露出真实大小**。想靠改 horizon 消掉它是不可能的。

    右panel 是真正能消掉它的口径：不问"窗口内出没出现"，而问"**在被发射的那些 token 里**，
    这个 token 占多少份额"。这一步把速率因子约掉了，剩下的只有组成。
    """
    from matplotlib.ticker import FuncFormatter
    ps.setup_style(); _use_cjk_font()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.2, 6.4))
    labs = [f"{h}y" for h in SWEEP_H]

    # ---- 左：低估倍数 vs 窗口
    xs, ys, rate = [], [], []
    for h, lab in zip(SWEEP_H, labs):
        d = df[(df.window == lab) & (df.n_events >= min_events)]
        if not len(d):
            continue
        xs.append(h)
        ys.append(float(np.nanmedian(d.observed / d.predicted.replace(0, np.nan))))
        rate.append(float(d.n_clin_pred.iloc[0] / d.n_clin_obs.iloc[0]))
    ax.plot(xs, ys, "-o", lw=2, ms=9, color="#7B3294", label="每 token 低估倍数（中位）")
    ax.plot(xs, [1 / r for r in rate], "--s", lw=1.8, ms=8, color="#56B4E9",
            label="临床 token 总数的缺口（实际/预测）")
    dn = df[(df.window == "nextvisit") & (df.n_events >= min_events)]
    if len(dn):
        un = float(np.nanmedian(dn.observed / dn.predicted.replace(0, np.nan)))
        ax.axhline(un, ls=":", lw=1.6, color="#D55E00")
        ax.text(max(xs) * 0.98, un * 1.04, f"下一次访视窗口: {un:.1f}x",
                fontsize=9, color="#D55E00", ha="right", va="bottom")
    ax.axhline(1, ls="--", lw=1.4, color="#333333")
    ax.text(min(xs), 1.04, "完美校准", fontsize=9, color="#333333", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(xs); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}x"))
    ax.set_xlabel("报告窗口（年）")
    ax.set_ylabel("低估倍数（实际 / 预测）")
    ax.set_title("A3 检验：缩短报告窗口有没有用\n没有——而且窗口越短，低估暴露得越大（饱和效应）",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower left", frameon=True, framealpha=0.94)
    ax.grid(alpha=0.22, which="both")

    # ---- 右：组成口径
    d = df[(df.window == "5y") & (df.n_events >= min_events)].copy()
    lo = min(d.share_obs[d.share_obs > 0].min(), d.share_pred[d.share_pred > 0].min()) * 0.6
    hi = max(d.share_obs.max(), d.share_pred.max()) * 1.6
    ax2.plot([lo, hi], [lo, hi], ls="--", lw=1.6, color="#333333", label="完美校准 (y = x)")
    for fam in FAMILY_ORDER:
        sub = d[d.family == fam]
        if not len(sub):
            continue
        ax2.scatter(sub.share_obs, np.maximum(sub.share_pred, lo), s=70,
                    color=FAMILY_COLORS[fam], edgecolor="white", linewidth=1.8,
                    zorder=4, label=fam)
    g = d.assign(gap=(np.log10(d.share_obs.clip(lower=lo))
                      - np.log10(d.share_pred.clip(lower=lo))).abs())
    for i, r in g.nlargest(5, "gap").reset_index(drop=True).iterrows():
        ax2.annotate(_ascii(r["display"]), (r["share_obs"], max(r["share_pred"], lo)),
                     textcoords="offset points",
                     xytext=(-11 if i % 2 else 11, 7 if i % 2 else -11),
                     ha="right" if i % 2 else "left", fontsize=8.5, color="#1a1a1a",
                     arrowprops=dict(arrowstyle="-", lw=0.7, color="#aaaaaa",
                                     shrinkA=0, shrinkB=3))
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi); ax2.set_aspect("equal")
    for a_ in (ax2,):
        a_.xaxis.set_major_formatter(FuncFormatter(_logfmt))
        a_.yaxis.set_major_formatter(FuncFormatter(_logfmt))
    med = float(np.nanmedian(d.share_obs / d.share_pred.replace(0, np.nan)))
    ax2.set_xlabel("实际份额（该 token 占被发射临床 token 的比例）")
    ax2.set_ylabel("预测份额（同一定义）")
    ax2.set_title(f"能消掉低估的口径：问「组成」而不是「是否出现」\n"
                  f"速率因子被约掉，中位偏差 {med:.2f}x", fontsize=10.5)
    ax2.legend(fontsize=8.5, loc="upper left", frameon=True, framealpha=0.94)
    ax2.grid(alpha=0.22)

    fig.suptitle(f"报告口径对照 · split=val · {n_pat} 名患者 · 全部 token、无 at-risk 筛选",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return ps.save_fig(fig, OUT, "reporting_frames")


def plot_bars(df, n_pat, horizon, min_events=10):
    d = df[df.n_events >= min_events].copy()
    ps.setup_style(); _use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.2, max(6.0, 0.235 * len(d) + 1.9)))
    y = np.arange(len(d))
    h = 0.36
    ax.barh(y + h / 2 + 0.02, d.observed, height=h, color="#3a3a3a", label="实际")
    ax.barh(y - h / 2 - 0.02, d.predicted, height=h,
            color=[FAMILY_COLORS[f] for f in d.family], label="模型预测（按家族着色）")
    for yy, r in zip(y, d.itertuples()):
        lab = "—" if (not np.isfinite(r.ratio) or r.ratio <= 0) else f"{1 / r.ratio:.0f}x"
        ax.text(max(r.observed, r.predicted) + 0.008, yy, lab, va="center",
                fontsize=7.8, color="#555555")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{_ascii(r.display)}  (n={r.n_events})" for r in d.itertuples()],
                       fontsize=7.8)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlabel(f"P(该 token 在基线后 {horizon:g} 年内出现)")
    ax.set_xlim(0, min(1.0, max(d.observed.max(), d.predicted.max()) * 1.20))
    ax.set_title(f"预测 vs 实际，按实际发生率排序 · 全部 {n_pat} 名患者、无 at-risk 筛选\n"
                 f"右侧数字 = 低估倍数（实际 / 预测）", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right", frameon=True, framealpha=0.94)
    ax.grid(alpha=0.22, axis="x")
    ax.invert_yaxis()
    return ps.save_fig(fig, OUT, f"predict_vs_observe_bars_h{horizon:g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mc", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sweep", action="store_true",
                    help="A3 的检验：一次 MC pass 出多个 horizon + 下一次访视 + 组成口径")
    ap.add_argument("--replot", action="store_true",
                    help="只重画，读上次的 CSV，跳过 5 分钟的 MC pass")
    a = ap.parse_args()
    if a.sweep:
        df, n = sweep(n_mc=a.n_mc, limit=a.limit, workers=a.workers,
                      device=a.device, min_events=a.min_events)
        _, png = plot_frames(df, n, a.min_events)
        print(f"-> {png}")
        return
    if a.replot:
        path = os.path.join(OUT, f"predict_vs_observe_h{a.horizon:g}_data.csv")
        df = pd.read_csv(path)
        n = int(df.n_known.max())
        print(f"读回 {path}（{len(df)} 个 token，n={n}）")
    else:
        df, n = build(n_mc=a.n_mc, limit=a.limit, horizon=a.horizon,
                      workers=a.workers, device=a.device)
    _, png1 = plot_scatter(df, n, a.horizon, a.min_events)
    _, png2 = plot_bars(df, n, a.horizon, a.min_events)
    d = df[df.n_events >= a.min_events]
    print(f"\n{len(d)} 个 token（实际事件 ≥ {a.min_events}）")
    print(f"  低估的: {(d.predicted < d.observed).sum()}   高估的: {(d.predicted > d.observed).sum()}")
    v = d.dropna(subset=["auc"])
    print(f"  AUC: 中位 {v.auc.median():.3f}   >0.7 的 {int((v.auc>0.7).sum())}/{len(v)}   "
          f"<0.55 的 {int((v.auc<0.55).sum())}/{len(v)}")
    print(f"  低估倍数中位: {np.nanmedian(d.observed / d.predicted.replace(0, np.nan)):.1f}×")
    print(f"-> {png1}\n-> {png2}")


if __name__ == "__main__":
    main()
