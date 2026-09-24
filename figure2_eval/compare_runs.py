"""
compare_runs.py -- 把两次 figure2 run 的 panel a / panel b 指标并排列出来。

    python compare_runs.py results/figure2/{gpubase,nodedup,fullvisit} \
        --names dedup cog-nodedup all-nodedup --cohort matched

为什么需要它：`FIG2_TAG` 让多次 run 的输出各自成目录，但 metrics_*.json 是嵌套的，肉眼对读
八行 AUC × 两组 CI 很容易串行。这个脚本只做**对齐和排版**，不重算任何统计量。

两个必须一起读的限制：
  * **行集合可能不同。** panel a 的 at-risk 规则是"基线分期 != 该行"，而 `_baseline()` 要求
    至少两次不同访视 + 基线有认知分期。不去重那份分词让**每次**随访都发射认知 token，所以
    可评估人数和每行的 at-risk/events 都会变。脚本把 n 一起打出来，差异大的行标 `!`。
  * **单个 seed。** 两边都只有一次训练，AUC 的 CI 宽到 ±0.06~0.11，所以逐行差异小于 CI 宽度时
    不要当成效应。
"""
import os
import sys
import json
import argparse


def load(d, cohort):
    p = os.path.join(d, f"metrics_{cohort}.json")
    if not os.path.exists(p):
        sys.exit(f"没有 {p}")
    with open(p) as fh:
        return json.load(fh)


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="各 run 的输出目录（results/figure2/<tag>）")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--cohort", default="matched", choices=["matched", "allcohort"])
    a = ap.parse_args()
    names = a.names or [os.path.basename(d.rstrip("/")) or d for d in a.dirs]
    if len(names) != len(a.dirs):
        sys.exit(f"--names 给了 {len(names)} 个，目录有 {len(a.dirs)} 个")
    Ms = [load(d, a.cohort) for d in a.dirs]
    runs = list(zip(names, Ms))

    print(f"# figure2 对照  cohort={a.cohort}")
    for n, d in zip(names, a.dirs):
        print(f"  {n:<14s} {d}")
    print()

    # ---------------------------------------------------------------- panel a
    As = [m["a"] for m in Ms]
    print("## panel a — 5 年内到达各状态")
    print("  可评估人数（panel c 的 n_patients 代理）: "
          + " | ".join(f"{n} {m['c']['n_patients']}" for n, m in runs))
    print("  median AUC        " + "  ".join(f"{n} {fmt(A['median_auc'])}"
                                             for n, A in zip(names, As)))
    print("  mean calib error  " + "  ".join(f"{n} {fmt(A['mean_calibration_error'],4)}"
                                             for n, A in zip(names, As)))
    print("  转移时间 R²/MAE/bias " + "  ".join(
        f"{n} {fmt(A.get('transition_time',{}).get('r2'))}/"
        f"{fmt(A.get('transition_time',{}).get('mae'),2)}/"
        f"{fmt(A.get('transition_time',{}).get('bias'),2)}" for n, A in zip(names, As)))
    print()

    rows = []
    for A in As:
        rows += [k for k in A["per_state"] if k not in rows]
    hdr = f"{'状态':<26s}" + "".join(f"{'AUC ' + n:>20s}" for n in names) + \
          f"{'events':>8s}  {'at-risk':>16s}"
    print(hdr)
    print("-" * len(hdr))
    # 第一个 run 是基准列，Δ 都相对它算
    for k in rows:
        ps = [A["per_state"].get(k) for A in As]
        line = f"{k:<26s}"
        for j, p in enumerate(ps):
            if p is None:
                line += f"{'—':>20s}"
                continue
            if j == 0:
                line += f"{p['auc']:>10.3f}          "
            else:
                base = ps[0]
                d = p["auc"] - base["auc"] if base else float("nan")
                # CI 宽度的一半作为"噪声尺度"；差异没超过它就不值得解读
                half = max(base["ci"][1] - base["ci"][0], p["ci"][1] - p["ci"][0]) / 2 if base else 0
                line += f"{p['auc']:>10.3f}({d:+.3f}){'*' if abs(d) > half else ' '}"
        ev = next((p["n_events"] for p in ps if p), 0)
        ar = "/".join(str(p["n_at_risk"]) if p else "—" for p in ps)
        print(line + f"{ev:>8d}  {ar:>16s}")
    print("  * = |ΔAUC| 相对第一列超过两者 CI 半宽的较大者（单 seed，小于这个别解读）")
    print()
    hdr = f"{'状态':<26s}{'实测':>8s}" + "".join(f"{'预测 ' + n:>16s}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    for k in rows:
        ps = [A["per_state"].get(k) for A in As]
        obs = next((p["observed_cif"] for p in ps if p), float("nan"))
        line = f"{k:<26s}{obs:>8.3f}"
        for p in ps:
            if p is None:
                line += f"{'—':>16s}"
            else:
                r = p["observed_cif"] / p["predicted"] if p["predicted"] > 0 else float("inf")
                line += f"{p['predicted']:>9.3f}({r:>4.1f}x)"
        print(line)
    print("  括号里是 实测/预测 的倍数，1.0x = 定标准确\n")

    # ---------------------------------------------------------------- panel b
    print("## panel b — 按 horizon 的 AUC")
    Bs = [m["b"]["auc_by_horizon"] for m in Ms]
    hz = sorted({h for B in Bs for v in B.values() for h in v}, key=float)
    eps = []
    for B in Bs:
        eps += [k for k in B if k not in eps]
    print(f"{'终点':<32s}{'run':<14s}" + "".join(f"{h + 'y':>8s}" for h in hz))
    for k in eps:
        for nm, B in zip(names, Bs):
            v = B.get(k, {})
            print(f"{k:<32s}{nm:<14s}"
                  + "".join(f"{v.get(h, float('nan')):>8.3f}" for h in hz))
        print()

    print("## panel b — timing error (MAE 年 / bias 年，按真实首达时间分箱)")
    Ts = [m["b"]["timing_error"] for m in Ms]
    eps = []
    for T in Ts:
        eps += [k for k in T if k not in eps]
    for k in eps:
        print(f"  {k}")
        bins = []
        for T in Ts:
            bins += [b for b in T.get(k, {}) if b not in bins]
        print(f"    {'bin':<8s}{'n':>5s}" + "".join(f"{'MAE ' + n:>14s}" for n in names)
              + "".join(f"{'bias ' + n:>14s}" for n in names) + f"{'naive MAE':>11s}")
        for b in bins:
            cells = [T.get(k, {}).get(b) for T in Ts]
            n = next((c["n"] for c in cells if c), 0)
            nv = next((c["naive_mae"] for c in cells if c), float("nan"))
            print(f"    {b:<8s}{n:>5d}"
                  + "".join(f"{(c['mae'] if c else float('nan')):>14.2f}" for c in cells)
                  + "".join(f"{(c['bias'] if c else float('nan')):>14.2f}" for c in cells)
                  + f"{nv:>11.2f}")
        print()


if __name__ == "__main__":
    main()
