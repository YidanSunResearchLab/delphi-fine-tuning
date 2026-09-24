"""
next_token_auc.py -- 最朴素的口径：**预测下一个 token**，就是训练目标本身。

    python next_token_auc.py --runs dedup:../delphi/Delphi-ROSMAP-gpubase/ckpt.pt:rosmap \
                                    nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup \
                                    fullvisit:../delphi/Delphi-ROSMAP-fullvisit/ckpt.pt:rosmap_fullvisit

每个病人一次 teacher-forced forward（三份分词的 max 长度都短于各自 block_size，所以整条
轨迹一次装得下，不用滑窗），取**每个位置**的 logits，目标是流里的下一个 token。
没有 rollout、没有 Monte-Carlo、没有 at-risk 筛选。

--------------------------------------------------------------------------------------
读这些数字之前必须知道的一件事：**三档的分母不一样，而且没法弄一样**

"下一个 token 是什么"这个问题本身就是**由分词定义**的。val split 的位置数是
15,575 / 20,892 / 36,080 —— 不去重的那几档要回答的问题更多、也更容易（一大半是"和上次
一样"）。这跟 next_visit_auc.py 不同：那里可以把真值统一成同一份逐访视记录，这里不能，
因为问题的**定义域**变了。

所以下面的表是"每个模型在自己的任务上考了多少分"，**不是**同一张卷子的分数对比。
要同一张卷子，看 next_visit_auc.py（它把三档的观测集合强制统一成 3,384 个访视对）。
这里额外给两个量帮助判断差异有多少来自"卷子变简单"：
  * `多数类基线` —— 每个位置都猜全语料最常见的那个 token 能拿到的 top-1；
  * 只在**一次性事件**（AD_DX / Death / STROKE / *_ONSET）上的 AUC。那些 token 在三份
    数据里出现次数完全相同，虽然负样本池不同，但至少正样本是同一批人同一批事件。

--------------------------------------------------------------------------------------
口径细节（每一条都照训练时来，不是随手选的）

  * `mask_ties=True`，`targets_age` = 流里下一个 token 的真实年龄。model.py 只在传 targets
    时施加这个掩码，而本分词一次访视的 token 同龄 —— 训练时每个位置都看不见自己访视的兄弟。
    不传就等于在一个从未训练过的注意力配置下打分（README 3.1 量到值 7.3x）。
  * **排除目标是 padding / 性别 / 背景块的位置**，即 model.py 里那个 `pass_tokens`。
    训练损失就是这么排的：那些 token 在 `ignore_tokens` 里，模型从不被要求预测它们。
  * **不注入 No-event 标记**，只用真实流。所以 No-event 永远不是正确答案，它那一列和
    statics 一起被屏蔽，softmax 在剩下的 content 列上重整（等价于 model.py 的
    `validation_loss_mode=True`）。
  * CI 是**按人** bootstrap：一个人贡献几十个位置，位置之间高度相关，按位置 bootstrap
    会把区间算窄好几倍。重采样的人集在各 token 之间**复用**（省时间；代价是各 token 的
    CI 相关，而我们不在 token 之间做比较，所以没影响）。
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                        # noqa: E402
from radc_delphi import engine as EN, vocab as V    # noqa: E402

MIN_POS = 30          # 与 figure2_panels 的上游规则一致：正样本 <30 的 token 不报
N_BOOT = 400


def fast_auc(y, s):
    """rank 法 AUC。比 sklearn 快，且这里要在 bootstrap 里调几万次。"""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s), dtype=np.float64)
    r[o] = np.arange(1, len(s) + 1)
    # 处理并列：同分取平均 rank
    su = s[o]
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def score_run(ckpt, dataset, device="cpu"):
    """返回 (P, tgt, pat_slices, res)：P=(n_pos, n_content) 概率, tgt=目标 token(model space)。"""
    data_dir = os.path.join(HERE, "data", dataset)
    eng = EN.load(os.path.join(HERE, ckpt), data_dir=data_dir, device=device)
    d = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint32).reshape(-1, 3)
    R = eng.res
    content = np.array(eng.content_ids, dtype=np.int64)          # 已排除 padding + statics
    keep_col = np.array([t for t in content if t != R.NO_EVENT], dtype=np.int64)
    ct = torch.as_tensor(keep_col, dtype=torch.long)
    drop_tgt = set(eng.ignore_tokens) | {R.PADDING, R.NO_EVENT}

    Ps, Ts, slices, pids = [], [], [], []
    n = 0
    for pid in np.unique(d[:, 0]):
        rows = d[d[:, 0] == pid]
        ages = rows[:, 1].astype(np.float64)
        toks = rows[:, 2].astype(np.int64) + 1                   # disk -> model space
        o = np.argsort(ages, kind="stable")
        ages, toks = ages[o], toks[o]
        if len(toks) < 2:
            continue
        assert len(toks) <= eng.block_size, (
            f"{dataset}: 病人 {pid} 有 {len(toks)} 个 token，超过 block_size "
            f"{eng.block_size}；这个脚本假设整条轨迹一次装得下")
        idx = torch.as_tensor(toks, dtype=torch.long)[None, :]
        age = torch.as_tensor(ages, dtype=torch.float32)[None, :]
        tgt_age = torch.cat([age[:, 1:], torch.full((1, 1), 1e9)], 1)
        with torch.no_grad():
            lg, _, _ = eng.model(idx.to(device), age.to(device),
                                 targets=idx.to(device), targets_age=tgt_age.to(device))
        lg = lg[0, :-1, :].float().cpu()                         # 最后一个位置没有"下一个"
        nxt = toks[1:]
        m = ~np.isin(nxt, list(drop_tgt))                        # = model.py 的 pass_tokens
        if not m.any():
            continue
        p = torch.softmax(lg[torch.as_tensor(m), :][:, ct], -1).numpy().astype(np.float32)
        Ps.append(p)
        Ts.append(nxt[m])
        slices.append((n, n + int(m.sum())))
        pids.append(int(pid))
        n += int(m.sum())
    return np.vstack(Ps), np.concatenate(Ts), slices, keep_col, R, eng


def chapter_of(name, R):
    if name == "Death":
        return "Death"
    if name.endswith(("_ON", "_OFF")):
        return "Medications"
    if name.endswith("_ONSET") or name in ("STROKE", "DEPRESSION", "AD_DX"):
        return "Events"
    if name.startswith(("BRAAK_", "CERAD_", "GPATH_", "AMYL_", "TANG_", "TDP_", "LEWY_",
                        "ARTSCL_", "CAA_", "CVDA_", "MICROINF_", "INFARCT_")):
        return "Pathology"
    return "Clinical measures"


ONCE_ONLY = None    # 由 R 决定，见 main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name:ckpt:dataset")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    results, meta = {}, {}
    for spec in a.runs:
        nm, ckpt, ds = spec.split(":")
        P, T, slices, cols, R, eng = score_run(ckpt, ds, a.device)
        # 预先生成 bootstrap 的人集索引，各 token 复用
        boots = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(slices), len(slices))
            boots.append(np.concatenate([np.arange(*slices[i]) for i in pick]))
        col_of = {int(c): j for j, c in enumerate(cols)}
        per = {}
        cnt = collections.Counter(T.tolist())
        for t, npos in cnt.items():
            if npos < MIN_POS or t not in col_of:
                continue
            y = (T == t).astype(np.int8)
            s = P[:, col_of[t]]
            auc = fast_auc(y, s)
            bs = [fast_auc(y[b], s[b]) for b in boots]
            bs = [x for x in bs if np.isfinite(x)]
            per[R.NAMES[t]] = dict(auc=auc, ci=[float(np.percentile(bs, 2.5)),
                                                float(np.percentile(bs, 97.5))],
                                   n_pos=int(npos),
                                   chapter=chapter_of(R.NAMES[t], R))
        top1 = float((cols[P.argmax(1)] == T).mean())
        # 多数类基线：永远猜全局最常见的那个 token
        maj = cnt.most_common(1)[0]
        results[nm] = per
        meta[nm] = dict(dataset=ds, n_pos_total=int(len(T)), n_patients=len(slices),
                        top1=top1, majority_token=R.NAMES[maj[0]],
                        majority_top1=maj[1] / len(T),
                        n_tokens_reported=len(per),
                        median_auc=float(np.median([v["auc"] for v in per.values()])),
                        loss_note="mask_ties=True, no injected No-event")
        print(f"  {nm:<10s} dataset={ds:<18s} {len(slices)} 人 / {len(T):,} 个位置 / "
              f"{len(per)} 个 token 达到 n>={MIN_POS}")

    names = [s.split(":")[0] for s in a.runs]
    print(f"\n{'='*104}")
    print("预测下一个 token 的 AUC（teacher-forced，无 rollout；每个模型在**自己的**流上；"
          f"CI = 按人 bootstrap {N_BOOT} 次）")
    print(f"{'='*104}")
    print(f"{'':<14s}" + "".join(f"{n:>22s}" for n in names))
    for k, lab in (("n_patients", "病人数"), ("n_pos_total", "评分位置数"),
                   ("n_tokens_reported", f"达到 n>={MIN_POS} 的 token"),
                   ("median_auc", "中位 AUC"), ("top1", "top-1 准确率"),
                   ("majority_top1", "多数类基线 top-1")):
        line = f"{lab:<14s}"
        for n in names:
            v = meta[n][k]
            line += f"{v:>22.3f}" if isinstance(v, float) else f"{v:>22,}"
        print(line)
    print(f"{'多数类 token':<14s}" + "".join(f"{meta[n]['majority_token']:>22s}" for n in names))

    allnames = []
    for n in names:
        allnames += [k for k in results[n] if k not in allnames]
    for chap in ("Events", "Death", "Clinical measures", "Medications", "Pathology"):
        sub = [k for k in allnames
               if any(results[n].get(k, {}).get("chapter") == chap for n in names)]
        if not sub:
            continue
        print(f"\n--- {chap} " + "-" * (96 - len(chap)))
        hdr = f"{'token':<20s}" + "".join(f"{'AUC ' + n:>22s}" for n in names) + f"{'n_pos':>18s}"
        print(hdr)
        for k in sorted(sub, key=lambda x: -max(results[n].get(x, {}).get("auc", 0)
                                                for n in names)):
            line = f"{k:<20s}"
            nps = []
            for n in names:
                v = results[n].get(k)
                line += (f"{v['auc']:>10.3f} [{v['ci'][0]:.2f},{v['ci'][1]:.2f}]" if v
                         else f"{'—':>22s}")
                nps.append(str(v["n_pos"]) if v else "—")
            print(line + f"{'/'.join(nps):>18s}")
        # 该章节的中位
        line = f"{'  -> 中位':<20s}"
        for n in names:
            vs = [results[n][k]["auc"] for k in sub if k in results[n]]
            line += f"{np.median(vs):>22.3f}" if vs else f"{'—':>22s}"
        print(line)

    print("\n注意：三档的**评分位置数不同**（见上表）—— "
          "「下一个 token 是什么」这个问题本身就是由分词定义的。")
    print("      同一张卷子的对比见 next_visit_auc.py；那里三档共用 3,384 个访视对。")
    print("      Events / Death 两章的正样本在三份数据里是同一批事件（一次性 token），"
          "只有负样本池不同。")

    if a.out:
        json.dump(dict(meta=meta, per_token=results), open(a.out, "w"),
                  ensure_ascii=False, indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
