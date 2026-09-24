"""
make_aux_labels.py -- 给多任务判别头（model.py 的 `aux_head`）生成**逐访视**的删失感知标签。

    python make_aux_labels.py --dataset rosmap_nodedup
    -> data/rosmap_nodedup/aux_labels_train.npy   (n_rows, 15) int8
       data/rosmap_nodedup/aux_labels_val.npy     (n_rows, 15) int8
       data/rosmap_nodedup/aux_labels_meta.json   列名、口径、.bin 的 md5

写这个脚本的理由见 figure2_eval/README.md 第 9.3 / 9.5 节：同一个 ckpt，把 rollout 换成
"基线隐状态接 logistic"，panel a 中位 AUC 0.680 -> 0.777、AD 5y 0.535 -> 0.818。塌陷在
**输出机制**不在表征，所以给模型一个直接的判别读出口；而 9.5 明确写了端到端要做对的三件
事，这个文件负责其中两件：
  (1) 标签必须是 `figure2_core.labels_at_h` 的**删失感知**版本（1 / 0 / -1，-1 丢弃）。
      用朴素的"5 年内有没有"会把"随访还没到 5 年"的人算成阴性，系统性低估发生率。
  (2) 监督点接在**每一次访视**上，不是只接基线。

--------------------------------------------------------------------------------------
逐行对齐，以及为什么这件事必须在这里做对

输出是一个 (n_rows, n_targets) 的数组，**第 i 行对应 <split>.bin 的第 i 行**（.bin 是
np.uint32 的 (patient_id, age_days, disk_token) 三元组）。delphi/utils.py 的 get_batch 用
同一套 `batch_idx` 去取它，然后跟着 mask / no-event 注入 / 稳定排序 / 两次裁剪一路走。
对不齐不会有任何报错 —— BCE 照样收敛，只是每个位置学的是别人的终点。train.py 因此在加载
时硬断言行数相等，这里也把 .bin 的 md5 写进 meta.json，重新分词之后必须重新生成。

--------------------------------------------------------------------------------------
"从这个位置往后 h 年"，不是"从基线往后 h 年"

这就是"逐访视监督"的全部含义：第 i 行的标签问的是"从**第 i 行这个 token 的年龄** a_i 起，
往后 h 年，终点会不会发生"。一个人有 ~7 次访视就有 ~7 个不同的监督点（figure2 的口径只有
基线那 1 个）。同一次访视的所有 token 共用同一个 a_i，所以标签在一次访视内是常数 —— 这是
对的：它们描述的是同一个时刻。

事件一律取 **a > a_i（严格大于）**，和 figure2_core._process 的 `a > base` 同一条规则。
不能用 >=：mask_ties 下"一次访视的最后一个 token"能 attend 到本次访视的全部 token，把同龄
的事件算成未来就是直接泄漏，那一格 AUC 会漂亮得毫无意义。代价是访视中间的位置看不见本次
访视里那些它其实没看到的事件，宁可保守。

--------------------------------------------------------------------------------------
口径完全复用 figure2，不自己重写

  终点   `figure2/radc_states.endpoints()` -> 3 个：Reach ≥Borderline / AD diagnosis / Death
  horizon `figure2_core.HORIZONS` -> [1, 2, 3, 5, 10]
  标签    直接调 `figure2_core.labels_at_h(ep, h)`，ep 的三项 (obs_time / d_obs / fu) 按
          `lr_endpoint_baseline.ep_obs()` 的定义构造，只是起点从"基线"换成"这一行"。
列顺序：终点为主序、horizon 为次序 -> 列 e*len(HORIZONS)+j。meta.json 里有列名，别靠记。

**没有**套 at-risk 谓词（"Reach ≥Borderline" 只对当前还 Normal 的人有意义那条）。
这和 figure2 panel b / lr_endpoint_baseline 的 panel b 一致：那里也只用 labels_at_h，
`endpoints()` 返回的第三项谓词只有 panel a 用。后果要知道：对已经 Impaired 的人，
"Reach ≥Borderline" 在不去重的分词下几乎恒为 1（每次访视都发一个 MMSE token），这一列对
他们是"流行率"不是"发病率"。要改成发病率口径，是在这里按当前 MMSE 状态写 -1，而不是去改
labels_at_h。
"""
import os
import sys
import json
import hashlib
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                      # .../delphi
_F2 = os.path.join(os.path.dirname(HERE), "figure2_eval")              # .../figure2_eval
if _F2 not in sys.path:
    sys.path.insert(0, _F2)

from radc_delphi import vocab as V                 # noqa: E402
import figure2.radc_states as S                    # noqa: E402
import figure2.figure2_core as F2                  # noqa: E402

D = 365.25


def _first_after(sorted_event_days, query_days):
    """每个 query 之后（**严格**大于）第一个事件，返回年数；没有就是 inf。

    searchsorted(side='right') 把"同龄"判成过去，这正是上面那条防泄漏的规则。
    """
    out = np.full(len(query_days), np.inf)
    if len(sorted_event_days) == 0:
        return out
    j = np.searchsorted(sorted_event_days, query_days, side="right")
    hit = j < len(sorted_event_days)
    out[hit] = (sorted_event_days[j[hit]] - query_days[hit]) / D
    return out


def row_quantities(data, p2i, endpoints):
    """按 .bin 的**原始行序**给出 labels_at_h 需要的三样东西。

    返回 (obs_time[n_ep][n_rows], d_obs[n_rows], fu[n_rows])，单位年，起点是**该行自己的
    年龄**。注意不要在这里按年龄重排：行序就是对齐契约本身，排序只在每个病人内部临时用。
    """
    n_rows = data.shape[0]
    obs = [np.full(n_rows, np.inf) for _ in endpoints]
    d_obs = np.full(n_rows, np.inf)
    fu = np.zeros(n_rows, dtype=np.float64)

    for k in range(len(p2i)):
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        rows = data[s:s + n]
        a = rows[:, 1].astype(np.float64)
        # disk -> model（labels.csv 的行号）。get_batch 最后那句 `tokens + 1` 做的是同一件事；
        # 这里必须自己做，因为我们直接读 .bin。
        t = rows[:, 2].astype(np.int64) + 1
        slot = S.slot_of(t)
        pos = a > 0
        if not pos.any():
            continue
        last_obs = float(a[pos].max())
        fu[s:s + n] = (last_obs - a) / D

        dd = np.sort(a[slot == S.DEATH_IDX])
        d_obs[s:s + n] = _first_after(dd, a)
        for ei, (_lab, slots, _pred) in enumerate(endpoints):
            ev = np.sort(a[np.isin(slot, slots)])
            obs[ei][s:s + n] = _first_after(ev, a)
    return obs, d_obs, fu


def build_split(data_dir, split, endpoints, horizons):
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.fromfile(path, dtype=np.uint32).reshape(-1, 3)
    # get_p2i 从 delphi/utils.py 来（radc_delphi.batching 再导出一遍），训练和这里必须
    # 用同一条"病人从哪一行开始"的规则，两份实现就是漂移的开始。
    from radc_delphi.batching import get_p2i
    p2i = get_p2i(data)

    obs, d_obs, fu = row_quantities(data, p2i, endpoints)
    cols, names = [], []
    for ei, (lab, _slots, _pred) in enumerate(endpoints):
        ep = dict(obs_time=obs[ei], d_obs=d_obs, fu=fu)
        for h in horizons:
            # 一个字都不改地复用 figure2 的标签函数
            cols.append(F2.labels_at_h(ep, h).astype(np.int8))
            names.append(f"{lab} @{h}y")
    y = np.stack(cols, axis=1).astype(np.int8)
    return data, p2i, y, names


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rosmap_nodedup",
                    help="delphi/data/<dataset>/，需要 train.bin / val.bin / labels.csv")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--data-root", default=os.path.join(HERE, "data"))
    ap.add_argument("--force", action="store_true", help="已存在也覆盖")
    a = ap.parse_args()

    data_dir = os.path.join(a.data_root, a.dataset)
    labels_csv = os.path.join(data_dir, "labels.csv")
    # 绑定到**这份 build 自己的**词表。图省事用 live 表是错的：分词一改，MMSE / AD / Death
    # 的 id 就全挪了，而 slot_of 不会报错，只会给出一整列错误的终点。
    res = V.resolve_csv(labels_csv)
    S.configure(res.NAMES)
    endpoints = S.endpoints()
    horizons = list(F2.HORIZONS)
    print(f"dataset={a.dataset}  vocab={res.VOCAB_SIZE}")
    print(f"终点 {len(endpoints)} × horizon {len(horizons)} = {len(endpoints)*len(horizons)} 列")

    meta = {"dataset": a.dataset,
            "endpoints": [lab for lab, _s, _p in endpoints],
            "horizons": horizons,
            "column_order": "endpoint-major: col = e*len(horizons) + j",
            "label_semantics": {"1": "事件在 h 年内发生（且先于死亡）",
                                "0": "已知 h 年内无事件（随访够长 / 先死 / 事件在 h 之后）",
                                "-1": "行政删失，未知 —— 必须从损失里剔除"},
            "origin": "每一行以**该行自己的年龄**为起点，事件取严格大于该年龄的第一次",
            "at_risk_applied": False,
            "source": "figure2_core.labels_at_h + radc_states.endpoints()",
            "splits": {}}

    for split in a.splits:
        out = os.path.join(data_dir, f"aux_labels_{split}.npy")
        if os.path.exists(out) and not a.force:
            print(f"已存在，跳过（--force 覆盖）：{out}")
            continue
        data, p2i, y, names = build_split(data_dir, split, endpoints, horizons)
        np.save(out, y)
        meta["columns"] = names
        meta["splits"][split] = {"n_rows": int(y.shape[0]), "n_subjects": int(len(p2i)),
                                 "bin_md5": md5(os.path.join(data_dir, f"{split}.bin"))}
        print(f"\n{split}: {y.shape}  ({len(p2i)} 人) -> {out}")
        # 每一列的三值分布。这是最容易发现"口径写错了"的地方：某一列全是 -1、或者
        # "Reach ≥Borderline" 的阳性率接近 1，都说明起点/严格大于/词表绑定有问题。
        hdr = f"{'列':<34s}{'n=1':>8s}{'n=0':>8s}{'n=-1':>8s}{'阳性率':>9s}"
        print(hdr)
        print("-" * len(hdr))
        for j, nm in enumerate(names):
            c = y[:, j]
            n1, n0, nm1 = int((c == 1).sum()), int((c == 0).sum()), int((c == -1).sum())
            r = n1 / max(n1 + n0, 1)
            print(f"{nm:<34s}{n1:>8d}{n0:>8d}{nm1:>8d}{r:>9.3f}")

    with open(os.path.join(data_dir, "aux_labels_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"\n-> {os.path.join(data_dir, 'aux_labels_meta.json')}")
    print("训练侧：config 里 aux_head = True, aux_lambda = <权重>；aux_n_targets 由 train.py "
          "从这份 .npy 的列数推出，不要手填。")


if __name__ == "__main__":
    main()
