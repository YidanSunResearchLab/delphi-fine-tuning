"""
test_snapshot_first_occurrence.py -- `--snapshot` 不能改变任何 token 的**首次出现**。

figure 3 的 AUC 用首次出现口径：病例 = 每人第一次出现 token k 的时刻。所以"完全不去重"
（--snapshot）和其它分词之间比 AUC，前提是每个 (人, token) 的首次出现时刻**逐位相同**，
否则比的就不是同一道题。唯一允许的例外是 *_OFF：snapshot 下它的语义从"停药"变成"本次未用药"，
从没用过药的人也会有 OFF，而评估本来就排除 OFF。

    python build.py --out /tmp/pve  --nodedup all --no-global-dedup --per-visit-events
    python build.py --out /tmp/snap --snapshot
    python test_snapshot_first_occurrence.py /tmp/pve /tmp/snap

还顺带钉住三件事：人和划分不变；snapshot 确实是"每次访视完整状态"（背景块每次访视都在）；
snapshot 只加 token、不删 token（pve 的每一行在 snapshot 里都有）。
"""
import sys
import numpy as np
import pandas as pd

PVE, SNAP = sys.argv[1], sys.argv[2]
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


names = pd.read_csv(f"{SNAP}/labels.csv").event_name.tolist()
assert names == pd.read_csv(f"{PVE}/labels.csv").event_name.tolist(), "词表不同"
off_ids = {i - 1 for i, n in enumerate(names) if n.endswith("_OFF")}      # pre-shift id


def load(d, split):
    return pd.DataFrame(np.fromfile(f"{d}/{split}.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64),
                        columns=["p", "a", "t"])


for split in ("train", "val"):
    a, b = load(PVE, split), load(SNAP, split)
    check(f"{split}: 同一批人", set(a.p) == set(b.p), f"({a.p.nunique()} / {b.p.nunique()})")
    fa = a[~a.t.isin(off_ids)].groupby(["p", "t"]).a.min()
    fb = b[~b.t.isin(off_ids)].groupby(["p", "t"]).a.min()
    same_keys = fa.index.equals(fb.index) or set(fa.index) == set(fb.index)
    extra = sorted({names[t + 1] for _, t in set(fb.index) - set(fa.index)})
    missing = sorted({names[t + 1] for _, t in set(fa.index) - set(fb.index)})
    check(f"{split}: (人, token) 集合相同（*_OFF 除外）", same_keys,
          f"(snapshot 多出 {len(set(fb.index) - set(fa.index))} 个: {extra[:6]}；少了 {len(set(fa.index) - set(fb.index))} 个: {missing[:6]})")
    common = fa.index.intersection(fb.index)
    diff = (fa.loc[common] != fb.loc[common])
    bad = sorted({names[t + 1] for _, t in diff[diff].index})
    check(f"{split}: 首次出现时刻逐位相同", not diff.any(), f"({int(diff.sum())} 处不同: {bad[:6]})")
    # 只加不删：pve 的每一行 (p, a, t) 都在 snapshot 里
    ka = set(map(tuple, a[~a.t.isin(off_ids)].values)); kb = set(map(tuple, b.values))
    check(f"{split}: snapshot 只加 token、不删", ka <= kb, f"({len(ka - kb)} 行在 snapshot 里消失)")

# 完整状态：snapshot 里每一次访视（= 有 token 的年龄）都带着性别 token。两类时刻不是访视，豁免：
#   * 死亡时刻：只有 Death 和死后病理；
#   * 只有 AD_DX 的时刻：首诊时间来自单独的 age_first_ad_dx，有时不落在任何访视上
#     （各分词都这样）。为了首次出现不变，这个时刻不能挪，也就不能硬塞一整块背景。
b = load(SNAP, "val")
sex_ids = {names.index("Female") - 1, names.index("Male") - 1}
death_id = names.index("Death") - 1
ad_id = names.index("AD_DX") - 1
_only_ad = b.groupby(["p", "a"]).t.apply(lambda t: set(t) == {ad_id})
_only_ad = _only_ad[_only_ad].reset_index().groupby("p").a.apply(set)
death_ages = b[b.t == death_id].groupby("p").a.apply(set)
death_ages = {p: death_ages.get(p, set()) | _only_ad.get(p, set()) for p in b.p.unique()}
sex_ages = b[b.t.isin(sex_ids)].groupby("p").a.apply(set)
all_ages = b.groupby("p").a.apply(set)
miss = sum(len(all_ages[p] - sex_ages.get(p, set()) - death_ages.get(p, set())) for p in all_ages.index)
nvis = sum(len(all_ages[p] - death_ages.get(p, set())) for p in all_ages.index)
check("snapshot：每次访视都带着背景块（死亡时刻、孤立的 AD 首诊时刻除外）", miss == 0,
      f"({nvis} 次访视, {miss} 次缺；孤立 AD 首诊时刻 {sum(len(v) for v in _only_ad.values)} 个)")

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(fails > 0)
