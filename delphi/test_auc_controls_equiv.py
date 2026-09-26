"""
test_auc_controls_equiv.py -- evaluate_auc_rosmap_controls.py 的外层循环必须**逐位**复现上游。

figure 3 的每个数都要能说"这是 Delphi 的 AUC"。包装脚本重写了 evaluate_auc_pipeline 的外层
（为了逐 token 播种、换打分矩阵），这里验证三件事：

  1. seeding="global" + model 打分 == 上游 evaluate_auc_pipeline，逐 token 的 AUC 与 DeLong 方差
  2. dedup 数据上 first_occurrence 是恒等变换（全局首次出现去重已保证）
  3. 逐 token 播种下，抽到的观测与 token 列表顺序无关（反转顺序，结果不变）

    python test_auc_controls_equiv.py [ckpt] [dataset]
"""
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from evaluate_auc import evaluate_auc_pipeline                           # noqa: E402
from evaluate_auc_rosmap import ROSMAP_AGE_GROUPS, build_labels          # noqa: E402
import evaluate_auc_rosmap_controls as C                                 # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else "Delphi-ROSMAP/ckpt.pt"
DS = sys.argv[2] if len(sys.argv) > 2 else "rosmap"
SEED = 1337
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


model, ck = C.load_model(CKPT, "cpu")
d, _ = C.load_val_batch(f"data/{DS}", ck["model_args"]["block_size"])
tok = C.select_tokens(f"data/{DS}", 100)
tokens = tok["token"].to_numpy()
p = C.model_scores(model, d, tokens, "cpu")

train = np.fromfile(f"data/{DS}/train.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
ids, counts = np.unique(train[:, 2] + 1, return_counts=True)
labels = build_labels(f"data/{DS}", dict(zip(ids.tolist(), counts.tolist())))

for offset in (0.1, 365.25):
    np.random.seed(SEED)
    _, up = evaluate_auc_pipeline(model, [torch.from_numpy(x) for x in d], None, labels,
                                  diseases_of_interest=tokens.tolist(), age_groups=ROSMAP_AGE_GROUPS,
                                  offset=offset, device="cpu", seed=SEED, n_bootstrap=1)
    mine = C.pool(C.auc_rows(d, p, tokens, offset, SEED, seeding="global"))
    m = up[["token", "auc", "auc_variance_delong"]].merge(mine, on="token")
    check(f"offset={offset}: 与上游 token 集合一致", len(m) == len(up) == len(mine),
          f"({len(up)} vs {len(mine)})")
    da = np.abs(m["auc_x"] - m["auc_y"]).max()
    # 上游 fastDeLong 的 AUC 是 float32（tz 数组），各层取平均时两边累加顺序不同，
    # 会差一个 float32 ulp（~1.2e-7）。抽样是否一致由下面的方差检查（~1e-19）判定。
    check(f"offset={offset}: AUC 一致（float32 精度内）", da < 1e-6, f"(max |Δ| {da:.2e})")
    dv = np.abs(up.set_index("token")["auc_variance_delong"].astype(float)
                - mine.set_index("token")["auc_var"]).max()
    check(f"offset={offset}: DeLong 方差一致", dv < 1e-9, f"(max |Δ| {dv:.2e})")

y_fo = C.first_occurrence(d[2])
check("dedup 上 first_occurrence 是恒等变换" if DS == "rosmap" else "first_occurrence 改动的位置数",
      DS != "rosmap" or np.array_equal(y_fo, d[2]), f"({int((y_fo != d[2]).sum())} 个位置被改)")

a = C.pool(C.auc_rows(d, p, tokens, 0.1, 3)).set_index("token")
b = C.pool(C.auc_rows(d, p[..., ::-1], tokens[::-1], 0.1, 3)).set_index("token")
check("逐 token 播种与 token 顺序无关", np.abs(a["auc"] - b.loc[a.index, "auc"]).max() < 1e-9)

# trunc：k 足够大时重建出来的就是完整序列，必须与 model 打分一致（只比预测点）。
# 交付版 ckpt 没有 pos_embedding，左侧 padding 个数变了不影响；model 打分是 float16，容差按它给。
if not model.config.pos_embedding:
    pt = C.truncated_scores(model, d, tokens, 10**6, "cpu")
    C.assert_scored(pt, d, 0.1)
    ok = ~np.isnan(pt[..., 0]) & (d[0] != C.PADDING)
    dm = np.abs(pt[ok] - p[ok].astype(np.float32)).max()
    check("trunc(k=∞) 与 model 打分一致", dm < 2e-2, f"(max |Δ| {dm:.2e}, {ok.sum()} 个预测点)")
    pt1 = C.truncated_scores(model, d, tokens, 1, "cpu")
    C.assert_scored(pt1, d, 365.25)
    d1 = np.abs(pt1[ok] - pt[ok]).max()
    check("trunc(k=1) 确实和完整历史不同", d1 > 0.1, f"(max |Δ| {d1:.2e})")

# query：预测点之后的 token 怎么改，查询分数都必须逐位不变（输入在预测点截断）
sub = [x[:24] for x in d]
pq = C.query_scores(model, sub, tokens, "cpu")
C.assert_scored(pq, sub, 0.1)
sub2 = [x.copy() for x in sub]
cut = 40
late = sub2[0][:, cut + 1:] > C.STATIC_MAX
sub2[0][:, cut + 1:][late] = 34 + (sub2[0][:, cut + 1:][late] - 34 + 17) % 95
pq2 = C.query_scores(model, sub2, tokens, "cpu")
okq = ~np.isnan(pq[:, : cut + 1, 0])
dq = np.abs(pq[:, : cut + 1][okq] - pq2[:, : cut + 1][okq]).max()
check("query：预测点之后的 token 改掉，分数逐位不变", dq == 0.0, f"(max |Δ| {dq:.2e}, {okq.sum()} 个点)")
# no-event 预测点上，query 与原读分掌握的信息相同 -> 应高度相关
ne = (sub[0] == C.NO_EVENT) & ~np.isnan(pq[..., 0])
r = np.corrcoef(pq[ne].ravel(), p[:24][ne].astype(np.float32).ravel())[0, 1]
check("query 与原读分在 no-event 预测点上高度相关", r > 0.95, f"(r = {r:.3f}, {ne.sum()} 个点)")

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(fails > 0)
