"""
test_getbatch_equiv.py -- 默认路径必须与改动前**逐位**相同。

get_batch 新增了 aux / return_visit_size 两条支路。只要 aux=None 且 return_visit_size=False，
它就必须和改动前一字不差 —— 包括 RNG 的抽取次数（gen 被 ix.sum() 播种，多抽一次就全错）。
这个检查很便宜但不可省：前面已经踩过一次"新功能默认开着却没人发现"。

utils_upstream_ref.py 是改动前那份 utils.py 的逐字副本。
"""
import sys, numpy as np, torch
sys.path.insert(0, ".")
import utils as new
import utils_upstream_ref as old

d = np.fromfile("data/rosmap_nodedup/train.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
p2i = new.get_p2i(d)
rng = np.random.default_rng(0)
bad = 0
for trial in range(6):
    ix = rng.integers(0, len(p2i), 128).tolist()
    kw = dict(block_size=144, device="cpu", select="random", padding="random",
              lifestyle_augmentations=(trial % 2 == 1), lifestyle_token_range=(3, 32),
              no_event_token_rate=5)
    A = old.get_batch(ix, d, p2i, **kw)
    B = new.get_batch(ix, d, p2i, **kw)
    assert len(B) == 4, f"默认路径返回了 {len(B)} 项，应该是 4"
    for nm, u, v in zip("xayb", A, B):
        if not torch.equal(u, v):
            bad += 1
            print(f"  trial {trial} lifestyle={kw['lifestyle_augmentations']}: {nm} 不一致 "
                  f"(差异 {(u != v).sum().item()} / {u.numel()})")
print("默认路径逐位一致" if bad == 0 else f"**{bad} 处不一致**")

# 打开支路：形状与取值域
K = 15
aux = rng.integers(-1, 2, size=(len(d), K)).astype(np.int8)
x, a, y, b, ax, vs = new.get_batch(list(range(128)), d, p2i, block_size=144, device="cpu",
                                   aux=aux, return_visit_size=True,
                                   lifestyle_augmentations=True, lifestyle_token_range=(3, 32))
assert ax.shape == (*x.shape, K), f"aux {tuple(ax.shape)} 应为 {(*x.shape, K)}"
assert vs.shape == x.shape, f"visit_size {tuple(vs.shape)} 应为 {tuple(x.shape)}"
pad = (x == 0)
assert (ax[pad] == -1).all(), "padding 位置的 aux 没有置 -1"
assert (vs[pad] == -1).all(), "padding 位置的 visit_size 没有置 -1"
assert vs.min() >= -1 and vs.max() >= 1, f"visit_size 取值域异常 [{vs.min()},{vs.max()}]"
print(f"支路形状 OK  aux{tuple(ax.shape)} vs{tuple(vs.shape)}  "
      f"有效 aux {int((ax >= 0).sum())}  有效 k {int((vs >= 1).sum())} "
      f"(k 范围 {int(vs[vs>=1].min())}..{int(vs.max())}, 中位 {int(vs[vs>=1].median())})")
# 只要 aux 不要 visit_size / 反之，也必须是 6 元组
assert len(new.get_batch(list(range(8)), d, p2i, block_size=144, aux=aux)) == 6
assert len(new.get_batch(list(range(8)), d, p2i, block_size=144, return_visit_size=True)) == 6
print("单独打开任一支路也返回 6 元组 OK")
