"""
test_scope_mask.py -- 改动 C（attn_visits / static_only）的防泄漏测试。

figure 3 的 transformer 基线的全部意义是"看不到某部分信息"。泄漏一点，基线就会偷偷变强，
"主模型 − 基线"就变小，而且不会有任何报错。所以这里不看形状，只做扰动实验：

  * 往**不该看到**的位置换 token / 改年龄 -> 预测点的 logits 必须**逐位不变**
  * 往**应该看到**的位置换 token          -> logits 必须**变**（否则上一条是空检验）

预测点取"每次访视的最后一个位置"，也就是 Delphi AUC 实际读分数的地方。
另外钉死默认路径：两个开关关着时，与不带这两个字段的配置逐位相同。

    python test_scope_mask.py
"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from model import Delphi, DelphiConfig          # noqa: E402
from utils import get_batch, get_p2i            # noqa: E402

SMAX = 33
BASE = dict(vocab_size=129, block_size=144, n_layer=3, n_head=2, n_embd=32, dropout=0.0,
            bias=False, mask_ties=True, t_min=0.1, ignore_tokens=[0] + list(range(2, 34)))
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def model(**kw):
    torch.manual_seed(0)
    return Delphi(DelphiConfig(**BASE, **kw)).eval()


def logits(m, x, a, y, b):
    with torch.no_grad():
        return m(x, a, y, b)[0]


val = np.fromfile("data/rosmap_nodedup/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
p2i = get_p2i(val)
x, a, y, b = get_batch(range(64), val, p2i, select="left", block_size=144, device="cpu",
                       padding="random", no_event_token_rate=5)


def visits(row):
    """该行的 (访视年龄, [位置]) 列表，只算非背景真实 token；以及每次访视的最后一个位置。"""
    real = np.where(x[row].numpy() > SMAX)[0]
    ages = a[row, real].numpy()
    out = []
    for t in np.unique(ages):
        out.append((t, real[ages == t]))
    return out


def perturb_ids(pos_list, row):
    x2 = x.clone()
    for p in pos_list:
        x2[row, p] = 34 + (x2[row, p] - 34 + 17) % (129 - 34)     # 换成另一个非背景 token
    return x2


# ------------------------------------------------------------------ 默认路径
m_ref = model()
m_off = model(attn_visits=0, static_only=False, static_token_max=SMAX)
check("默认路径：开关关着时逐位相同", torch.equal(logits(m_ref, x, a, y, b), logits(m_off, x, a, y, b)))

# ------------------------------------------------------------------ attn_visits
# 只取"目标在更晚时间"的预测点：这正是 Delphi AUC 读分数的位置。目标和自己同龄时
# （窗口末尾、死亡后同龄的神经病理 token），mask_ties 按设计会遮掉当前访视，上游也一样。
rows = [r for r in range(len(x)) if len(visits(r)) >= 4][:10]


def scope_checks(m, xx, aa, yy, bb, tag):
    base = logits(m, xx, aa, yy, bb)
    leak, dead, n = 0.0, [], 0
    for r in rows:
        real = np.where(xx[r].numpy() > SMAX)[0]
        ages = aa[r, real].numpy()
        vs = [real[ages == t] for t in np.unique(ages)]
        for v in range(2, len(vs)):
            q = vs[v].max()
            if not bb[r, q] > aa[r, q]:
                continue
            n += 1
            hidden = np.concatenate(vs[:v])
            x_h = xx.clone(); x_h[r, hidden] = 34 + (x_h[r, hidden] - 34 + 17) % 95
            leak = max(leak, (logits(m, x_h, aa, yy, bb)[r, q] - base[r, q]).abs().max().item())
            if len(vs[v]) > 1:
                seen = vs[v][:-1]
                x_s = xx.clone(); x_s[r, seen] = 34 + (x_s[r, seen] - 34 + 17) % 95
                dead.append((logits(m, x_s, aa, yy, bb)[r, q] - base[r, q]).abs().max().item())
    check(f"attn_visits=1{tag}：更早的访视换 token，预测点逐位不变", n > 0 and leak == 0.0,
          f"({n} 个点, max |Δ| {leak:.2e})")
    check(f"attn_visits=1{tag}：当前访视其它 token 换掉，预测点会变", len(dead) > 0 and min(dead) > 0,
          f"(min |Δ| {min(dead):.2e})")


scope_checks(model(attn_visits=1), x, a, y, b, "")

# 训练时 lifestyle_augmentations 会把背景块挪到访视之间；背景 token 若能看它所在的访视，
# 就会把那次访视转交给后面的预测点。用增强过的 batch 再验一遍。
xa, aa_, ya, ba = get_batch(range(64), val, p2i, select="left", block_size=144, device="cpu",
                            padding="random", no_event_token_rate=5, lifestyle_augmentations=True,
                            lifestyle_token_range=(3, 32))
moved = ((xa >= 4) & (xa <= SMAX) & (aa_ > aa_[:, :1] + 3650)).sum().item()
check("增强后确实有背景 token 被挪到了后面的访视之间", moved > 0, f"({moved} 个)")
x_bak, a_bak, y_bak, b_bak = x, a, y, b
x, a, y, b = xa, aa_, ya, ba
rows = [r for r in range(len(x)) if len(visits(r)) >= 4][:10]
scope_checks(model(attn_visits=1), x, a, y, b, "（背景块被增强打乱）")
x, a, y, b = x_bak, a_bak, y_bak, b_bak
rows = [r for r in range(len(x)) if len(visits(r)) >= 4][:10]

try:
    logits(model(attn_visits=2), x, a, y, b)
    check("attn_visits=2 必须拒绝（滑动窗口跨层泄漏）", False)
except AssertionError:
    check("attn_visits=2 必须拒绝（滑动窗口跨层泄漏）", True)

# 访视之间的 no-event 预测点看到的是上一次访视
m1 = model(attn_visits=1)
base = logits(m1, x, a, y, b)
checked = 0
leak = 0.0
for r in rows:
    vs = visits(r)
    ne = np.where(x[r].numpy() == 1)[0]
    for v in range(1, len(vs) - 1):
        between = ne[(a[r, ne].numpy() > vs[v][0]) & (a[r, ne].numpy() < vs[v + 1][0])]
        if len(between) == 0:
            continue
        q = between.max()
        out = logits(m1, perturb_ids(vs[v - 1][1], r), a, y, b)[r, q]
        leak = max(leak, (out - base[r, q]).abs().max().item())
        checked += 1
check("attn_visits=1：访视之间的 no-event 点看不到上上次访视", checked > 0 and leak == 0.0,
      f"({checked} 个点, max |Δ| {leak:.2e})")

# ------------------------------------------------------------------ static_only
ms = model(static_only=True)
base = logits(ms, x, a, y, b)
nonstatic = x > SMAX
x2 = x.clone()
x2[nonstatic] = 34 + (x2[nonstatic] - 34 + 17) % (129 - 34)
d = (logits(ms, x2, a, y, b) - base).abs().max().item()
check("static_only：**所有**非背景 token 换掉（含预测点自己），全部位置逐位不变", d == 0.0, f"(max |Δ| {d:.2e})")

# 其它非背景 token 的年龄也不能漏进来（只允许预测点自己的年龄）
r = rows[0]
vs = visits(r)
q = vs[-1][1].max()
a2 = a.clone()
for _, pos in vs[:-1]:
    a2[r, pos] += 0.5                                           # 不改变排序、不跨访视
d = (logits(ms, x, a2, y, b)[r, q] - base[r, q]).abs().max().item()
check("static_only：更早的非背景 token 改年龄，预测点逐位不变", d == 0.0, f"(max |Δ| {d:.2e})")

stat_pos = np.where(((x[r] >= 4) & (x[r] <= SMAX)).numpy())[0]
x3 = x.clone()
x3[r, stat_pos[0]] = 4 + (x3[r, stat_pos[0]] - 4 + 1) % (SMAX - 4 + 1)
d = (logits(ms, x3, a, y, b)[r, q] - base[r, q]).abs().max().item()
check("static_only：背景 token 换掉，预测点会变", d > 0, f"(|Δ| {d:.2e})")

# ------------------------------------------------------------------ 反向 + NaN
for kw in (dict(attn_visits=1), dict(static_only=True)):
    m = model(**kw).train()
    lg, loss, _ = m(x, a, y, b)
    tot = sum(v for v in loss.values())
    tot.backward()
    ok = torch.isfinite(lg).all().item() and torch.isfinite(tot).item()
    check(f"{kw}：前向无 NaN、反向可跑", ok, f"(loss {tot.item():.3f})")


# ------------------------------------------------------------------ snapshot：背景块每次访视都重发
# static_first_only 必须让重复的背景拷贝对两个基线都不可见，否则数拷贝份数 = 数访视次数。
# 扰动方式只动"非首份"的背景拷贝，并保证它们扰动后仍是非首份（换成本行已出现过的背景 id），
# 所以任何输出变化都只能来自"拷贝本身被看见了"。
import os
if os.path.exists("data/rosmap_snapshot/val.bin"):
    sv = np.fromfile("data/rosmap_snapshot/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    sp = get_p2i(sv)
    xs, as_, ys, bs = get_batch(range(32), sv, sp, select="left", block_size=736, device="cpu",
                                padding="random", no_event_token_rate=5)
    stat = (xs >= 2) & (xs <= SMAX)
    first = torch.zeros_like(stat)
    for r in range(xs.shape[0]):
        seen = set()
        for i in range(xs.shape[1]):
            k = int(xs[r, i])
            if stat[r, i] and k not in seen:
                first[r, i] = True
            if stat[r, i]:
                seen.add(k)
    dup = stat & ~first
    check("snapshot：确实有重复的背景拷贝", dup.sum().item() > 0, f"({dup.sum().item()} 个)")
    x2, a2 = xs.clone(), as_.clone()
    for r in range(xs.shape[0]):
        fr = xs[r][first[r]]
        if len(fr) == 0:
            continue
        d = dup[r].nonzero().squeeze(-1)
        x2[r, d] = fr[torch.arange(len(d)) % len(fr)]            # 换成本行已出现过的背景 id -> 仍是非首份
        a2[r, d] = a2[r, d] + 0.25                                # 年龄也挪一点（不跨访视、不改排序）
    for kw, tag in [(dict(static_only=True, static_first_only=True), "static_only"),
                    (dict(attn_visits=1, static_first_only=True), "attn_visits=1")]:
        m = model(**kw)
        b0, b1 = logits(m, xs, as_, ys, bs), logits(m, x2, a2, ys, bs)
        # 只看"目标在更晚时间"的真实预测点（Delphi AUC 读分数的位置），且预测点本身不是被扰动的拷贝
        q = (bs > as_) & (xs > SMAX)
        dq = (b0 - b1).abs().max(-1).values[q].max().item()
        check(f"snapshot {tag}：重复背景拷贝（id/年龄）扰动后预测点逐位不变", dq == 0.0, f"(max |Δ| {dq:.2e})")
    m = model(attn_visits=1, static_first_only=False)
    dq = (logits(m, xs, as_, ys, bs) - logits(m, x2, a2, ys, bs)).abs().max(-1).values[(bs > as_) & (xs > SMAX)].max().item()
    check("对照：不开 static_first_only 时重复拷贝确实会漏（上面不是空检验）", dq > 0, f"(max |Δ| {dq:.2e})")
    m = model(static_first_only=True)
    check("static_first_only 单独打开（无 attn_visits / static_only）不改变默认路径",
          torch.equal(logits(m, xs, as_, ys, bs), logits(model(), xs, as_, ys, bs)))

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(fails > 0)
