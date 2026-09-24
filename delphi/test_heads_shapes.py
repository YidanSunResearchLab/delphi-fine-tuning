"""
test_heads_shapes.py -- 形状 / 对齐 / "关着的时候什么都没变" 的最小回归测试。

    python test_heads_shapes.py                 # 全部（CPU，约 1 秒）
    python test_heads_shapes.py --ref /path/to/pristine/model.py   # 再和改动前的 model.py 对数

覆盖 `aux_head` × `visit_heads` 四种组合的前向 + 反向，以及三件"错了不会报错"的事：

  1. **(False, False) 必须和改动前逐字符一样。** 参数量、state_dict 的 key 集合、loss dict
     的 key 集合都钉死；给了 --ref 还会把 loss 的数值和改动前的 model.py 逐位对比。
  2. **get_batch 的 aux 对齐。** 用"标签第 0 列 = 这一行的 disk token"这个构造，跑完整条
     mask / 注入 / 稳定排序 / 两次裁剪 / +1 位移的流水线，再断言
     `aux[..., 0] == x - 1`。这是整个改动里最容易静默错的一步：错位之后 BCE 照样收敛。
  3. **访视大小标签。** 手算一个已知访视结构的病人，逐个位置对答案，并逐条验剔除规则
     （右边界截断的半次访视、注入的 no-event 不开访视、被年龄抖动过的背景块 token）。

不用 pytest：集群环境里不保证装了。失败就 AssertionError，exit code 非 0。
"""
import os
import sys
import argparse
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model import Delphi, DelphiConfig          # noqa: E402
from utils import get_batch, get_p2i            # noqa: E402

# 交付版配方（config/train_delphi_rosmap_nodedup.py），只有 block_size 是 144 那一版。
BASE = dict(vocab_size=129, block_size=144, n_layer=6, n_head=6, n_embd=96,
            dropout=0.0, bias=False, mask_ties=True, t_min=0.1,
            ignore_tokens=[0] + list(range(2, 34)))
N_AUX = 15          # 3 个终点 × 5 个 horizon，见 make_aux_labels.py
MAX_VISIT = 24

# 改动前的参数量。bias 决定用哪一个 —— 交付版 ckpt 的 model_args 里 bias=False
# （Delphi-ROSMAP/ckpt.pt 实测 686,400 个参数；注意 state_dict 里的 754,128 还包含
# 6 个 attn.bias 因果掩码 buffer、AgeEncoding 的 div_term，以及和 wte 共享存储的
# lm_head.weight —— 那三样都不是独立参数）。
EXPECTED_PARAMS = {False: 686_400, True: 692_832}
_OK = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name} {detail}"
    _OK.append(name)
    print(f"  ok  {name}{('  ' + detail) if detail else ''}")


def build(aux_head=False, visit_heads=False, seed=0):
    torch.manual_seed(seed)
    cfg = DelphiConfig(**BASE, aux_head=aux_head, aux_n_targets=(N_AUX if aux_head else 0),
                       aux_lambda=1.0, visit_heads=visit_heads, max_visit_size=MAX_VISIT)
    return Delphi(cfg)


# --------------------------------------------------------------------------- 合成数据
def synth_dataset(n_pat=12, seed=0):
    """(data, p2i, aux)：形如 .bin 的 (pid, age_days, disk_token) 三元组。

    aux 的第 0 列刻意放**这一行的 disk token**，第 1 列放行号。对齐测试全靠这两列。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        age = int(rng.integers(25000, 30000))
        n_visit = int(rng.integers(3, 7))
        # 第一次访视兼作 statics 块（本 build 里 statics 和首访同龄），所以更大
        for v in range(n_visit):
            k = int(rng.integers(6, 10)) if v == 0 else int(rng.integers(1, 5))
            for _ in range(k):
                rows.append((p, age, int(rng.integers(2, 127))))
            age += int(rng.integers(300, 500))
    data = np.array(rows, dtype=np.uint32)
    aux = np.full((len(data), N_AUX), -1, dtype=np.int8)
    aux[:, 0] = data[:, 2].astype(np.int8)           # = disk token
    aux[:, 1] = (np.arange(len(data)) % 100).astype(np.int8)
    # 其余列放随机三值标签，用来喂损失
    aux[:, 2:] = np.random.default_rng(seed + 1).integers(-1, 2, size=(len(data), N_AUX - 2))
    return data, get_p2i(data), aux


def fixed_patient():
    """一个手算得出答案的病人：访视大小 3 / 2 / 1，第二个病人只为制造边界。"""
    rows = [(7, 30000, 10), (7, 30000, 11), (7, 30000, 12),
            (7, 30365, 13), (7, 30365, 14),
            (7, 30730, 15),
            (9, 20000, 20), (9, 20000, 21),
            (9, 20365, 22), (9, 20365, 23), (9, 20365, 24), (9, 20365, 25)]
    data = np.array(rows, dtype=np.uint32)
    aux = np.full((len(data), N_AUX), -1, dtype=np.int8)
    aux[:, 0] = data[:, 2].astype(np.int8)
    return data, get_p2i(data), aux


# --------------------------------------------------------------------------- 测试
def t_params_and_keys():
    print("\n[1] 参数量 / state_dict key —— (F,F) 必须和改动前一致")
    m00 = build(False, False)
    n00 = m00.get_num_params()
    check("(F,F) 参数量", n00 == EXPECTED_PARAMS[BASE["bias"]],
          f"{n00} vs {EXPECTED_PARAMS[BASE['bias']]}")
    k00 = set(m00.state_dict())
    for bad in ("aux_head.weight", "time_head.weight", "size_head.weight"):
        check(f"(F,F) 没有 {bad}", bad not in k00)

    m10 = build(True, False)
    check("(T,F) 只多了 aux_head", set(m10.state_dict()) - k00 == {"aux_head.weight", "aux_head.bias"})
    check("(T,F) 参数量增量", m10.get_num_params() - n00 == (96 + 1) * N_AUX,
          f"+{m10.get_num_params() - n00}")

    m01 = build(False, True)
    check("(F,T) 只多了 time/size head",
          set(m01.state_dict()) - k00 == {"time_head.weight", "time_head.bias",
                                          "size_head.weight", "size_head.bias"})
    check("(F,T) 参数量增量", m01.get_num_params() - n00 == (96 + 1) + (96 * MAX_VISIT + MAX_VISIT),
          f"+{m01.get_num_params() - n00}")

    m11 = build(True, True)
    check("(T,T) 参数量 = 两个增量之和",
          m11.get_num_params() - n00 == (m10.get_num_params() - n00) + (m01.get_num_params() - n00))

    # 新参数必须落进优化器的某一组，否则它们永远不被更新（而且不会有任何报错）
    for tag, m in (("(T,F)", m10), ("(F,T)", m01), ("(T,T)", m11)):
        opt = m.configure_optimizers(0.2, 6e-4, (0.9, 0.99), "cpu")
        seen = {id(p) for g in opt.param_groups for p in g["params"]}
        missing = [n for n, p in m.named_parameters() if id(p) not in seen
                   and n != "lm_head.weight"]      # 和 wte 共享存储，上游刻意只登记一次
        check(f"{tag} 所有参数都进了优化器", not missing, str(missing))
        decay = {id(p) for p in opt.param_groups[0]["params"]}
        if tag != "(F,T)":
            check(f"{tag} aux_head.weight 在 decay 组", id(m.aux_head.weight) in decay)
        if tag != "(T,F)":
            check(f"{tag} size_head.weight 在 decay 组", id(m.size_head.weight) in decay)
            check(f"{tag} time_head.bias 在 no_decay 组",
                  id(m.time_head.bias) in {id(p) for p in opt.param_groups[1]["params"]})


def t_forward_backward():
    print("\n[2] 四种组合的前向 + 反向")
    data, p2i, aux = synth_dataset()
    ix = torch.arange(8)
    for a_flag in (False, True):
        for v_flag in (False, True):
            tag = f"(aux={int(a_flag)}, visit={int(v_flag)})"
            m = build(a_flag, v_flag)
            out = get_batch(ix, data, p2i, block_size=48, device="cpu",
                            select="left", padding="random",
                            no_event_token_rate=5, cut_batch=True,
                            aux=(aux if a_flag else None),
                            return_visit_size=v_flag)
            # (F,F) 走的是老契约：4 元组。这一行本身就是在测那个契约。
            check(f"{tag} get_batch 返回长度", len(out) == (4 if not (a_flag or v_flag) else 6))
            X, A, Y, B, AUX, KSZ = (*out, None, None)[:6]
            _, loss, _ = m(X, A, Y, B, aux_targets=AUX, visit_size_targets=KSZ)
            want = {"loss_ce", "loss_dt"} | ({"loss_aux"} if a_flag else set()) \
                | ({"loss_size"} if v_flag else set())
            check(f"{tag} loss keys", set(loss) == want, str(sorted(loss)))
            for k, v in loss.items():
                check(f"{tag} {k} 有限且是标量", v.ndim == 0 and torch.isfinite(v), f"{v.item():.4f}")
            total = sum(loss.values())
            total.backward()
            g = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
            check(f"{tag} 梯度全部有限", all(torch.isfinite(v).all() for v in g.values()))
            if a_flag:
                check(f"{tag} aux_head 拿到了非零梯度", g["aux_head.weight"].abs().sum() > 0)
            if v_flag:
                check(f"{tag} time_head 拿到了非零梯度", g["time_head.weight"].abs().sum() > 0)
                check(f"{tag} size_head 拿到了非零梯度", g["size_head.weight"].abs().sum() > 0)


def t_aux_masking():
    print("\n[3] -1（删失）必须被真正剔除")
    data, p2i, aux = synth_dataset()
    ix = torch.arange(6)
    m = build(True, False)
    X, A, Y, B, AUX, _ = get_batch(ix, data, p2i, block_size=48, device="cpu", select="left",
                                   no_event_token_rate=0, padding="none", cut_batch=True, aux=aux)
    # (a) 整批全 -1 -> loss 恰好 0，且不是 NaN（分母 clamp(1) 的意义）
    _, loss, _ = m(X, A, Y, B, aux_targets=torch.full_like(AUX, -1))
    check("全删失时 loss_aux == 0", float(loss["loss_aux"]) == 0.0)
    # (b) 把某个有效位置的标签翻面，损失必须变；把某个 -1 位置改成别的 -1，损失必须不变
    _, l0, _ = m(X, A, Y, B, aux_targets=AUX)
    flip = AUX.clone()
    valid = (AUX >= 0)
    if bool(valid.any()):
        j = valid.nonzero()[0]
        flip[j[0], j[1], j[2]] = 1 - int(AUX[j[0], j[1], j[2]])
        _, l1, _ = m(X, A, Y, B, aux_targets=flip)
        check("翻一个有效标签 -> loss_aux 变", float(l0["loss_aux"]) != float(l1["loss_aux"]))
    # (c) 数值对答案：手算 masked BCE
    with torch.no_grad():
        store = {}
        h = m.transformer.ln_f.register_forward_hook(lambda mod, i, o: store.__setitem__("x", o))
        try:
            m(X, A, Y, B, aux_targets=AUX)
        finally:
            h.remove()
        logit = m.aux_head(store["x"]).reshape(-1, N_AUX)
        tgt = AUX.reshape(-1, N_AUX)
        pt = Y.reshape(-1) != -1
        for k in BASE["ignore_tokens"]:
            pt = pt & (Y.reshape(-1) != k)
        vmask = (tgt >= 0) & pt.unsqueeze(-1)
        bce = F.binary_cross_entropy_with_logits(logit, tgt.clamp(min=0).float(), reduction="none")
        ref = (bce[vmask]).mean()
        _, loss2, _ = m(X, A, Y, B, aux_targets=AUX)
    check("loss_aux == 手算 masked BCE", torch.allclose(loss2["loss_aux"], ref, atol=1e-6),
          f"{float(loss2['loss_aux']):.6f} vs {float(ref):.6f}")


def t_alignment():
    print("\n[4] get_batch：aux 逐行对齐（第 0 列 = 该行的 disk token）")
    data, p2i, aux = synth_dataset()
    for pad_mode, rate in (("none", 0), ("regular", 5), ("random", 5)):
        for sel in ("left", "right", "random"):
            ix = torch.arange(10)
            X, A, Y, B, AUX, _ = get_batch(ix, data, p2i, block_size=32, device="cpu",
                                           select=sel, padding=pad_mode,
                                           no_event_token_rate=rate, cut_batch=True, aux=aux)
            real = (X > 1)                       # 0 = padding, 1 = 注入的 no-event
            check(f"对齐 padding={pad_mode} select={sel}",
                  bool((AUX[..., 0][real].to(torch.int64) == (X[real] - 1)).all()),
                  f"{int(real.sum())} 个真实位置")
            check(f"注入/padding 位置无标签 padding={pad_mode} select={sel}",
                  bool((AUX[~real] == -1).all()))


def t_default_unchanged():
    print("\n[5] aux=None 且 return_visit_size=False 时行为不变")
    data, p2i, aux = synth_dataset()
    ix = torch.arange(10)
    kw = dict(block_size=32, device="cpu", select="left", padding="random",
              lifestyle_augmentations=True, lifestyle_token_range=(3, 32),
              no_event_token_rate=5, cut_batch=True)
    out4 = get_batch(ix, data, p2i, **kw)
    check("默认返回 4 元组", len(out4) == 4)
    out6 = get_batch(ix, data, p2i, aux=aux, return_visit_size=True, **kw)
    check("请求额外项时返回 6 元组", len(out6) == 6)
    for i, nm in enumerate("xayb"):
        check(f"带不带 aux，{nm} 逐位相同", bool(torch.equal(out4[i], out6[i])))


def t_visit_size():
    print("\n[6] 访视大小标签（手算对答案）")
    data, p2i, aux = fixed_patient()
    # 病人 7：访视 3 / 2 / 1 个 token。block_size=8 -> 窗口越过病人末尾，最后一段完整。
    X, A, Y, B, _, K = get_batch(torch.tensor([0]), data, p2i, block_size=8, device="cpu",
                                 select="left", padding="none", no_event_token_rate=0,
                                 return_visit_size=True)
    got = K[0].tolist()
    check("完整窗口的 k", got == [-1, -1, -1, -1, -1, 2, -1, 1], str(got))
    check("有标签的位置正好是每次访视的最后一个 token",
          [int(X[0, i]) - 1 for i, v in enumerate(got) if v > 0] == [12, 14], str(got))

    # 同一个病人，窗口在访视中间截断 -> 最后那段半次访视不能给标签
    X2, _, _, _, _, K2 = get_batch(torch.tensor([0]), data, p2i, block_size=3, device="cpu",
                                   select="left", padding="none", no_event_token_rate=0,
                                   return_visit_size=True)
    check("右边界截断的访视不监督", bool((K2 < 0).all()), str(K2[0].tolist()))

    # 注入的 no-event 不开访视
    data3, p2i3, aux3 = synth_dataset(n_pat=8)
    X3, _, Y3, _, _, K3 = get_batch(torch.arange(8), data3, p2i3, block_size=48, device="cpu",
                                    select="left", padding="regular", no_event_token_rate=5,
                                    cut_batch=True, return_visit_size=True)
    check("目标是 no-event 的位置没有 k", bool((K3[Y3 == 1] == -1).all()))
    check("输入是 padding 的位置没有 k", bool((K3[X3 == 0] == -1).all()))
    check("k 至少为 1", bool((K3[K3 > -1] >= 1).all()))

    # 被年龄抖动过的背景块 token 既不能当输入也不能当目标（它坐在一个假年龄上）
    lo, hi = 3, 32
    X4, _, Y4, _, AUX4, K4 = get_batch(torch.arange(8), data3, p2i3, block_size=48, device="cpu",
                                       select="left", padding="random", no_event_token_rate=5,
                                       lifestyle_augmentations=True, lifestyle_token_range=(lo, hi),
                                       cut_batch=True, aux=aux3,
                                       return_visit_size=True)
    jit_x = (X4 >= lo + 1) & (X4 <= hi + 1)
    jit_y = (Y4 >= lo + 1) & (Y4 <= hi + 1)
    check("抖动过的 token 不参与访视大小", bool((K4[jit_x | jit_y] == -1).all()),
          f"{int((jit_x | jit_y).sum())} 个位置")
    check("抖动过的 token 不参与判别头监督", bool((AUX4[jit_x] == -1).all()),
          f"{int(jit_x.sum())} 个位置")


def t_guards():
    print("\n[7] 形状不符必须炸，不能静默 reshape")
    data, p2i, aux = synth_dataset()
    X, A, Y, B, AUX, _ = get_batch(torch.arange(6), data, p2i, block_size=48, device="cpu",
                                   select="left", padding="none", no_event_token_rate=0,
                                   cut_batch=True, aux=aux)
    m = build(True, False)
    # 列数不等：b*t*N 能被 K 整除时 reshape(-1, K) 会**成功**，每一列学成别的终点。
    # 这正是 resume 旧 ckpt + 重新生成过的 aux_labels_*.npy 的路径。
    bad = AUX[..., :5].contiguous()          # 5 列 vs aux_n_targets=15，15 能被 5 整除
    try:
        m(X, A, Y, B, aux_targets=bad)
        check("列数不符 -> AssertionError", False, "没有报错（静默重解释了标签）")
    except AssertionError as e:
        check("列数不符 -> AssertionError", "列" in str(e), str(e)[:60])
    # 位置对不齐（少一格 = 用下一个 token 的年龄起算 h 年）
    try:
        m(X, A, Y, B, aux_targets=AUX[:, :-1, :].contiguous())
        check("时间维不符 -> AssertionError", False, "没有报错")
    except AssertionError as e:
        check("时间维不符 -> AssertionError", "idx" in str(e), str(e)[:60])
    mv = build(False, True)
    try:
        mv(X, A, Y, B, visit_size_targets=torch.zeros(X.shape[0], X.shape[1] - 1,
                                                      dtype=torch.long))
        check("访视大小时间维不符 -> AssertionError", False, "没有报错")
    except AssertionError as e:
        check("访视大小时间维不符 -> AssertionError", "idx" in str(e), str(e)[:60])


def t_reference(ref_path):
    """和改动前的 model.py 逐位对数。--ref 指向一份改动前的副本时才跑。"""
    print(f"\n[8] 与改动前的 model.py 对数：{ref_path}")
    spec = importlib.util.spec_from_file_location("_ref_model", ref_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    data, p2i, _ = synth_dataset()
    X, A, Y, B = get_batch(torch.arange(8), data, p2i, block_size=48, device="cpu",
                           select="left", padding="random", no_event_token_rate=5, cut_batch=True)
    # 同一个 seed 建模型 -> 权重逐位相同（init 的调用顺序不能变，这一点本身也在被测）
    torch.manual_seed(0)
    old = ref.Delphi(ref.DelphiConfig(**BASE))
    new = build(False, False, seed=0)
    sd_o, sd_n = old.state_dict(), new.state_dict()
    check("state_dict key 集合相同", set(sd_o) == set(sd_n))
    check("权重逐位相同", all(torch.equal(sd_o[k], sd_n[k]) for k in sd_o))
    for mode in (False, True):
        _, lo, _ = old(X, A, Y, B, validation_loss_mode=mode)
        _, ln, _ = new(X, A, Y, B, validation_loss_mode=mode)
        check(f"loss key 相同 (validation_loss_mode={mode})", set(lo) == set(ln))
        for k in lo:
            check(f"{k} 逐位相同 (validation_loss_mode={mode})",
                  lo[k].item() == ln[k].item(), f"{lo[k].item():.10f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="改动前 model.py 的副本，给了就逐位对数")
    a = ap.parse_args()
    torch.set_num_threads(1)
    np.random.seed(0)
    t_params_and_keys()
    t_forward_backward()
    t_aux_masking()
    t_alignment()
    t_default_unchanged()
    t_visit_size()
    t_guards()
    if a.ref:
        t_reference(a.ref)
    print(f"\nALL PASS ({len(_OK)} 项)")


if __name__ == "__main__":
    main()
