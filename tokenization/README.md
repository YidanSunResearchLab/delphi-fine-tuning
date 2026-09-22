# ROSMAP → Delphi-2M Tokenization

把 RADC ROSMAP/MAP/LATC 三个数据集转成 [Delphi-2M](https://github.com/gerstung-lab/delphi) 的训练格式。
纯硬分箱，临床切点优先（与 Delphi 对 BMI 的做法一致）。

## 文件

| 文件 | 说明 |
|---|---|
| `spec.py` | 词表与分箱规格：每个变量用什么切点、分几箱 |
| `build.py` | 转换脚本，重新生成全部输出 |
| `data_rosmap/train.bin` | 训练集，`np.uint32` 的 `(patient_id, age_days, token_id)` |
| `data_rosmap/val.bin` | 验证集 |
| `data_rosmap/labels.csv` | 词表，第 N+1 行对应 token N |
| `data_rosmap/meta.json` | token 映射、区间、排除变量、分层 gap 分布、去重豁免 |
| `data_rosmap_nodedup/` | **消融分支**：认知量表每次随访都发射，不去重。见下节 |
训练配置不在这里，见 `../delphi/config/train_delphi_rosmap.py`（避免两份副本漂移）。

## 用法

数据已经装进 `../delphi`（gerstung-lab/delphi 的 clone）：

```bash
cd ../delphi
python train.py config/train_delphi_rosmap.py --device=mps   # checkpoint 落在 Delphi-ROSMAP/
```

重新生成数据后需要同步过去：

```bash
cp data_rosmap/{train.bin,val.bin,labels.csv,meta.json} ../delphi/data/rosmap/
```

## 消融分支：`--nodedup`（认知量表不去重）

交付版对每个纵向连续量施加**两级**去重，两级都在 `build.py` 里：

1. **run-length**（`build.py` 的 CONT 循环）：只在箱发生变化时发射；
2. **全局 first-occurrence**（`seen` 那段）：每个 token id 每人最多出现一次。

第 2 级比看起来更狠：一个在 26 分附近来回跳的人，真实轨迹是 `N-B-N-B`，`.bin` 里只剩 `N-B`,
**往复被结构性地删掉了**。`figure2_eval/README.md` 的"下一步"第 4 条就是这件事。

`--nodedup` 把列出的前缀从**两级**去重里都豁免掉，改成每次随访发射一个 token：

```bash
python build.py --out data_rosmap_nodedup --dataset rosmap_nodedup --nodedup MMSE,COGN
cp data_rosmap_nodedup/{train.bin,val.bin,labels.csv,meta.json} ../delphi/data/rosmap_nodedup/
```

两份分词的**词表完全相同**（129 个 token，`ignore_tokens` 也相同）——只改发射次数，不新增 token。
这正是它危险的地方：`figure2_eval/radc_delphi/engine.py:load` 只能校验 `vocab_size` 和
`ignore_tokens`，所以**用交付版 ckpt 去评不去重的数据不会崩**，只会安静地给出一张错图。
配对的方式是 `meta.json` 里的 `repeatable_token_prefixes`，下游据此决定 rollout 的 no-repeat 规则。

| | 交付版 `data_rosmap` | `--nodedup MMSE,COGN` |
|---|---|---|
| token 行数 | 158,103 | **212,078**（+34%） |
| 其中认知 token | 15,029（9.5%），人均 3.4 | **69,004（32.5%）**，人均 15.6 |
| 序列长度 中位 / p99 / max | 35 / 57 / 65 | **46 / 90 / 114** |
| 含 no-event 标记后的 max | 85 | **134** → `block_size` 96 → **144** |
| MMSE 往复（回到曾经离开过的箱） | **0**（结构上不可能） | **2,794** |
| `cogn_global` 往复 | **0** | **2,426** |
| 非基线访视的临床 token 数（均值） | 3.22 | 3.77 |
| train/val 划分 | 3,985 / 443 人 | **同一划分**（RNG 抽取次数没变） |

人和划分都一样，所以两份数据可以逐人对读。

`build.py` 里有一条自检：非豁免家族一旦出现重复就 assert 失败，也就是"绕过了全局去重"这件事
不会静默发生。默认分支（不给 `--nodedup`）的输出与交付版 `train.bin/val.bin/labels.csv`
**md5 完全一致**，已验证。

## 对 delphi 上游代码的两个补丁

其余训练逻辑、损失、augmentation 全部是上游原样。

1. `get_batch` 把 lifestyle token 范围从硬编码的 UKB `3..11` 改成可配置的
   `lifestyle_token_range`，ROSMAP 传 `3..32`（见配置里的 `lifestyle_token_min/max`）。
2. `get_batch` 的 `torch.argsort(ages, 1)` 加 `stable=True`。本仓库的年度网格让同一次访视内
   所有 token 年龄完全相同（83.81% 的 Δt = 0），非稳定排序会让访视内 token 顺序随平台/torch
   版本变化，`evaluate_auc` 的 `pred_idx` 因此选到不同的预测点，AUC 能差到 0.07（实测
   `Death` 在 macOS/MPS 与 Linux/CPU 上分别是 0.791 和 0.827）。UKB 的 tie 少得多所以上游没暴露。
   加 `stable=True` 后两个平台的 `x/a/y/b` 张量 hash 完全一致。

## AUC 评估

`../delphi/evaluate_auc_rosmap.py`，套用上游 `evaluate_auc.evaluate_auc_pipeline`（打分、
case/control 构造、年龄-性别分层、每人每段取一个观测、DeLong 方差全部原样），只替换三样
UKB 专用的东西：按 token 名前缀造 chapter 表、年龄段改 `arange(65, 100, 5)`、
用 DeLong 而非 CUDA-only 的 bootstrap。

```bash
cd ../delphi && python evaluate_auc_rosmap.py --offset=0.1      # 结果见 Delphi-ROSMAP/auc/
```

注意：因为时间轴是年度网格，`offset=0.1` 这个上游叫 "no gap" 的设置在本数据上**实际已经是
提前 1 年**（同龄 token 被 offset 排除），`offset=365.25` 对应提前 2 年。

## 规模

- **127 个 token**（vocab_size = 129，含 Padding + No event）
- **158,103 条记录 / 4,428 人**（train 3,985 人 / 142,528 条，val 443 人 / 15,575 条）
- 序列长度：中位 **35**，p90 49，p95 52，p99 57，max 65 → `block_size = 96`（留出 no-event token 的余量）
- 年龄范围 36.5 – 109.1 岁

## 词表结构（pre-shift id）

| 区间 | 内容 |
|---|---|
| 1–2 | 性别 |
| 3–32 | 背景块：队列/教育/种族/族裔/APOE/吸烟/饮酒 + 5 个基线已患病 |
| 33–92 | 事件、用药开关、纵向连续量分箱 |
| 93 | `Death` |
| 94–127 | 死后病理（12 个变量） |

`ignore_tokens` = padding + 性别 + 背景块，post-shift 即 `[0] + range(2, 34)`，与 Delphi 的约定一致。
（`labels.csv` 用的是 post-shift id = pre-shift + 1。）
背景块同时会被 `lifestyle_augmentations` 随机偏移年龄（−20 ~ +40 年），用于消除 immortality bias。

## 关键设计决定

1. **死亡放在整数年网格上**：`末次访视年龄 + 整数年`，与访视间隔同一尺度，避免死亡在时间轴上可识别。
   整数 gap 按**末次访视年龄分层**抽样（实测均值：<80 岁 1.86 年 → ≥88 岁 1.07 年），
   有精确 `age_death` 的 975 人用真值取整；`'90+'` 的人强制死亡年龄 ≥ 90；上限 110 岁。
2. **没有 `[CENSORED]` token**：活人序列直接结束，右删失由 padding + 自动注入的 no-event token 处理（同 Delphi）。
3. **连续量去重**：只在箱发生变化时发射 token，整体压缩约 41%。另有一级全局 first-occurrence
   去重（每 token 每人一次）。两级都可以用 `--nodedup` 按前缀豁免，见上节。
4. **用药只发射状态变化**：`_ON` / `_OFF` 成对，不重复声明初始状态。
5. **病理 token 放在 `[DEATH]` 之后**：只有尸检可得，放死亡之前是泄漏。

## 排除的 13 个变量

见 `meta.json` 的 `excluded` 字段。最关键的三个：

- `cogng_demog_slope` — 严重泄漏，由含未来的全部 `cogn_global` 拟合而来（实测 r = 0.83）
- `cogdx` / `dcfdx_lv` — 研究结束时的汇总诊断，当输入即泄漏
- `pmi` — 尸检排期而非病人属性，会让扰动引擎产生虚假靶点

## 时间轴：强制 365 天网格

`age_days = round(age_bl × 365.25) + k × 365`，其中 k 是距基线的整年数。
基线年龄保留真实天数，之后每年恰好 +365 天。

**严格校验结果：全部 Δt 都是 365 的整数倍，0 处违规。**
Δt 取值仅 `{0, 365, 730, 1095, ...}` 共 19 个。

| Δt | 占比 |
|---|---|
| 0 天（同一次访视内） | 83.81% |
| 365 天 | 13.62% |
| 730 天 | 1.89% |

`DEATH` 的 Δt 众数是 1年(73%) / 2年(18%) / 3年(5%)，与普通 token 同形，**无法据此识别死亡**。

配套改动：
- `AD_DX` 吸附到最近整年网格（位移中位 26 天，均值 46 天，最大 182 天；17.7% 位移超过 90 天）
- 死后病理与 `DEATH` **同一天**（Δt = 0），不再是死后 1 天

## 已知限制

- 时间轴是年度网格，丢失了真实访视日期的 ±数十天抖动；`AD_DX` 为对齐网格损失了中位 26 天的精度。
  申请 RADC 的 `age_at_visit`（id=617）可恢复真实连续时间。
- AD 确诊只有 `age_first_ad_dx` 单一时点，没有逐访诊断。可申请 `dcfdx`（id=349）。
- 语料量级（15.8 万条记录 / 4,428 人）远小于 UKB（40 万人），配置里已把模型缩小（6 层 / 96 维，0.69M 参数）。
