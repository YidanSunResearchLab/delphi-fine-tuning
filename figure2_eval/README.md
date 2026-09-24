# Figure 2 → ROSMAP

把 [YidanSunResearchLab/delphi-fine-tuning](https://github.com/YidanSunResearchLab/delphi-fine-tuning)
的 `figure2/` 评估搬到本项目：上游 gerstung-lab Delphi（`../delphi`）+ 本仓库的 129-token ROSMAP
分词（`../tokenization`），checkpoint 是 `../delphi/Delphi-ROSMAP/ckpt.pt`。

## 怎么跑：重计算一律提交 RIS

```bash
# ssh 到 compute1-client，仓库放在 /storage1/fs1/<PI>/Active/ 下
vim ris/env.sh                # 填 COMPUTE_GROUP / STORAGE_ROOT / DOCKER_IMAGE 三个占位符
ris/submit.sh all             # figure2 + predict_vs_observe + 口径扫描 + diagnostics
ris/submit.sh figure2         # 只跑 figure2
DRYRUN=1 ris/submit.sh all    # 只打印 bsub 命令，不提交
bjobs -w                      # 看状态；日志在 results/ris_logs/
```

`ris/submit.sh` 固定住三件忘了就**静默出错**的事：`LSF_DOCKER_VOLUMES` 必须显式挂载
（漏了不报错，只是 python 找不到 `.bin`）、`OMP_NUM_THREADS=1`（MC pass 是 multiprocessing 的，
worker 再开满线程会严重超订）、`-n` 与脚本的 `--workers` 对齐。它还会检查仓库是否在挂载前缀下，
不在就拒绝提交。`DOCKER_IMAGE` 需要 torch + numpy + pandas + matplotlib + **scikit-learn + scipy**
——`delphi/containers/Dockerfile` 是上游的 EMBL/singularity 配置，缺后两个。

本地只适合跑小探查：

```bash
./run_figure2.sh --limit 40    # 冒烟，30 秒；数字不可引用
python3 diagnostics.py --n 20  # 快速诊断
```

全量在本地也能跑（`./run_figure2.sh`，约 5 分钟 / 8 核 + 30 秒绘图），但每改一次采样器就要
全量重跑，所以默认走 RIS。

`results/` = **RIS 权威结果**（compute2，`use_visit_sizes=True`，panel d 用真 UMAP）。
`results/figure2/{gpubase,nodedup,fullvisit}/` 是**去重消融**的三档，见第 6/7 节；
它们和 `results/figure2/` 的 ckpt 都不是同一个，不要混着引用。
`results_local_archive/` 是本地旧版（采样器不同、panel d 是 PCA 回退），**不要引用**，
留着只为跨平台对照——同配置下本地与 RIS 的预测值最大差异 0.000e+00。

输出在 `results/figure2/`，主图 `figure2_combined_matched.png`，每个面板配一份 `*_data.csv`
source data，指标在 `metrics_*.json` / `SUMMARY_*.md`。

**先读第 3 节再引用任何数字。** panel a 的 calibration 那一列被一个已测量、可修复的 rollout
假象主导，而且 split 不是真正的 held-out。

---

## 1. 搬了什么，写了什么

`figure2/` 下的文件是从上游复制过来的，只动了领域映射；真正新写的是一个 `radc_delphi/`
适配层——上游那三个模块绑定的是它自己的模型和 50/56-token RADC 词表，我们得换成我们的。
保留包名是为了让 `figure2/` 的代码几乎不用改，以后上游更新还能直接 diff。

| 文件 | 来源 | 改动 |
|---|---|---|
| `figure2/figure2_core.py` | 上游 | 默认 ckpt/dataset/split；`_engine` 总是按 build 自己的 labels.csv 解析；cohort mask |
| `figure2/figure2_panels.py` | 上游 | **零改动**（面板完全由 `S.NSTAGE` 参数化，3 级和 4 级分期都能画） |
| `figure2/perdomain.py` | 上游 | **零改动**（outcome 集合完全由 `radc_states` 驱动） |
| `figure2/radc_states.py` | 上游 | 分期规格、event 家族、endpoint 标签 —— 见第 2 节 |
| `figure2/plotting_style.py` | 上游 | 色带按 outcome 数量生成；药物按后缀而非药名分类 |
| `figure2/per_state_auc.py`、`fig_perdomain.py` | 上游 | 零改动，未跑过（独立脚本，不进主图） |
| `radc_delphi/vocab.py` | **新写** | 129-token 词表按名字解析成 figure2 要的 id 家族 |
| `radc_delphi/batching.py` | **新写** | `.bin` 布局 + disk→model 的 +1 位移 |
| `radc_delphi/engine.py` | **新写** | 加载我们的 checkpoint，和一个语义正确的 rollout 采样器 |
| `diagnostics.py`、`run_figure2.sh` | 新写 / 改写 | |

上游的 `score_perdomain.py` 没搬：它 import `delphi.predict_adapter`，那个模块在上游仓库里就不存在。

### engine 为什么不能直接用上游 Delphi 的 `generate()`

模型本身原样加载，没改一行。但 `../delphi/model.py:generate` 有三个地方对这个图是错的，而且
**错得很安静**：

1. **`termination_tokens` 默认是 `[1269]`** —— 一个 UKB id，在我们 129 个 token 里根本不存在，
   只用 `warnings.warn` 提示。继承这个默认值意味着模拟出来的人**永生**，会一直累积 AD 风险。
2. **`no_repeat=True` 是一刀切的**：它封掉轨迹里出现过的**每一个** token。对一次性事件（第二个
   STROKE token 没有意义）是对的，对 52 个序数 bin 和用药开关是错的——它们编码的是**当前状态**，
   必须能重复。按上游的规则，`MMSE_normal → MMSE_borderline` 之后再也回不去，"恢复"在结构上
   不可采样，panel a 的 Normal 行会恒等于 0。engine 改成显式的 `repeatable_tokens`。
3. **上下文窗口**：上游把整条增长的序列喂回去，自己的 block_size assert 还被注释掉了
   （`model.py:214`）。我们的 attention mask buffer 按 block_size=96 建，前缀 ~20 + 采样 96 会
   越界。engine 改成滑动 block_size 窗口——**这在这里成立、在 GPT 上不成立**：这个模型没有位置
   编码（`model.py:165`，`wpe` 是注释掉的），位置只通过 age embedding 进入，所以移窗不等于移表示。
   no-repeat 的记账仍然读**完整**历史，不是窗口。

---

## 2. 映射：哪些是等价的，哪些是替换

这是**第二次移植**——上游先从 NACC 搬到它自己的 RADC 分词（见 `FIGURE2_upstream.md`），我们再从
那里搬到 ROSMAP。上游的替换仍然成立的就沿用并标注出处；我们的分词不同的地方，写在这里而不是
悄悄吸收掉。

| 上游 RADC | 我们 ROSMAP | 是否等价 |
|---|---|---|
| MMSE 折叠成 4 个 Folstein 期 | MMSE 折叠成 **3 期**：Normal(≥27) / Borderline([24,27)) / Impaired(<24) | **更粗**，见下 |
| Death，吸收态，竞争风险 | 同 | 等价 |
| panel b 的 "Dementia" endpoint | `AD_DX` token | 等价（两边都是真实的发病诊断） |
| "Reach ≥Mild (MMSE ≤26)" | "Reach ≥Borderline (MMSE <27)" | 角色等价 |
| 3 个序数量表 + 4 个既往史 + 2 个分级 onset + 4 个用药 | 3 个序数量表 + 7 个一次性事件 + 8 个用药 | 角色等价 |
| 两个**分级** onset 家族（`Stroke, probable/possible`，会反复） | 无。我们每个病一个一次性 onset token | 我们更简单；`vocab.ONSET_IDS` 为空 |
| `filter_cohort` 复现 train.py 的规则 | 我们的 `train.py` **没有** cohort filter | 见第 3.9 节 |
| — | **34 个死后病理 token** | 上游没有对应物，见第 3.5 节 |

**分期少一级，这是 `.bin` 的性质，不是这里的选择。** 上游有 7 个 MMSE level，能压到 4 个 Folstein
切点上；`../tokenization/spec.py` 只发射 **3 个**（切点 24 和 27），所以 Folstein 的
Moderate 和 Severe 在这里**分不开**，它们是同一个 `MMSE_impaired`。panel a 因此说不了"重度损害"
这件事。想要回来得重新分词，加一个 18 的切点。我把它命名为 `Impaired` 而不是 `Moderate`，就是
为了不让人拿它当上游的 18–23 行读。

**切点闭区间方向和上游相反。** `spec.py` 用 `searchsorted(edges, v, side="right")`，每个箱
**左闭右开**：`borderline` 是 `24 ≤ x < 27`，不是"24–26 闭区间"。所以 `vocab.DISPLAY` 把区间
直接写进了 token 的显示名，省得读者把上游的 Folstein 边界搬过来。

---

## 3. 必须跟数字一起读的限制

### 3.0 先读懂 panel a 的行是什么：三个 MMSE 行的方向不一样

每一行的 at-risk 是"基线分期 != 该行"，所以**行与行问的临床问题方向并不相同**：

| 行 | at-risk 构成 | 方向 |
|---|---|---|
| **MMSE Normal** | Borderline(54, 好转) + Impaired(26, 好转) | **纯好转 / 恢复** |
| MMSE Borderline | Normal(336, 恶化) + Impaired(26, 好转) | 混合（以恶化为主） |
| MMSE Impaired | Normal(336, 恶化) + Borderline(54, 恶化) | **纯恶化** |

`MMSE Normal` 是整张图里**唯一的纯好转终点**（`cogn_global ≥0` 同理，是该量表的最好箱）。
这两行也正是至今唯二没有校准好的行，见 3.1。

### 3.1 剩下两行塌陷的原因是**自回归漂移**，不是模型学不会

**这一节被推翻重写过。** 早先版本断言"模型几乎不预测恢复，因为全语料只有 371 次恢复事件"，
并据此把它归成训练数据稀疏。**那个结论是错的**，下面是证伪它的实测，以及现在的归因。

**模型是好的。** 按训练口径（`get_batch` + 传 `targets`）评估：

| | 值 | 参照 |
|---|---|---|
| `loss_ce` | **1.30**（train）/ 1.36（val） | 均匀分布 = ln(96) = **4.56** |
| 边际分布 vs 训练目标 | 中位 **1.01x**，范围 0.77–1.40x | 完美 = 1.00x |
| P(下一个 MMSE = normal)，后续转移位置 | **0.151** | 真值 **0.118** |

train ≈ val，既不欠拟合也不过拟合；`MMSE_normal` 在训练口径下**答得对**。

**问题在 rollout。** 同一个模型、同一批受试者，rollout 发出的 MMSE 里 normal 只占 **0.055**
（训练口径 0.151，真值 0.118）。而同一次测量里 rollout 发出的 `MMSE_impaired` 占 **0.875**，
真值只有 0.390 —— **生成出来的病史本身就过度恶化**，模型在这段离群历史上继续预测，只会更恶化。
这是自回归漂移，不是某一个可定位的 bug。

**逐条排除过的解释**（每条都有实测，别再重查）：

| 假设 | 判定 | 证据 |
|---|---|---|
| token id 差 ±1 | ❌ | 14 个序数家族只有 MMSE/COG 呈位移；解码后每个背景类别恰好一个 token |
| `MMSE_normal` 被排除出损失 | ❌ | 作为训练目标的频率 0.62x，与**所有** token 一致（注入的 No-event 占目标 37.9% 带来的均匀稀释） |
| 输出头权重退化 | ❌ | ‖w‖=1.999，高于 content token 中位 1.967，排 22/61；被压最狠的 CRP/IL6 反而 ‖w‖ 最大 |
| 评估前缀缺 no-event 标记 | ❌ | 补上后 0.009 → 0.044，真值 0.804 |
| statics 年龄放错（训练时被 jitter） | ❌ | as-is / jittered / 去掉 / 提前 20 年，四种前缀结果全在 0.006–0.012 |
| 恢复样本太少（371 次） | ❌ | "基线首个 MMSE"这道题有 3,137 个样本，模型同样只给 0.9% —— 但那是**生成口径**下的数字，见下 |
| 生成时缺 `mask_ties` | ❌ **修了反而更差** | 见下 |

**`mask_ties` 这条值得单独记，因为它看起来最像答案。** `model.py` 只在传 `targets` 时屏蔽同龄
token，而本分词一次访视的所有 token 同龄，所以训练时每个位置都看不见自己的访视兄弟，生成时却
看得见 —— 在**真实病史**上测，这一项确实值 7.3x（训练口径 0.151 vs 生成口径 0.016）。
`engine._logits(mask_ties=True)` 把它精确复现了（喂 `targets_age` 即可，掩码不依赖 targets 内容）。
**但在 rollout 里打开它，normal 从 0.055 掉到 0.034**，因为 rollout 的上下文是模型自己生成的
离群病史，在上面再屏蔽信息只会更糟。开关保留、默认 False，就是为了让这句话可复现。

**结论**：这两行的绝对概率**不可用**，而且**在评估侧修不好**。要修得换预测方式——直接判别模型
（用 baseline 特征训分类器，不做 rollout），或者只把 rollout 用于排序。上游
`delphi-fine-tuning` 的 README 记过同类结论：它们的模型在主要终点上输给同期认知的 logistic
回归 0.02 AUC，并直言 "tabular methods may be more appropriate for this dataset"。

### 3.2 时间模型按"访视"训练、生成按"token"采样 —— **已修**（visit_sizes）

**根因。** `model.py` 用 `dt` 训练总速率，而 `mask_ties=True` 下

```python
dt = torch.clamp(targets_age - age, min=1.0)
dt = torch.gather(dt, -1, ...)   # "Use time from last untied token"
```

`dt` 被换成**到上一个「非同龄」token 的时间**，也就是**访视间隔**。所以模型学到的是
"下一个 token 大约在一个访视间隔之后到"，实测 `exp(-logsumexp(logits))` 中位 **338 天**
≈ 观测访视间隔中位 365 天。而逐 token 采样**一次等待只发一个 token**，真实的非基线访视却
携带 **3.22** 个临床 token（`make_visit_sizes.py`，21,070 次非基线访视；基线访视均值 9.80，
已排除——基线一次做全套，混进去会高估）。

**修法。** `engine.simulate(use_visit_sizes=True)`（现为默认）：采一个等待发出第一个 token 后，
从经验分布抽 k，在**同一个 age** 上补发 k−1 个。补发复用**同一次 forward 的 logits** —— 这不是
近似：mask_ties 下同访视 token 互不 attend，按模型自己的因子分解它们条件独立。

三条数据硬不变量在生成侧都执行了（实测 0 违规）：全局不重复（`vocab.REPEATABLE_TOKENS` 为空，
因为 `build.py:204` 的 `seen` 去重保证每 token 每人最多一次）、一次访视内同家族最多一个 token、
No-event/Death 不开访视。

**效果**（RIS，val，416 人）：

| | 改前 | 改后 |
|---|---|---|
| 临床 token / 年 | 0.39（观测 2.66 的 0.15x） | **1.27（0.48x）** |
| Δt == 0 的比例 | 0.3% | **67.9%**（观测 84.7%） |
| 时间 MAE | 5.01 年 | **3.71 年** |
| 时间 bias | **+3.58 年** | **−0.19 年** |
| 时间 R² | −0.41 | **+0.11** |
| 转移矩阵 r | 0.923 | **0.957** |
| mean calib error | −0.192 | **−0.102** |

**每一行拿到的提升几乎相同（约 3 倍 = 访视大小）**，所以本来只差 3 倍的 A 组校准了，
本来差 100 倍的 B 组仍然差（见 3.1）。median AUC 0.706 → 0.698，基本不变 —— 符合预期，
这是保序改动。

**剩下的 2 倍没修**：No event 占了 63% 的"步"，那些步不开访视。封掉它没用（总速率等比例
下降、等待变长，单位时间的临床事件率不变，实测印证：P(AD) 反而从 0.250 降到 0.223）。
要动它得在训练侧改 `dt` 的定义或注入率。

**实现时踩的三个坑**（都在 `engine.simulate` 的注释里，改代码前先读）：
`cur_age` 不能从 `age[:,-1]` 读（补发会写 PAD_AGE）；`max_new_tokens` 必须数"步"不是"列"；
**k 必须整批共用**——逐轨迹抽 k 会给未激活的行填 padding，而 padding 位置在 `model.py` 的
attn_mask 下**只能 attend 自己**，那一行下一步的 logits 就成了瞎猜（P(死亡 ever) 从 0.90 塌到 0.53）。

### 3.3 没有真正的 held-out split

`../tokenization/build.py` 只写 `train.bin` 和 `val.bin`，而 `../delphi/train.py` 就是**在这个 val 上
选的 checkpoint**（`always_save_checkpoint = False`，ckpt.pt 是 best-val 那一步）。所以这里每一个数字
都是在**参与过模型选择**的数据上测的。上游打的是真正的第三个 split，它的数字在这一点上比我们干净。

偏差不大但**不是零**：选择只碰了 checkpoint step、没碰权重，所以影响是"24 个评估点里挑最好那个"的
乐观度。**把这里的每个 AUC 当上界读。** 要根治得在 `build.py` 里切三份然后重训。

### 3.4 No event 保留为可采样（和上游相反）

上游封掉 No event，理由是它是合成 marker 不是临床事件。对我们**不成立**：
`config/train_delphi_rosmap.py` 设了 `no_event_token_rate=5`，这个 checkpoint 就是带着每 5 年一个
注入 marker 训出来的，它学到了一个速率。封掉一个 token 等于把它的速率从指数竞争里拿走，会让所有
真实事件提前——即用 timing 偏差去换掉一些图里根本不看的 token。两种设定都量过（40 人 × 50 条）：

| | P(AD ever) | P(death ever) | 15 年前被截断的轨迹 |
|---|---|---|---|
| 保留 No event（当前） | 0.250 | 0.882 | 1.3% |
| 封掉 No event（上游） | 0.223 | 0.816 | 3.8% |

### 3.5 死后病理 token 被禁止采样

34 个尸检 token 在数据里只出现在 Death 同一天或之后（`../tokenization/README.md`：放在死亡之前是
泄漏）。Death 会终止 rollout，所以采样出来的病理 token 只可能出现在**死亡之前**——训练数据里不存在
的位置。不封的话，一个活着的 80 岁会被判一个 Braak 分期。

封掉不是免费的（同样是从指数竞争里拿走速率）。代价量过：baseline 前缀上被移走的速率份额
**中位 0.90%，最大 2.6%**。

### 3.6 panel d：RIS 上是真 UMAP，本地是 PCA

RIS 的 conda 环境 `ad-projection` 里有 `umap-learn`，所以 `results/`（权威结果）的 panel d 用的是
**真 UMAP**。本地 Mac 没装，`results_local_archive/` 里那套是上游的 PCA 回退（标题、坐标轴、
CSV 列名 `pca1_base` 和 metrics.json 里都如实标注了）。两者的 headline 数字相同，因为
10-NN purity 算在原始 96 维空间里，不用投影。

### 3.7 配色的验证结论不能照搬上游

上游的色带是按它 10 个离散 outcome 手写的字面量，我们有 15 个（7 事件 + 8 用药），照搬会触发它
自己的 `_cycle_free` 报错——那个报错是设计好的。我改成按 outcome 数量生成色带。**上游 docstring 里
那些 ΔE 数字是对 10 个 outcome 验证的，对 15 个不成立**，家族内的区分必然更紧。可以接受是因为这些
面板里颜色只承担"属于哪个家族"，每行都在坐标轴上具名、每个面板都写 source-data CSV——但别拿上游的
验证结论给这张图背书。

### 3.8 AD_DX 的时间被量化到整年

`../tokenization/README.md`：`AD_DX` 被吸附到年度网格，中位位移 26 天、最大 182 天，且 ROSMAP 只有
单一的 `age_first_ad_dx`，没有逐访诊断。**panel b 里关于 AD 的 timing 结论精度上限就是 1 年。**

### 3.9 "matched" 和 "all" 在这里是同一批人

上游 `train.py` 训练前会 `filter_cohort`，所以它的 Figure 2 需要"matched"来把评估限制到同一批人。
我们的 `../delphi/train.py` 是上游 Delphi，**没有这个 filter**，它在 train.bin 上的每个人身上都训。
filter 规则照样实现了（签名一致），它在 val 上会丢 25/443 人，但那 25 人 `_baseline()` 早就丢掉了
（同样的理由：没有可预测的第二次访视），所以**进到 frame 里的人一个都没少**：
`metrics_matched.json` 的 `_meta.n_dropped` 是 **0**。两个版本照样都出，好让图永远说得清它描述的是谁。
哪天训练加了 filter，这里会自动接上。

### 3.10 语料和模型都很小

4,428 人 / 15.8 万条记录（val 443 人），模型 0.69M 参数（6 层 / 96 维），对照 UKB Delphi-2M 的
~40 万人。`best_val_loss = 9.82`。所有 CI 都宽，n<30 的行按上游规则已经不报。

---

## 4. 跑出来的结果

**全部来自 RIS**（compute2，`results/`）。`split=val`，416 个可评估病人，每人 100 条 Monte-Carlo
轨迹，起点 = 首次访视，主 horizon 5 年，`use_visit_sizes=True`。
`results_local_archive/` 是本地旧版（采样器不同），**不要引用**。

### a1 — 5 年内到达各状态的判别力（中位 AUC 0.698）

按**图里的行序**（状态槽位，自上而下）排；图中括号里的 `n=` 是 events 数，不是 at-risk 数。

| 行 | at risk | events | AUC | 95% CI | 实测 | 预测 |
|---|---|---|---|---|---|---|
| Global cognition ≥0 | 168 | 51 | 0.529 | [0.424, 0.639] | 0.313 | 0.025 |
| Global cognition −1.0..0 | 272 | 77 | 0.572 | [0.490, 0.655] | 0.301 | 0.243 |
| Global cognition <−1.0 | 392 | 39 | 0.841 | [0.771, 0.903] | 0.108 | 0.136 |
| AD diagnosis | 416 | 50 | 0.826 | [0.763, 0.886] | 0.129 | 0.150 |
| Death | 416 | 80 | 0.730 | [0.665, 0.792] | 0.212 | 0.157 |
| Impaired | 390 | 54 | 0.802 | [0.737, 0.863] | 0.147 | 0.139 |
| Borderline | 362 | 97 | 0.620 | [0.555, 0.684] | 0.286 | 0.331 |
| Normal | 80 | 40 | 0.666 | [0.550, 0.775] | 0.516 | 0.013 |

### a2 — 定标：A 组已校准，B 组仍塌

mean(predicted − observed) = **-0.102**（改 visit_sizes 前是 −0.192）。
六行落在 **0.8–1.4×**；只剩 `MMSE Normal`（39×）和 `cogn_global ≥0`（13×）—— 原因见 3.1，
**在评估侧修不好**。

### a3 / b-left — 绝对时间现在成立了

| | 改前 | 改后 |
|---|---|---|
| R² | −0.41 | **+0.109** |
| MAE | 5.01 年 | **3.71 年** |
| bias | +3.58 年 | **-0.19 年** |
| Spearman | 0.419 | 0.324 |

系统性推迟三年半的偏差消失了。**早先 README 写的"绝对时间完全不成立"已作废。**

### b — 按 horizon 的判别力

| 终点 | 1y | 2y | 3y | 5y | 10y |
|---|---|---|---|---|---|
| Reach ≥Borderline (MMSE <27) | 0.730 | 0.683 | 0.689 | 0.689 | 0.655 |
| AD diagnosis | 0.851 | 0.857 | 0.855 | 0.826 | 0.723 |
| Death | 0.740 | 0.714 | 0.743 | 0.730 | 0.763 |

### c — 轨迹：三个统计量都赢过"什么都不变"

| | model | carry-baseline |
|---|---|---|
| **Brier**（严格恰当） | **0.5277** [0.5012, 0.5557] | 1.1118 |
| P(truth) | 0.4736 | 0.4441 |
| modal Jaccard | 0.4975 | 0.3845 |

per-state IoU：Normal 0.56 / Borderline 0.10 / Impaired 0.12 / Death 0.55（macro 0.334）。

### d — embedding 编码的是"现在"，不是"将来"（RIS 上是真 UMAP）

轨迹类别 purity 0.316（chance 0.243）；
baseline 分期 purity **0.792**（chance 0.673）；
限制在同一 baseline 分期内 lift **+0.056**（上游 +0.051，NACC +0.050，三次独立测量落在同一位置）。

### 补充 — 一步转移矩阵

观测 vs 预测 Pearson r = **0.957**，MAE 0.072。

### 全 token 校准（`predict_vs_observe`，无 at-risk 筛选，53 个 token）

低估倍数中位 **1.39×**（改前 4.3×），38 个低估 / **15 个高估**。按家族：
Events 0.8× · Medications 0.9× · Death 1.4× · Clinical measures 2.0×。
仍然离群的只有 5 个：`MMSE ≥27` 39× · `CRP_q2` 28× · `IL6_q2` 16× · `cogn ≥0` 13× · `TNFA_q1` 5.9×。

### 报告口径扫描（`--sweep`）

| 窗口 | 临床 token 实/预 | 每 token 低估 | 组成口径 |
|---|---|---|---|
| 1 年 | 2.81 / 1.37 | 2.6× | 1.20× |
| 2 年 | 4.68 / 2.65 | 1.9× | 0.99× |
| 5 年 | 8.51 / 5.89 | **1.4×** | 0.85× |
| 10 年 | 13.26 / 10.02 | 1.3× | 0.78× |
| 下一次访视 | 3.34 / 1.67 | 2.2× | 1.09× |

短窗口低估更大（饱和效应），所以**换报告窗口消不掉低估**——这条（早先的 A3）已被证伪。
"组成口径"（问 token 份额而非是否出现）把速率因子约掉，全部落在 0.78–1.20×。

### 一句话

**判别力和绝对时间现在都站得住；定标除两行外也站得住。**
AD 5 年 AUC 0.83、时间 bias -0.19 年、转移矩阵 r 0.96、Brier 领先参照
53%。**不可用的是 `MMSE Normal` 与 `cogn_global ≥0` 的绝对概率**（3.1）。

## 5. 下一步（按性价比排序）

1. **切一个真正的 test split 并重训**（3.3）。现在 val 已用于选 checkpoint，所有 AUC 都是上界，
   而且没有独立数据可供重标定。改 `build.py` 三分 + 重跑训练。
2. **换掉 rollout，做直接判别模型**（3.1）。`MMSE Normal` / `cogn_global ≥0` 的绝对概率在评估侧
   修不好——自回归漂移是逐 token 生成的固有失败模式。用 baseline 特征直接训分类器做对照，
   既能给这两行一个可用的数字，也能回答"自回归到底买到了什么"。上游 `delphi-fine-tuning`
   的 README 记过同类结论：其模型在主要终点上输给同期认知的 logistic 回归 0.02 AUC。
3. **训练侧动 `dt` 的定义**（3.2 未修的那 2 倍）。把生成过程显式拆成"何时来访 / 来了发什么"，
   这是 `visit_sizes` 的训练侧版本，做对了它就不再是补丁。
4. ~~**去掉 `build.py:204` 的全局去重**~~ —— **已做，见第 6 节。** 认知量表（MMSE +
   `cogn_global`）两级去重全部豁免、重训、重评完了。结论是**一半修好一半打坏**：
   `MMSE Normal` / `cogn ≥0` 这两行的绝对概率从 129×/22× 低估收到 1.5×/2.0×（即第 3.1 节
   "在评估侧修不好"的那两行，在**分词侧**确实修得动），但 `AD diagnosis` 的 rollout 判别力
   从 0.836 塌到 0.535。
5. MMSE 加一个 18 的切点重新分词，把 Impaired 拆成 Moderate/Severe，panel a 就能和上游逐行对读。
6. 跑 `figure2/per_state_auc.py` 和 `fig_perdomain.py`——perdomain 的 18 个 outcome 已经在 cache 里
   （同一批轨迹，零额外 MC 成本），只差画出来。


---

## 6. 消融（第一档）：认知量表完全不去重（2026-09-18）

> **第 7 节把这条线推到了尽头**（全部 15 个纵向量都不去重），并给出三档并排的结论。
> 本节的两档对照仍然成立，但"不去重更好/更坏"这句话必须带口径 —— 见 7.4。

**一句话**：第 3.1 节那两行"在评估侧修不好"的塌陷，**在分词侧修得动**；代价是 AD 诊断的
rollout 判别力塌掉，而且塌的是 **rollout**，不是模型。

### 6.1 做了什么

`../tokenization/build.py --nodedup MMSE,COGN`：MMSE 与 `cogn_global` 改成**每次随访发射一个
token**，run-length 去重和全局 first-occurrence 去重都豁免（其余 13 个连续量照旧两级去重）。
词表一个字没改（129 token，`ignore_tokens` 相同）。数据侧的后果见
`../tokenization/README.md` 的"消融分支"一节；最关键的一条是**往复回来了**：MMSE 2,794 次、
`cogn_global` 2,426 次"回到曾经离开过的箱"，交付版这两个数恒为 0。

配套改动（都是"规则跟着数据走"，不是开关）：

| 改了什么 | 为什么不能靠人记 |
|---|---|
| `radc_delphi/vocab.py`：`REPEATABLE_TOKENS` 从写死的 `()` 改成读 build 自己 `meta.json` 的 `repeatable_token_prefixes` | 忘了改不会报错，只会让不去重的模型在生成侧**永远发不出第二个 MMSE token**，panel a 的 MMSE 行整片塌回 0 —— 正好是这次要看的东西 |
| `radc_delphi/engine.py`：`visit_sizes.npy` 先在 `data_dir` 找，再退回仓库根 | 访视大小是**分词的**统计量（3.22 → 3.77）。用交付版的分布去补发会静默把访视采样调小，而第 3.2 节量到的每行提升几乎正好等于访视大小 |
| `make_visit_sizes.py --dataset`、`run_figure2.sh` 的 `DATA`/`ROSMAP_LABELS`、`ris/figure2.sbatch` 的 `DATASET`/`CKPT`/`FIG2_TAG` | 两份分词并存，默认到错的那份是静默的 |
| `compare_runs.py`（新） | 只做对齐排版，不重算任何统计量 |

### 6.2 对照是**同平台重训的**去重版，不是交付版 ckpt

交付版 `Delphi-ROSMAP/ckpt.pt` 是在 Mac/MPS 上训的（`best_val_loss = 9.82`）。同一份配置、
同一个 seed 在 RIS 的 H100 上重训得到 **9.04** —— 0.78 nats 的差，远超噪声。所以**不能**拿
交付版 ckpt 当这次的对照，否则平台差异会混进结论。三个 ckpt：

| ckpt | 数据 | 平台 | best_val | ckpt iter | block_size |
|---|---|---|---|---|---|
| `Delphi-ROSMAP/` （交付版，未动） | rosmap | Mac / MPS | 9.8166 | 5000 | 96 |
| `Delphi-ROSMAP-gpubase/` （本次对照） | rosmap | RIS / H100 | 9.0359 | 3250 | 96 |
| `Delphi-ROSMAP-nodedup/` | rosmap_nodedup | RIS / H100 | 8.4682 | 6250 | 144 |

**两份数据之间的 val loss 不可比**：不去重那份有 32.5% 的 token 是高度可预测的认知持续，
8.47 < 9.04 里没有"模型更好"的信息。

交付版 vs GPU 重训（同数据、同配方、只差平台）给出了**单次训练的噪声尺度**：panel a 八行
AUC 差 −0.033 ~ +0.049，没有一行超过自己的 CI 半宽；但转移时间 R² 从 +0.11 掉到 −0.09。
**所以下面 |ΔAUC| < 0.05 的行不要解读。**

### 6.3 panel a（5 年 horizon，matched cohort，val，n_mc=100）

去重 = `gpubase`，不去重 = `nodedup`。`*` = |ΔAUC| 超过两边 CI 半宽的较大者。

| 行 | at risk 去重/不去重 | events | AUC 去重 | AUC 不去重 | Δ | 实测 / 预测 去重 | 实测 / 预测 不去重 |
|---|---|---|---|---|---|---|---|
| cogn_global ≥0 | 168 / 168 | 51 | 0.570 | **0.698** | +0.128* | 0.313 / 0.014 (22×) | 0.312 / **0.159 (2.0×)** |
| cogn_global −1.0..0 | 272 / 274 | 77 | 0.599 | 0.501 | −0.098* | 0.301 / 0.221 | 0.298 / 0.276 |
| cogn_global <−1.0 | 392 / 394 | 39 | 0.842 | 0.602 | −0.239* | 0.108 / 0.062 | 0.107 / 0.086 |
| AD diagnosis | 416 / 418 | 50 | 0.836 | **0.535** | **−0.301*** | 0.129 / 0.100 | 0.128 / 0.144 |
| Death | 416 / 418 | 80 | 0.757 | 0.781 | +0.025 | 0.212 / 0.136 | 0.209 / 0.224 |
| MMSE Impaired | 390 / 392 | 54 | 0.821 | 0.808 | −0.013 | 0.147 / 0.106 | 0.146 / 0.189 |
| MMSE Borderline | 362 / 364 | 97 | 0.670 | 0.663 | −0.007 | 0.286 / 0.172 | 0.283 / 0.210 |
| MMSE Normal | 80 / 80 | 40 | 0.633 | **0.764** | +0.131* | 0.516 / 0.004 (129×) | 0.514 / **0.342 (1.5×)** |

| | 去重 | 不去重 |
|---|---|---|
| median AUC | 0.713 | 0.680 |
| mean(predicted − observed) | −0.150 | **−0.046** |
| 转移时间 R² / MAE / bias | −0.092 / 4.27y / +2.66y | **+0.275 / 2.87y / −1.16y** |

**可评估人数（进 frame）416 → 418，panel c 的 372 → 376。** `_baseline()` 要求"至少两次不同访视"，而每次随访都发认知 token
让几个人多出了可数的访视。差 4 个人，逐行 at-risk 差 0~2，不影响对读。

### 6.4 panel b

| 终点 | run | 1y | 2y | 3y | 5y | 10y |
|---|---|---|---|---|---|---|
| Reach ≥Borderline (MMSE <27) | 去重 | 0.745 | 0.676 | 0.692 | 0.703 | 0.682 |
| | 不去重 | **0.769** | **0.754** | **0.733** | 0.705 | 0.679 |
| AD diagnosis | 去重 | 0.827 | 0.876 | 0.865 | 0.836 | 0.727 |
| | 不去重 | 0.660 | 0.647 | 0.584 | 0.535 | **0.432** |
| Death | 去重 | 0.784 | 0.701 | 0.746 | 0.757 | 0.789 |
| | 不去重 | 0.723 | 0.758 | 0.761 | 0.781 | 0.783 |

timing error（MAE / bias，年）几乎每一格都变好，`Death` 尤其明显：

| 终点 / 分箱 | 0–2y | 2–5y | 5–10y | >10y |
|---|---|---|---|---|
| ≥Borderline MAE 去重→不去重 | 5.09→**2.30** | 3.50→**1.53** | 3.15→3.24 | 4.22→7.81 |
| AD MAE | 2.86→3.46 | 4.36→**2.69** | 4.00→**2.98** | 4.63→5.05 |
| Death MAE | 8.90→**5.88** | 6.57→**4.23** | 4.85→**2.93** | 3.73→**3.38** |

### 6.5 AD 的塌陷是 **rollout** 的，不是模型的 —— 这条决定下一步往哪走

同一对 ckpt 用**训练口径**（`../delphi/evaluate_auc_rosmap.py`，一次 forward + DeLong，
不做任何 rollout）评，AD 判别力基本没动，而认知 token 大幅变好：

| token | offset=0.1（≈提前1年） 去重→不去重 | offset=365.25（≈提前2年） 去重→不去重 |
|---|---|---|
| `AD_DX` | 0.874 → 0.836 | 0.726 → **0.740** |
| `Death` | 0.792 → 0.799 | 0.760 → 0.788 |
| `MMSE_normal` | 0.499 → **0.873** | 0.461 → **0.826** |
| `MMSE_borderline` | 0.689 → 0.747 | 0.616 → 0.680 |
| `MMSE_impaired` | 0.668 → 0.764 | 0.580 → 0.723 |
| `COGN_high` | 0.523 → **0.857** | 0.410 → **0.825** |
| `COGN_mid` | 0.506 → 0.634 | 0.465 → 0.600 |
| `COGN_low` | 0.689 → 0.712 | 0.667 → 0.742 |
| 全 61 token 中位 | 0.667 → 0.661 | 0.580 → 0.588 |

读法要小心：认知 token 的 AUC 涨得这么多，**一部分是问题变简单了**。不去重之后
"下一个 MMSE 是 normal 吗"这道题里，"上次访视就是 normal"是一个几乎免费的答案，而交付版把
这条信息从数据里删掉了（同一个 token 不能出现第二次）。所以这不是纯粹的"模型变强"。

**更正（2026-09-21）："基本不动"说过头了。** 换一个 CI 更窄的口径（`next_visit_auc.py`，
3,384 个访视对，按人 bootstrap）测同一对 ckpt，`AD_DX` 是 **0.843 → 0.764**，两个 CI
`[0.81,0.88]` / `[0.73,0.80]` **不重叠**。所以准确的说法是：**rollout 把一个 0.08 的退化放大
成了 0.30**，不是凭空造出来的。下面这句话的方向仍然对，量级要按 0.08 读。

`AD_DX` 那一行没有"问题变简单了"的问题（它是一次性 token，两份数据里出现次数完全相同，
n=129 对 129），而它在非 rollout 口径下只掉 0.04~0.08、在 rollout 里掉 0.30。也就是说：

> 模型照旧知道谁会得 AD；是**逐 token 的 Monte-Carlo 前向模拟**把这个信息丢掉了。

一个可检验的机制（**未验证，别当结论**）：不去重让每次访视必发 2 个认知 token，认知家族因此
占掉指数竞赛里很大一块速率，而 MMSE/COGN 是**可重复**的、`AD_DX` 是**一次性**的，两者在
`engine.simulate` 的竞争里结构上不对等。注意 AD 的**定标反而变好**了（预测 0.100→0.144，
实测 0.128），塌的只有**排序**——这形态像"AD 风险退化成了一个和个体几乎无关的常速率"。

### 6.6 所以这次消融该怎么用

- **要往复、要校准、要绝对时间** → 用不去重那份。它把第 3.1 节判过"在评估侧修不好"的两行
  修到了 1.5×/2.0×，mean calib error −0.150 → −0.046，转移时间 R² 从负数变 +0.275。
- **要 AD 的 rollout 排序** → 不能用。0.535 @5y、0.432 @10y 不可用。
- **下一步的性价比最高项变了**：不再是第 5 节第 4 条（已做），而是**第 5 节第 2 条**
  ——既然训练口径下 AD 信息还在、只有 rollout 丢，那"换掉 rollout、做直接判别模型"就同时
  解释并解决了这里的塌陷。第 5 节第 3 条（训练侧把"何时来访/来了发什么"拆开）是同一件事的
  上游版本：不去重把"一次访视发几个 token"的方差整体放大了，而采样器仍然是逐 token 的。

### 6.7 复现

```bash
# 1) 分词（本地，约 1 分钟）
cd ../tokenization
python build.py --out data_rosmap_nodedup --dataset rosmap_nodedup --nodedup MMSE,COGN
cp data_rosmap_nodedup/{train.bin,val.bin,labels.csv,meta.json} ../delphi/data/rosmap_nodedup/

# 2) 训练（RIS，H100，约 3 分钟）—— 两侧都重训，别拿交付版 ckpt 当对照
cd ../delphi
./ris/deploy.sh --data rosmap_nodedup
ssh compute2 "cd <BASE>/delphi && CFG=config/train_delphi_rosmap_nodedup.py sbatch -J rosmap-nodedup ris/train.sbatch"
ssh compute2 "cd <BASE>/delphi && CFG=config/train_delphi_rosmap.py sbatch -J rosmap-gpubase ris/train.sbatch --out_dir=Delphi-ROSMAP-gpubase"

# 3) AUC（RIS，CPU，约 1 分钟）
ssh compute2 "cd <BASE>/delphi && CKPT=Delphi-ROSMAP-nodedup/ckpt.pt DS=rosmap_nodedup sbatch ris/eval_auc.sbatch"
ssh compute2 "cd <BASE>/delphi && CKPT=Delphi-ROSMAP-gpubase/ckpt.pt DS=rosmap       sbatch ris/eval_auc.sbatch"

# 4) figure2（RIS，30 核，整个 job 去重 3.5 分钟 / 不去重 11 分钟）
cd ../figure2_eval && ./ris/deploy.sh
ssh compute2 "cd <BASE>/figure2_eval && DATASET=rosmap_nodedup CKPT=../delphi/Delphi-ROSMAP-nodedup/ckpt.pt FIG2_TAG=nodedup sbatch ris/figure2.sbatch"
ssh compute2 "cd <BASE>/figure2_eval && DATASET=rosmap        CKPT=../delphi/Delphi-ROSMAP-gpubase/ckpt.pt FIG2_TAG=gpubase sbatch ris/figure2.sbatch"

# 5) 对照表
python compare_runs.py results/figure2/gpubase results/figure2/nodedup --names dedup nodedup
```

不去重的 MC pass 比去重慢 5 倍（**9.5 min vs 1.9 min**，30 worker / 418 对 416 人）：轨迹更少提前死亡、
上下文窗口从 96 宽到 144。`ris/figure2.sbatch` 默认 3 小时够用，这里提交时给了 `--time=08:00:00`。


---

## 7. 消融（第二档）：**全部**不去重 —— "每次都看到完整的一次访视"（2026-09-21）

**一句话**：三档是**单调**的 —— 越不去重，"这个人下一步处在什么状态"越好预测，
"一次性事件什么时候发生"越差。这不是两个独立现象，是同一个**持续性先验**的两面。

### 7.1 三档是什么

全程在 RIS 上跑（含分词）。三份数据的人和 train/val 划分完全相同，可逐人对读。

| tag | 数据 | `build.py` 参数 | token 行数 | 逐访视测量占比 | 实测可重复 token | `block_size` |
|---|---|---|---|---|---|---|
| `gpubase` | `rosmap` | （无） | 158,103 | 9.5% | **0** | 96 |
| `nodedup` | `rosmap_nodedup` | `--nodedup MMSE,COGN` | 212,078 | 32.5% | **6** | 144 |
| `fullvisit` | `rosmap_fullvisit` | `--nodedup all --no-global-dedup` | **369,354** | **76.6%** | **43** | **320** |

`--no-global-dedup` 额外让**用药 ON/OFF 的第二次开关**回来了（2,008 个 (人,token) 组合）——
"停了又吃"在交付版里第二次 `ON` 被整条删掉。CRP/IL6/TNFA 在第三档里仍不可重复：那三个标记
ROSMAP 只测了一次。

> **已知偏差：第三档并不是字面意义的"全部不去重"。** 129 个 token 里只有 43 个在这份数据上
> 真的会重复；其余 84 个（性别 / 背景块 / `AD_DX` / 5 个 `*_ONSET` / CRP-IL6-TNFA / `Death` /
> 34 个病理）是**源数据的性质**——一个人只有一个值，关掉去重也变不出第二个。
> 例外是 **`STROKE` / `DEPRESSION`**：它们在源数据里是逐访视标志（411 / 474 人被置位多次），
> 但 `build.py` 的**发射逻辑**写死了只取首次，而两个 dedup 开关管不到那里。
> `--per-visit-events`（2026-09-22 加，默认 off）补上这一条，量级 **+0.47% 的 token**
> （371,088 vs 369,354），可重复 token 43 → 45。**下面所有 fullvisit 的数字都带着这个偏差**；
> 因为 0.47% 远小于单 seed 噪声（panel a 的 CI 半宽 ±0.03~0.06）而没有重跑。
> 详见 `../tokenization/README.md` 的"已知偏差"一节。这就是为什么可重复 token 必须**从 .bin 实测**（`meta.json` 的
`repeatable_tokens_postshift`）而不是按前缀声明：声明了 15 个家族只有 12 个真会重复，
而会重复的用药不在任何前缀里。

三个 ckpt 同配方、同 seed、同平台（H100），只有 `block_size` 随数据变（它不是超参，
是"能不能装下一条完整轨迹"的硬约束）。`best_val_loss` 9.04 / 8.47 / 7.86
**不可横向比较** —— token 组成完全不同，越不去重就有越多几乎免费的"持续"token。

`visit_sizes` 均值 3.22 / 3.77 / **8.34**（中位 10，max 17）：第三档的 rollout 每走一步
确实发出一次完整访视。代价是 MC pass 从 1.9 min → 9.5 min → **44.9 min**（30 worker，
419 人），峰值内存 51 GB（申请了 64 G，偏紧）。

### 7.2 panel a（5 年 horizon，matched，val，n_mc=100）

`*` = |ΔAUC| 相对第一列超过两者 CI 半宽的较大者。

| 行 | AUC dedup | AUC cog-nodedup | AUC all-nodedup | events |
|---|---|---|---|---|
| cogn_global ≥0 | 0.570 | 0.698 (+0.128)* | **0.734 (+0.165)*** | 51 |
| cogn_global −1.0..0 | 0.599 | 0.501 (−0.098)* | 0.502 (−0.097)* | 77 |
| cogn_global <−1.0 | **0.842** | 0.602 (−0.239)* | 0.620 (−0.222)* | 39 |
| AD diagnosis | **0.836** | 0.535 (−0.301)* | 0.542 (−0.294)* | 50 |
| Death | 0.757 | **0.781** (+0.025) | 0.712 (−0.044) | 80 |
| MMSE Impaired | **0.821** | 0.808 (−0.013) | 0.782 (−0.039) | 54 |
| MMSE Borderline | **0.670** | 0.663 (−0.007) | 0.600 (−0.070)* | 97 |
| MMSE Normal | 0.633 | **0.764** (+0.131)* | 0.693 (+0.060) | 40 |
| **median AUC** | **0.713** | 0.680 | 0.657 |  |

定标（实测 / 预测 的倍数，1.0x = 准）：

| 行 | 实测 | dedup | cog-nodedup | all-nodedup |
|---|---|---|---|---|
| cogn_global ≥0 | 0.313 | 0.014 (21.7x) | 0.159 (2.0x) | **0.254 (1.2x)** |
| MMSE Normal | 0.516 | 0.004 (125.8x) | **0.342 (1.5x)** | 0.320 (1.6x) |
| MMSE Impaired | 0.147 | 0.106 (1.4x) | **0.189 (0.8x)** | 0.376 (**0.4x**) |
| cogn_global <−1.0 | 0.108 | 0.062 (1.7x) | **0.086 (1.2x)** | 0.202 (0.5x) |
| Death | 0.212 | 0.136 (1.6x) | **0.224 (0.9x)** | 0.294 (0.7x) |
| mean(predicted − observed) | | −0.150 | −0.046 | **+0.022** |

**mean calibration error 最接近 0 的是第三档（+0.022），但那是对消出来的**：它把原来低估的
几行推到了**高估**（`MMSE Impaired` 2.6× 高估、`cogn <−1.0` 2× 高估）。所以"定标最好"这句话
只对**均值**成立，逐行读的话第二档更稳。这一行别只看总结数字。

转移时间：R² −0.092 → **+0.275** → 0.048；MAE 4.27 → **2.87** → 3.27 年；
bias +2.66 → **−1.16** → −1.91 年。**第二档最好**，第三档开始过度提前。

### 7.3 panel b

| 终点 | run | 1y | 2y | 3y | 5y | 10y |
|---|---|---|---|---|---|---|
| Reach ≥Borderline | dedup | 0.745 | 0.676 | 0.692 | **0.703** | **0.682** |
| | cog-nodedup | **0.769** | **0.754** | **0.733** | 0.705 | 0.679 |
| | all-nodedup | 0.669 | 0.709 | 0.682 | 0.638 | 0.515 |
| AD diagnosis | dedup | **0.827** | **0.876** | **0.865** | **0.836** | **0.727** |
| | cog-nodedup | 0.660 | 0.647 | 0.584 | 0.535 | 0.432 |
| | all-nodedup | 0.734 | 0.639 | 0.579 | 0.542 | 0.508 |
| Death | dedup | **0.784** | 0.701 | 0.746 | 0.757 | **0.789** |
| | cog-nodedup | 0.723 | **0.758** | **0.761** | **0.781** | 0.783 |
| | all-nodedup | 0.752 | 0.732 | 0.718 | 0.712 | 0.734 |

timing error（MAE 年）在**短窗口**上单调变好、在 >10 y 上单调变坏：

| 终点 / 分箱 | 0–2y | 2–5y | 5–10y | >10y |
|---|---|---|---|---|
| ≥Borderline | 5.09 → 2.30 → **1.72** | 3.50 → 1.53 → **1.24** | **3.15** → 3.24 → 3.83 | **4.22** → 7.81 → 10.31 |
| AD | **2.86** → 3.46 → 3.25 | 4.36 → 2.69 → **1.48** | 4.00 → 2.98 → **2.71** | **4.63** → 5.05 → 6.34 |
| Death | 8.90 → 5.88 → **4.82** | 6.57 → 4.23 → **3.15** | 4.85 → 2.93 → **2.40** | **3.73** → 3.38 → 5.78 |

短窗口 MAE 几乎腰斩（Death 0–2y 8.90 → 4.82 年），>10 y 则从"低报 2.6 年"变成"低报 10 年"。
方向一致：模型越倾向"下次和这次一样"，近期就越准、远期就越不敢发事件。

### 7.4 决定性的那张表：**下一次访视**的判别力（无 rollout）

`next_visit_auc.py`，3,384 个访视对（443 人），teacher-forced 单次 forward，
真值三档共用 `rosmap_nodedup` 的逐访视记录（所以观测集合完全相同）。CI 按**人** bootstrap。

| | 全部访视对 | | | 只看真实**变化**的访视对 | | |
|---|---|---|---|---|---|---|
| | dedup | cog-nodedup | all-nodedup | dedup | cog-nodedup | all-nodedup |
| MMSE macro AUC | 0.575 | 0.879 | **0.887** | **0.688** | 0.521 | 0.550 |
| MMSE top-1 | 0.158 | 0.827 | **0.830** | **0.474** | 0.344 | 0.373 |
| cogn macro AUC | 0.458 | 0.892 | **0.913** | **0.793** | 0.501 | 0.439 |
| cogn top-1 | 0.217 | 0.775 | **0.818** | **0.538** | 0.300 | 0.177 |
| `AD_DX` | **0.843** | 0.764 | 0.768 | — | — | — |
| `Death` | 0.787 | 0.763 | **0.792** | — | — | — |

**两张表方向相反，这是整次消融最值得记的一件事。**

全集上不去重碾压（cogn top-1 0.217 → 0.818）。但 **73% 的访视对真值是"没变"**，它赢在
能表达持续 —— 而交付版的分词**结构上**发不出重复 token，这道题它必输，那不是模型的能力差。

换到**变化子集**（变化在三份分词里都表达得出来，谁都不吃结构性亏），交付版看起来最好
（cogn macro 0.793 vs 0.501 vs 0.439），而不去重的两档在若干类上**低于随机**
（`COGN_mid` 0.272 / **0.138**，`MMSE_borderline` 0.405 / 0.407）。

> **⚠ 这段解读被第 8 节推翻了，别照抄。** 这个变化子集的分数里**没有排除当前箱**，而不去重的
> 模型把绝大部分概率质量放在"维持当前箱"上 —— 于是它在其余箱上的相对排序被机械地压扁，
> 不去重的低分里混进了这个纯机械效应。把当前箱排除掉再重整（第 8 节的"条件方向"），
> 结论**反过来**：交付版在 MMSE 上只有 0.525（≈随机），不去重是 0.741 / 0.689。
> 下面那段"交付版在'该往哪变'上更准"的因果叙述因此是错的，保留原文只为留痕。

**所以机制是清楚的**：不去重让模型学到一个极强的"下次和这次一样"先验。
panel a 那两行校准修好、短窗口 timing 腰斩，都是这个先验的红利；
`AD_DX` / `cogn <−1.0` 的排序变差、>10 y timing 崩掉，是同一个先验的账单。
第三档把两边都推得更极端。

### 7.5 用哪一档

| 你要的 | 用哪档 | 为什么 |
|---|---|---|
| AD 的 5 年风险排序 | **dedup** | 0.836 vs 0.535 / 0.542，没得比 |
| "这个人明年大概什么状态"（持续为主） | **all-nodedup** | next-visit cogn top-1 0.818 |
| 已知要变，判断**往哪变** | **dedup** | 变化子集 cogn macro 0.793，另两档 ≈ 随机或反向 |
| 逐行定标 + 绝对时间 + 往复 | **cog-nodedup** | 转移时间 R² +0.275、MAE 2.87y 三档最好；定标逐行最稳 |
| 一个总结数字最好看的 | 别这么选 | 第三档 mean calib +0.022 是高估和低估**对消**出来的 |

### 7.6 下一步

第 5 节的优先级再次变了。**第 2 条（换掉 rollout、做直接判别模型）现在是唯一值得先做的**，
理由比上次更强：7.4 说明"预测下一次访视"和"预测什么时候发生一次性事件"是**两个互相拉扯的
目标**，逐 token 自回归被迫用一套速率同时承担。拆成两个模型（状态转移用判别模型、事件用
生存模型）才能各拿各的最优，而不是在分词上来回找折中。

第 3 条（训练侧把"何时来访 / 来了发什么"拆开）是同一件事的上游版本，且第三档把它的必要性
量化了：`visit_sizes` 均值 8.34 意味着 `engine.simulate` 的"补发 k−1 个"现在承担了整个生成
过程的绝大部分，而它复用同一次 forward 的 logits —— mask_ties 下这在理论上成立，但 k 已经
大到让这个近似的任何偏差都被放大 8 倍。

### 7.7 复现

```bash
# 1) 分词（RIS，1.5 分钟）—— 原始表格已在 $AD_BASE/raw/，一次性上传，不随代码同步
cd tokenization && ris/deploy.sh --raw --submit --     --out data_rosmap_fullvisit --dataset rosmap_fullvisit --nodedup all --no-global-dedup

# 2) 训练（RIS GPU，8.5 分钟）。general-gpu 排不上队就用 -p general-interactive
cd ../delphi && ./ris/deploy.sh
ssh compute2 "cd <BASE>/delphi && CFG=config/train_delphi_rosmap_fullvisit.py     sbatch -J rosmap-fullvisit -p general-interactive ris/train.sbatch"

# 3) AUC（RIS CPU，11 秒）
ssh compute2 "cd <BASE>/delphi && CKPT=Delphi-ROSMAP-fullvisit/ckpt.pt DS=rosmap_fullvisit     sbatch ris/eval_auc.sbatch"

# 4) figure2（RIS，30 核，47 分钟）
cd ../figure2_eval && ./ris/deploy.sh
ssh compute2 "cd <BASE>/figure2_eval && DATASET=rosmap_fullvisit     CKPT=../delphi/Delphi-ROSMAP-fullvisit/ckpt.pt FIG2_TAG=fullvisit     sbatch --time=12:00:00 --cpus-per-task=32 ris/figure2.sbatch"

# 5) 下一次访视的判别力（RIS CPU，约 2 小时，瓶颈是按人 bootstrap）
ssh compute2 "cd <BASE>/figure2_eval && RUNS='dedup:../delphi/Delphi-ROSMAP-gpubase/ckpt.pt:rosmap     nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup     fullvisit:../delphi/Delphi-ROSMAP-fullvisit/ckpt.pt:rosmap_fullvisit'     JOB=nva NVA_TAG=three sbatch ris/figure2.sbatch"

# 6) 三方对照表
python compare_runs.py results/figure2/{gpubase,nodedup,fullvisit}     --names dedup cog-nodedup all-nodedup
```

RIS 上跑默认分词的输出与交付版 `.bin` **md5 完全一致**，三份 `labels.csv` md5 也相同 ——
把分词搬上集群没有引入任何漂移。


---

## 8. AUC 什么时候不可信，以及怎么判"学会了变化"还是"只会复读"（2026-09-22）

**问题**：不去重之后，一次访视的真值有 80%+ 是"和上次一样"。一个永远输出当前箱的复读机，
在 next-token / next-visit 的边际 AUC 上就能拿高分。那些表因此**不能**当作"学会了进展"的证据。

**这不是 AUC 坏了，是那个问题被持续性主导了。** `MMSE_normal` 的边际 AUC 很大一部分在测
"能不能读出这个人现在是 normal" —— 那是从上下文直接抄的。换一个复读机刷不高的问题就行。

`change_skill.py`（`JOB=chg sbatch ris/figure2.sbatch`）。口径与 `next_visit_auc.py` 完全
一致（同样的访视对、同样的真值），加两个**不会学习**的基线，都只在 **train** 上拟合：

* **基线 A 持续性** —— 永远预测当前箱，`P(stay)` 取 train 经验持续率。这就是纯复读机。
* **基线 B 一阶 Markov** —— train 上数出来的 3×3 转移矩阵，**没有任何协变量**。
  这是"只知道当前状态"的最优预测器，是**硬门槛**：打不过它 = 年龄 / APOE / 病史全没用上。

### 8.1 MMSE（2,780 个访视对 / 398 人，实测变化率 0.197）

| | CE | skill vs A | **skill vs B** | **AUC(会不会变)** | 预测变化率 |
|---|---|---|---|---|---|
| dedup | 3.827 | **−5.04** | **−6.27** | **0.362** | 0.899 (**4.6×** 实测) |
| cog-nodedup | 0.460 | +0.274 | **+0.126** | **0.867** | 0.231 (1.2×) |
| all-nodedup | **0.453** | **+0.285** | **+0.139** | **0.881** | 0.239 (1.2×) |
| 基线A 持续性 | 0.634 | 0 | −0.205 | 0.500 | 0.204 (1.0×) |
| 基线B Markov | 0.526 | +0.170 | 0 | 0.717 | 0.200 (1.0×) |

给定确实变了（n=549），**排除当前箱后**重整，问往哪变（随机 = 0.500）：

| | top-1 | skill vs B |
|---|---|---|
| dedup | **0.525**（≈随机） | −1.75 |
| cog-nodedup | **0.741** | **+0.055** |
| all-nodedup | 0.689 | −0.008 |
| 基线B Markov | 0.692 | 0 |

### 8.2 cogn_global（2,997 个访视对 / 402 人，实测变化率 0.165）

| | CE | skill vs A | **skill vs B** | **AUC(会不会变)** | 预测变化率 |
|---|---|---|---|---|---|
| dedup | 3.479 | −5.17 | −6.02 | **0.349** | 0.833 (**5.0×**) |
| cog-nodedup | 0.555 | +0.015 | **−0.121** | 0.707 | 0.339 (2.0×) |
| all-nodedup | 0.505 | +0.103 | **−0.020** | **0.737** | 0.294 (1.8×) |
| 基线A 持续性 | 0.564 | 0 | −0.138 | 0.500 | 0.172 |
| 基线B Markov | **0.495** | +0.121 | 0 | 0.617 | 0.171 |

给定确实变了（n=496）：dedup **0.619** / cog-nodedup **0.885** / all-nodedup **0.871** /
Markov 0.780（随机 0.500）。

### 8.3 结论

**1. 三个模型都不是复读机 —— 但交付版是它的镜像失败。**
`dedup` 预测变化率 0.833–0.899，实测只有 0.165–0.197，**高估 4.6–5.0 倍**，
`AUC(会不会变)` 是 **0.35–0.36，低于 0.5**（反着排）。原因是结构性的：它的分词**只在变化时
发射 token**，所以训练时它从没见过一次"维持"事件，压根没有"不变"这个概念。
它的 CE 3.5–3.8 对比基线 A 的 0.56–0.63 —— 差了一个数量级。

**2. 不去重的两档确实学到了变化，不只是复读。** `AUC(会不会变)` 0.71–0.88，显著高于
Markov 基线的 0.62–0.72；MMSE 上 skill vs B 是 **+0.13 / +0.14**，也就是用上了当前状态
之外的信息。这是"真的学到了"的直接证据。

**3. 但它们在 `cogn_global` 上打不过 Markov（skill −0.12 / −0.02）**，原因是**定标**而不是
判别：它们把变化率预测成实测的 1.8–2.0 倍。判别（AUC）赢、总损失输，就是过度预测变化的签名。

**4. 这条更正了第 7.4 节。** 早先我说"交付版在'该往哪变'上明显更准（cogn macro 0.793 vs
0.501）"。那个数**没有排除当前箱**，而不去重模型的概率质量几乎全压在当前箱上，其余箱的相对
排序被机械压扁。排除当前箱之后结论**反过来**：MMSE 方向 top-1 交付版 **0.525（≈随机）**
对不去重 0.741 / 0.689；cogn 0.619 对 0.885 / 0.871。**不去重的模型在"往哪变"上也更准。**

**5. 所以该怎么报这类模型。** 边际 AUC 只该和"当前状态可读性"一起读，不能单独当进展预测的
证据。要报进展能力，报这四个：`AUC(会不会变)`、对 Markov 的 skill score、
排除当前箱的条件方向准确率、P(变化) 的定标倍数。前三个复读机刷不高，第四个直接暴露它。

**读数警告**：基线 A 的"条件方向 top-1" 0.352 / 0.262 **低于随机**是 argmax 打平的产物
（A 给两个非当前箱完全相同的概率，`argmax` 总取小下标），不是真信号，别引用。


---

## 9. 两个"视野极窄"的基线，和它们推翻的东西（2026-09-22/23）

第 8 节只在"会不会变"上报了基线。这一节补两个更要命的对照，脚本
`window_baselines.py`（`JOB=wbl`）、`lr_endpoint_baseline.py`（`JOB=lrb`）、
`direct_head.py`（`JOB=dh`）。

### 9.1 边际 AUC 的**地板**：完全不转变的基线

| 边际 per-token macro AUC | MMSE | cogn_global |
|---|---|---|
| **N 完全不转变**（P(下一箱=当前箱)=1） | **0.733** | **0.822** |
| dedup | 0.575 | 0.458 |
| cog-nodedup | 0.879 | 0.892 |
| all-nodedup | 0.887 | **0.913** |
| **W 单访视窗口 logistic** | **0.889** | 0.908 |
| W+ 窗口 + statics | 0.889 | 0.908 |

第 7.4 节那张 0.88–0.91 的表，**地板是 0.73–0.82**，是"永远不变"白拿的。模型的真实增量
只有 0.09–0.15 个 macro AUC 点，不是从 0.5 起算。**以后引用边际 AUC 必须同时给 N。**

W 是只看**当次访视 token**（多热 + 年龄）的多项 logistic，看不到任何历史。三个模型
`skill vs W` 全为负（−0.05 ~ −0.21），MMSE 的条件方向 W 反而最高（0.765 对 0.741/0.689）。
**整条纵向历史没有带来可测的增量。** W+ 每一格与 W 完全相同 —— statics 也没有。

### 9.2 figure2 自己的终点上，逻辑回归胜过 rollout

LR 的特征 = `_baseline()` 定义的 prompt（statics + 整个基线访视）的多热 + 基线年龄，
**和 rollout 的起始前缀一个 token 不多不少**；at-risk / `labels_at_h` / `aalen_johansen` /
`boot_auc` 全部复用 figure2 自己的函数，只替换 `ep["risk"][h]`。

panel a 中位 AUC（5y）：**LR 0.789 / 0.792 / 0.793** 对 rollout **0.713 / 0.680 / 0.657**。
在 dedup 上 LR 赢 8 行里的 7 行，唯一输的是 `AD diagnosis`（0.790 对 0.836，而且模型的点估计
落在 LR 的 CI `[0.73,0.85]` 里）。panel b 上 LR 在 `≥Borderline` 和 `Death` 的每个 horizon
都赢。**LR 的定标几乎不用修**（预测 vs 实测 CIF 基本在 10% 以内）。

### 9.3 塌陷在**输出机制**，不在表征 —— 直接判别头拿回大部分

冻结 encoder，用基线前缀最后一位的隐状态（96 维，`F2._hidden(...)[-1]`，就是 rollout 的条件
向量）接 logistic：

| panel a 中位 AUC（5y） | raw（=9.2 的 LR） | **emb（直接头）** | raw+emb | rollout |
|---|---|---|---|---|
| dedup | 0.789 | 0.788 | 0.788 | 0.713 |
| cog-nodedup | 0.792 | 0.777 | 0.773 | 0.680 |
| all-nodedup | 0.793 | 0.787 | 0.772 | 0.657 |

`AD diagnosis` 5y：raw 0.790/0.794/0.795，**emb 0.808/0.818/0.800**，rollout 0.836/0.535/0.542。
也就是说**对两个不去重的模型，直接头把 AD 从 0.535/0.542 救回到 0.818/0.800**；
`emb` 是唯一在三档上都稳定跑赢 `raw` 的地方（AD 与 panel b 的 `≥Borderline`/`Death`）。

`raw+emb` 三档都没有更好 -> 两者信息高度冗余，**"LR 主干 + transformer 残差"没有直接证据**。

### 9.4 这改变了结论

* **"不去重把 figure2 打坏了"是 rollout 放大的假象。** 换成直接头之后
  不去重 0.787 ≈ 去重 0.788；而 rollout 下是 0.657 对 0.713。
* **做扰动引擎只能用不去重的分词。** dedup 预测变化率是实测的 4.6–5.0 倍、
  `AUC(会不会变)` 0.35（反着排）—— 一个永远模拟不出"保持不变"的引擎，反事实轨迹无意义。
* 所以**扰动需要的分词**和**方案 2 需要的输出机制**指向同一个配置：
  **不去重分词 + 直接判别头 + 保留 LM 头**。判别 ≈ LR，AD 0.818 与 dedup rollout 的 0.836
  在 CI 内，同时保住生成/干预能力。
* transformer 在**判别**上不比一个基线前缀的 LR 强。它的正当性得落在 LR 做不到的事上：
  反事实模拟。报判别数字时**必须带 LR 那一列**。

### 9.5 端到端（未做）

探针是**冻结** encoder 的可行性判断，不是方案 2 本身。端到端要联合损失，否则 LM 头退化、
扰动引擎就没了：

```
loss = loss_ce + loss_dt + λ · Σ_{endpoint,h} BCE(head(h_t), y_censor_aware)
```

三个必须做对的：(1) 标签用 `labels_at_h` 的删失感知版本，`y=-1` 丢弃，用朴素"5 年内有没有"
会系统性低估；(2) head 接在**每个访视位置**而不是只接基线，监督量从每人 1 个点变成 ~7 个，
CI 按人算；(3) 先冻结 encoder 训 head，再小学习率解冻。
过拟合风险实打实：三个 ckpt 的 best_val 分别在 iter 3250 / 6250 / 7250，3,985 人训 0.69M 已吃紧。


---

## 10. 噪声地板，以及它推翻的东西（2026-09-23）

**先读这一节再读第 7–9 节的任何逐行差异。** 这里给的是"同配方换 seed 重训"的变异幅度；
小于它的差异**不可解读**，而第 7–9 节里有若干条差异正落在这个幅度以内。

### 10.1 怎么测的

`config/train_delphi_rosmap_nodedup.py` 与 `..._nodedup_pos.py` 各训 3 个 seed（42/43/44），
其余一字不改，全部 RIS/H100。然后同一套 `wbl` / `dh` 对照跑 6 个 ckpt。
**组内 std 就是单次训练的噪声尺度。**

| 指标 | nodedup（3 seed） | nodedup-pos（3 seed） | 噪声尺度 |
|---|---|---|---|
| MMSE `AUC(会不会变)` | 0.866 ± 0.009 | 0.877 ± 0.002 | ±0.01 |
| MMSE 条件方向 top-1 | 0.736 ± **0.040** | 0.739 ± 0.014 | **±0.04** |
| MMSE `skill vs W` | −0.062 ± **0.051** | −0.050 ± 0.026 | **±0.05** |
| MMSE 预测变化率 | 0.228 ± 0.020 | 0.185 ± 0.010 | ±0.02 |
| cogn `skill vs W` | −0.216 ± **0.125** | −0.141 ± 0.085 | **±0.13** |
| 直接头 per-row AUC | ±0.001 ~ ±0.024 | ±0.004 ~ ±0.020 | **±0.02** |
| 直接头 8 行均值 | 0.787 ± 0.008 | 0.798 ± 0.006 | ±0.01 |

### 10.2 位置编码（`pos_embedding`）：两条站得住，其余在噪声里

**站得住**：

1. **MMSE 变化率定标**。0.228 [0.208,0.247] → **0.185 [0.177,0.197]**，实测 0.197。
   两组范围**完全不重叠**，且 pos 组落在实测值上。这是唯一一个干净分离的判据指标。
2. **LM val loss**。nodedup 8.468/8.476/8.502 对 pos 8.445/8.427/8.465 —— **三对三不重叠**。
3. 直接头里两行清晰分离：`MMSE Impaired` 0.856→0.869、`cogn −1.0..0` 0.746→0.766。

**在噪声里（不可解读）**：`skill vs W`（−0.062→−0.050，std ±0.05）、`AUC(会不会变)`、
条件方向 top-1、AD 5y（0.804±0.023 → 0.812±0.019）。

### 10.3 被推翻的单 seed 读数

第 9 节之后我用**单 seed**报过两组数，现在都不成立：

| 我报过的 | 实际 | 判定 |
|---|---|---|
| "skill vs W −0.069 → −0.020，几乎补平" | 组内 std ±0.05，两组重叠 | ❌ **噪声** |
| "AD 直接头 0.818 → 0.834，追平 dedup rollout" | 0.804±0.023 → 0.812±0.019，重叠 | ❌ **噪声** |
| "直接头 8 行均值 0.789 → 0.803，表征首次超过 raw" | 0.787±0.008 → 0.798±0.006，与 raw 0.789 勉强分离 | ⚠️ **减弱**（3/3 pos > raw，但幅度只有 1 个 std） |
| "MMSE 变化率 0.231 → 0.197 精确对上" | 0.228±0.020 → 0.185±0.010，不重叠 | ✅ **成立** |

### 10.4 核心问题没解决

`skill vs W` 在两个配置、六次训练里**全部为负**（MMSE −0.108 ~ −0.007，cogn −0.345 ~ −0.081）。
也就是说：**"整条纵向历史没有超过一个只看当次访视的 logistic"这一条，位置编码没有改变。**
根因假设（加性绝对年龄码使"最近一次的值"难算）**只得到了部分支持**——它改善了变化率定标，
但没有让历史产生可测的增量。

下一个该测的是 V2：在 attention logits 上加 `age_q − age_k` 的相对时间 bias。
"多久以前"比"第几个"更接近临床语义，而 V1 给的只是序号。

### 10.5 给后续所有实验的规则

**任何单 seed 的差异，小于下面这些数就不要写进结论：**
`skill vs W` ±0.05（MMSE）/ ±0.13（cogn）· 条件方向 top-1 ±0.04 ·
直接头 per-row AUC ±0.02 · 直接头均值 ±0.01 · 变化率 ±0.02。
figure2 panel a 的 AUC 另有一个更早的估计（换平台重训 ±0.05，见 6.2），量级一致。
