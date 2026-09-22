# ALADYNOULLI → ROSMAP

把 [A Bayesian framework for longitudinal EHR and genetic discovery](https://www.nature.com/articles/s41586-026-10780-5)
(Urbut et al., Nature 2026, 模型名 ALADYNOULLI) 按论文方程复现，跑在本项目的 4,428 人
ROSMAP 队列上，作为 `../delphi` 那个 transformer 的**独立对照**。

目的只有一个：在花力气把贝叶斯那套东西嫁接进 Delphi 之前，先回答

> **n = 4,428 到底能不能稳定地学出 signature？**

学不出来，后面所有嫁接工作都不用做。

---

## 0. 只在 RIS 上跑

**本仓库目录下不保留任何运行产物。** 没有 `tensors.npz`、没有 `results/`、没有 `__pycache__`。
全部计算和输出都在 RIS：

```
/scratch2/fs1/mdan/huang.yi1/aladynoulli-rosmap/
├── .venv/                 python 3.12 + torch 2.14(cpu) + numpy + scikit-learn + scipy
├── _tokenization/         从 ../tokenization/ 同步过来的分词数据
├── tensors.npz            data.py 在 RIS 上生成（已验证与本地生成逐元素一致）
├── results/               checkpoint / 原始打分 / AUC 表
└── logs/                  Slurm 的 .out / .err
```

调度器是 **Slurm**（compute2，没有 `bsub`），账户 `compute2-mdan`，分区 `general-cpu`。
和 `../delphi/` 在 RIS 上那套 (`delphi-rosmap/ris_auc.sbatch`) 是同一个约定。

### 同步 + 提交

```bash
# 代码（不带产物）
rsync -av --exclude __pycache__ --exclude 'results/' --exclude 'tensors.npz' \
      ./ compute2:/scratch2/fs1/mdan/huang.yi1/aladynoulli-rosmap/

# 分词数据（只需在数据变了以后做一次）
rsync -av --exclude __pycache__ ../tokenization/ \
      compute2:/scratch2/fs1/mdan/huang.yi1/aladynoulli-rosmap/_tokenization/

ssh compute2
cd /scratch2/fs1/mdan/huang.yi1/aladynoulli-rosmap
.venv/bin/python data.py          # 生成 tensors.npz
sbatch ris_cv.sbatch              # 60 个任务 = 6 个 K 值 x 10 折
```

跑完：

```bash
.venv/bin/python pool_cv.py   --k 8      # 10 折池化 -> n=4428 的 AUC 表
.venv/bin/python stability.py --k 8      # signature 跨折稳定性
.venv/bin/python interpret.py --k 8      # 组成 / APOE 载荷 / AEX vs 尸检病理
```

日志是缓冲的，任务跑完才刷到 `logs/*.out`；要实时看加 `.venv/bin/python -u`。

---

## 1. 文件

| 文件 | 说明 |
|---|---|
| `data.py` | `.bin` 事件流 → ALADYNOULLI 的 N×D×T 生存张量。三处对论文的偏离都写在文件头 |
| `model.py` | 论文方程的 PyTorch 实现：GP 先验 + MAP |
| `fit.py` | 两阶段拟合。stage1 估全局参数，stage2 冻住全局只给 held-out 重估 λ |
| `evaluate.py` | landmark AUC + `pop` / `logit` 两个对照，存原始打分 |
| `pool_cv.py` | 把 10 折的 held-out 打分池化 |
| `stability.py` | ψ 的跨折匈牙利匹配 + 相关系数 + top-m 组成保留度 |
| `interpret.py` | signature 组成、Γ 的 APOE 载荷、AEX 对 Braak/CERAD/淀粉样等 |
| `ris_cv.sbatch` | Slurm 数组作业，K × 折 的完整网格 |

---

## 2. 数据转换：三处必须偏离论文的地方

规模：**N=4,428，D=52，T=46（60–105 岁），风险集 1,315,069 个格子**。

### (1) 左截断

UKB 的人从 30 岁起就有 EHR，论文直接令 `E_id = min(诊断, 删失) - 30`，所有人从 t=0
同时进入风险集。ROSMAP 是队列研究，**入组年龄中位 78.9 岁**（p1=60.2），入组前没有观测。
所以每个人的风险窗口从他自己的 `S_i = idx0+1` 开始。

不这么做，"80 岁入组的人在 65 岁没得 AD" 会被当成真实阴性证据，而那一段根本没观测过。

### (2) 基线患病 = 协变量，不是事件

`../tokenization/build.py` 对每个 subject 做了全局去重，所以"首次出现"对增量事件是对的
（`AD_DX` 只有 0% 落在基线），对**状态型 token 是错的**：

| token | 首次出现落在基线访视的比例 |
|---|---|
| `MMSE_normal` | 89% |
| `COGN_high` | 84% |
| `BMI_high` | 75% |
| `ANTIHYP_ON` | 68% |

那是入组时的横断面状态，不是新发。基线就带着 token *d* 的人从 *d* 的风险集里移出，
同时把 *d* 写进他的基线协变量。留在风险集里的才是真正的转变
（`MMSE_normal` 在基线**之后**首次出现 = 认知恢复正常，是个有意义的事件）。

### (3) 炎症标志物剔除

`CRP_*` / `IL6_*` / `TNFA_*` 共 9 个 token 只有 **142/4428 = 3.2%** 的人测过。
没测过的人在生存模型里长得和"一直没发生"一模一样，但那是 missing 不是 negative。
`--keep-inflammation` 可以留着。

### 另一处：μ_d 随年龄变

论文说 `μ_d` 是 "logit of population prevalence"，没说是否随年龄变。ROSMAP 跨 60–105 岁，
hazard 随年龄变化极大，取常数会逼着 φ 去承担全部年龄趋势，而 GP 先验会把它压平。
这里 `μ_d(t)` 取经验的年龄别 hazard 的 logit（5 年窗平滑）。`set_mu(mode="const")` 可切回。

---

## 3. 评估设计

### landmark

在若干绝对年龄 `t0`（默认 75/80/85/90）上：

| | |
|---|---|
| 风险集 | `atrisk=1` 且 `S_i <= t0` 且 `E_id > t0` |
| 标签 | `Yobs=1` 且 `t0 < E_id <= t0+H` |
| 剔除 | `Yobs=0` 且 `E_id < t0+H`（窗口结束前就删失，阴阳都判不了）|
| 打分 | `1 - prod_{t=t0+1}^{t0+H} (1 - π_idt)` |
| λ | 用 `horizon=t0` 重新拟合，只吃 t0 之前的信息 |

λ 那一条是论文的 prospective 协议（"fixed φ̄, refit only individual λ̂, using data available
up to each prediction timepoint"）。不这么做，held-out 的人要么没有 λ，要么用训练时的 λ = 泄漏。

### 两条必须看的规矩

**① AUC 分 landmark 报，不是池化报。**
单个 landmark 内所有人年龄完全相同，所以"只看年龄"的 `pop` 基线在这里的 AUC **恰好是 0.500**
（已在实跑中验证）。超过 0.500 的部分才是模型真正从个体身上学到的东西。
池化多个 landmark 会混入年龄差异，把所有方法的 AUC 一起抬上去——数字好看，但不回答
"signature 有没有用"。两个数都存，主结论看 per-landmark。

**② 用 10 折全队列，不是 443 人的 val split。**
每个人恰好被留出一次，AUC 的样本量从 443 变成 4,428。`../delphi` 原本的 train/val 划分
在 `tensors.npz` 的 `split` 里保留着，`evaluate.py` 不加 `--fold` 就走那条路，用于和
Delphi 的既有结果对齐。

### 两个对照

| | |
|---|---|
| `pop` | 只用训练集的经验年龄别 hazard `h_d(t)`，完全不看这个人是谁 |
| `logit` | 每个病一个逻辑回归，协变量 = 随访时长 + 性别 + APOE ε4 剂量 + ε2 + 教育 |

风险集、标签、剔除规则和 ALADYNOULLI 逐格一致。
**ALADYNOULLI 打不过这两个，就说明 signature 没加任何信息。**

---

## 4. 和论文的另外两点不同（不是偏离，是这个数据的优势/劣势）

- **做不了 GWAS。** 论文 n=400k 才找出 151 个位点，ROSMAP n=4,428 没有任何功效。
  所以 `Γ` 的用法从"发现新位点"改成"**APOE ε4 的效应主要加载在哪个 signature 上**"——
  这个在 4,428 上是可估的（`Γ` 只有 K×P 个参数）。
- **可以做论文做不了的病理验证。** 论文只能拿 GWAS 位点佐证 signature 是真的；ROSMAP 有
  **尸检金标准**（Braak / CERAD / 淀粉样 / Tau / TDP / Lewy / CAA / 梗死）。
  `interpret.py` 把 `AEX_ik = Σ_t θ_ikt`（论文的 lifetime signature exposure）对上这些。
  病理 token 在 `[DEATH]` 之后，全程排除在 D 之外，只当外部 outcome 用。
