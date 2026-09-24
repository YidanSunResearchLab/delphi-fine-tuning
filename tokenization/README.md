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
| `data_rosmap/test.bin` | **只有 `--split` 的 test 比例 > 0 时才写**；默认两分不产生这个文件 |
| `data_rosmap/labels.csv` | 词表，第 N+1 行对应 token N |
| `data_rosmap/meta.json` | token 映射、区间、排除变量、分层 gap 分布、去重豁免 |
| `data_rosmap_nodedup/` | **消融分支 A**：认知量表每次随访都发射，不去重。见下节 |
| `ris/build.sbatch` | 在 RIS 上重新分词，跑完自动装进 `../delphi/data/<dataset>/` |
| `ris/deploy.sh` | 推 tokenizer 代码；`--raw` 才上传原始表格（受限数据） |

**消融分支 B（全部不去重）只在 RIS 上**：`data/rosmap_fullvisit`，见下节。它是在集群上
建的（用户要求"全程在 RIS 上跑"），`.bin` 按 `ris/deploy.sh` 的约定不回传。
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
# 三分数据集还要多带一个 test.bin（`ris/build.sbatch` 会自动处理，手工 cp 容易漏）
```

## 消融分支：`--nodedup`（认知量表不去重）

交付版对每个纵向连续量施加**两级**去重，两级都在 `build.py` 里：

1. **run-length**（`build.py` 的 CONT 循环）：只在箱发生变化时发射；
2. **全局 first-occurrence**（`seen` 那段）：每个 token id 每人最多出现一次。

第 2 级比看起来更狠：一个在 26 分附近来回跳的人，真实轨迹是 `N-B-N-B`，`.bin` 里只剩 `N-B`,
**往复被结构性地删掉了**。`figure2_eval/README.md` 的"下一步"第 4 条就是这件事。

两个开关，可以单独用也可以叠加：

| 开关 | 关掉哪一级 | 作用范围 |
|---|---|---|
| `--nodedup MMSE,COGN` | 第 1 级（run-length） | 只对列出的前缀；同时让它们跳过第 2 级 |
| `--nodedup all` | 同上 | `spec.CONT` 的**全部 15 个**家族 |
| `--no-global-dedup` | 第 2 级（全局），**整体** | 所有 token。额外影响的是 CONT 之外还会往复的东西：**用药 ON/OFF 的第二次开关** |
| `--per-visit-events` | 不是去重，是**发射逻辑** | `STROKE` / `DEPRESSION` 每个被置位的访视都发。必须配 `--no-global-dedup`。默认 off，见下面的"已知偏差" |

`--nodedup` 把列出的前缀从**两级**去重里都豁免掉，改成每次随访发射一个 token：

```bash
# 分支 A：只放开认知量表
python build.py --out data_rosmap_nodedup --dataset rosmap_nodedup --nodedup MMSE,COGN
cp data_rosmap_nodedup/{train.bin,val.bin,labels.csv,meta.json} ../delphi/data/rosmap_nodedup/

# 分支 B：全部不去重（"每次都看到完整的一次访视"）。在 RIS 上跑，跑完自动装好
ris/deploy.sh --raw --submit -- --out data_rosmap_fullvisit --dataset rosmap_fullvisit \
                                 --nodedup all --no-global-dedup
```

两份分词的**词表完全相同**（129 个 token，`ignore_tokens` 也相同）——只改发射次数，不新增 token。
这正是它危险的地方：`figure2_eval/radc_delphi/engine.py:load` 只能校验 `vocab_size` 和
`ignore_tokens`，所以**用交付版 ckpt 去评不去重的数据不会崩**，只会安静地给出一张错图。
配对的方式是 `meta.json` 里的 `repeatable_token_prefixes`，下游据此决定 rollout 的 no-repeat 规则。

| | 交付版 `rosmap` | A: `--nodedup MMSE,COGN` | B: `--nodedup all --no-global-dedup` |
|---|---|---|---|
| token 行数 | 158,103 | **212,078**（1.34x） | **369,354**（2.34x） |
| 逐访视测量占比 | 9.5%（只有变化点） | 32.5%，人均 15.6 | **76.6%，人均 63.9** |
| 序列长度 中位 / p99 / max | 35 / 57 / 65 | 46 / 90 / 114 | **73 / 233 / 285** |
| 含 no-event 标记后的 max | 85 | 134 | **304** |
| → `block_size` | 96 | **144** | **320** |
| **实测**可重复 token 数 | **0** | **6**（MMSE+COGN） | **43**（35 个分箱 + 8 个用药开关） |
| MMSE / `cogn_global` 往复 | 0 / 0（结构上不可能） | 2,794 / 2,426 | 2,794 / 2,426 |
| SBP / DBP / GLU 往复 | 0 / 0 / 0 | 0 / 0 / 0 | **8,027 / 5,294 / 3,618** |
| 用药 ON/OFF 重复 | 0 | 0 | **2,008 个 (人,token) 组合** |
| 非基线访视的临床 token 数（均值） | 3.22 | 3.77 | 见 `make_visit_sizes.py` |
| train/val 划分 | 3,985 / 443 人 | **同一划分** | **同一划分**（RNG 抽取次数没变） |

人和划分三份都一样，所以可以逐人对读。

**CRP / IL6 / TNFA 在 B 里也不可重复**，往复恒为 0 —— 那三个生物标记在 ROSMAP 里只测了一次，
没有第二次可发射。这是为什么"可重复 token"必须**实测**而不是按前缀声明：`--nodedup all`
声明了 15 个家族，实际只有 12 个真的会重复。

`meta.json` 的 `repeatable_tokens_postshift` 就是这个实测集合，下游
`figure2_eval/radc_delphi/vocab.py` 优先读它、而不是读前缀。反过来也重要：
`--no-global-dedup` 让用药 ON/OFF 重复，而用药**不在任何 `--nodedup` 前缀里**，
按前缀推会漏掉，漏掉的后果是 rollout 永久封掉第二次 `STATIN_ON`，而数据里明明有。

### 已知偏差：`--nodedup all --no-global-dedup` 并不是字面意义的"全部不去重"

129 个 token 里只有 **43** 个在这份数据上真的会重复。剩下 86 个（含 2 个技术 token）里，
**84 个是源数据的性质**，关掉去重不可能让它们重复：

| 区块 | 个数 | 为什么每人只能一次 |
|---|---|---|
| 性别 | 2 | 常量 |
| 背景块 | 30 | 只在基线发射，值不变（教育/种族/APOE/吸烟/饮酒/5 个基线已患病） |
| `AD_DX` | 1 | 源数据只有 `age_first_ad_dx` 一个横截面值，**没有逐访诊断** |
| 5 个 `*_ONSET` | 5 | `*_cum` 是累积标志 0→1，"第二次 onset"在这个编码里不存在 |
| CRP / IL6 / TNFA 的 9 个分箱 | 9 | 这三个标记每人**只测了 1 次**（426 人，max = 1） |
| `Death` | 1 | 吸收态 |
| 死后病理 | 34 | 尸检，单次测量 |

**剩下 2 个是实现上的偏差，不是数据的性质**：`r_stroke` / `r_depres` 在源数据里是**逐访视**
标志（411 / 474 人在多次访视上被置位，最多 11 / 18 次），但发射代码写死了 `.iloc[0]` 只取首次，
而两个 dedup 开关**管不到发射逻辑**。`--per-visit-events` 补上这一条。

量级：打开后 371,088 行（+1,734，**+0.47%**），可重复 token 43 → 45，序列中位 73 → 74。
**已交付的 `rosmap_fullvisit` 是在这个开关 off 下建的**，也就是它带着这个 +0.47% 的偏差；
因为量级远小于单 seed 噪声（panel a 的 CI 半宽 ±0.03~0.06），没有重跑。
打开它还会改变 token 的语义 —— 从"首次发病"变成"本次访视有记录"，所以默认必须是 off。

`build.py` 里有一条自检：没关全局去重时，非豁免家族一旦出现重复就 assert 失败，也就是
"绕过了全局去重"这件事不会静默发生；`--no-global-dedup` 下改成如实列出"CONT 之外还在重复的"
有哪些（就是那 8 个用药 token）。

默认分支（不给任何开关）的输出与交付版 `train.bin/val.bin/labels.csv` **md5 完全一致**，
本地和 RIS 上各验证过一次 —— 也就是说把分词搬到集群没有引入任何漂移。
加 `--per-visit-events` 之后，`--nodedup all --no-global-dedup` 的输出也仍与集群上那份
`rosmap_fullvisit` **md5 一致**（新开关是纯新增，不改现有路径）。

## 三分 split：`--split train,val,test`

`figure2_eval/README.md` 第 3.3 节的问题：`delphi/train.py` 用 `always_save_checkpoint = False`，
`ckpt.pt` 就是 **best-val** 那一步，所以 val 既选了 checkpoint 又被拿来报数，**所有 AUC 只能当上界读**。
根治要在这里切第三份。

```bash
# 默认：什么都不给 = 交付版的两分行为，不写 test.bin
python build.py --out data_rosmap --dataset rosmap

# 三分（在 RIS 上跑，跑完自动装进 ../delphi/data/<dataset>/，含 test.bin）
sbatch ris/build.sbatch --out data_rosmap_split811 --dataset rosmap_split811 --split 0.8,0.1,0.1
```

**默认是 `0.9,0.1,0`，也就是不开。** 理由和别的开关一样：已交付的 `Delphi-ROSMAP/ckpt.pt`、
`figure2_eval/results/` 下的每一张图，都长在那份两分数据上。默认切三分等于把所有人脚下的训练集
换掉一遍，而 `.bin` 换没换**从文件本身看不出来** —— `engine.load` 只校验 `vocab_size` 和
`ignore_tokens`，词表压根没变。

### 段的顺序是 `[train | test | val]`，不是 `[train | val | test]`

划分代码只 `RNG.shuffle(pids)` **一次**（加 test 多抽一次随机数就是另一份数据集，而两份
`.bin` 肉眼完全一样，只有 md5 不同），三份是同一个排列上的三段：

| | 切法 | 0.9,0.1,0（默认） | 0.8,0.1,0.1 |
|---|---|---|---|
| train | `pids[0 : int(f_tr·N)]` | `[0:3985]` 3,985 人 / 142,528 行 | `[0:3542]` **3,542 人 / 126,471 行** |
| test | `pids[int(f_tr·N) : va_start]` | 空 → **不写 test.bin** | `[3542:3985]` **443 人 / 16,057 行** |
| val | `pids[va_start :]`，`va_start = int((1−f_val)·N)` | `[3985:4428]` 443 人 / 15,575 行 | `[3985:4428]` **同一批人，同样的字节** |

**为什么 val 必须和原来那份完全相同。** val 的起点只由 `f_val` 决定，和 train/test 怎么分剩下那
90% 无关。于是：

* 两份数据集上报的 **val 指标直接可比** —— 同一批 443 人、同样的行、`val.bin` 的 md5 都一样
  （实测 `5957dd211765f5261bc9c7646d32ed6a`，两分和三分一致）；
* train 只是**变小**（3,985 → 3,542 是前缀子集），差异只能归到"少了 443 个人"，不是"换了一份数据"；
* test 是**从原 train 里划出来的**，所以它对现有那个 ckpt **并不是** held-out（那 443 人训过），
  只对**用三分数据重训出来的新 ckpt** 才是 held-out。这一条别记反了。

反过来，如果按直觉切成 `[train|val|test]`，val 会整体平移到另一批人身上：现有的每一个 AUC
都失去参照物，新旧数字并排放在一张表里就是在比两件不同的事，**而且不会有任何报错**。
所以 `build.py` 把这条写成了**断言**（不只是打印）：`f_val` 还是基线的 `0.1` 时，val 必须
仍是 `pids[int(0.9·N):]` 那批人，段顺序一旦被改回 `[train|val|test]` 就在写 `.bin` **之前**
直接炸。结果同时打印、也写进 `meta.json`：

```
  基线 val.bin <- .../tokenization/data_rosmap/val.bin  (443 人)  与本次 val 同一批人 = True
  与两分基线（切点 int(0.9*N)=3985）对读: 切点 val 段相同 = True；train 是前缀 = True；与落盘基线 val.bin 同一批人 = True
```

**这三个布尔量不是一回事，别混着读。** 前两个（`val_identical` / `train_is_prefix`）是**同一次
运行内部的索引关系** —— 分别等价于"`va_start == int(0.9·N)`"和"`ntr <= int(0.9·N)`"。它们能抓住
段顺序或切点被改，但抓不住**排列本身变了**：只要有人在 `RNG.shuffle(pids)` 之前多抽一次随机数
（`death_age` 用的是同一个 `RNG`），整份 val 就换人了，而这两个量照样是 `True`，两份 `.bin` 肉眼
也一模一样。所以还有第三个、也是唯一实测的一个：`val_projids_match_reference_bin` —— 把本次的 val
和**一份已经落盘的 `val.bin`** 逐人对比。

* 基线默认取 `tokenization/data_rosmap/val.bin`，可用 `ROSMAP_BASELINE_VAL_BIN=<path>` 覆盖；
  读的是**写出之前**的那份文件，所以 `--out data_rosmap` 原地重建时比的就是上一轮（交付版）。
* projid 的划分和去重开关无关（同一个排列），实测 `data_rosmap` / `data_rosmap_nodedup` 的
  val 是同一批 443 人，所以这份基线对 nodedup / fullvisit / 三分 都适用。
* **人数相同却不是同一批人** + `f_val` 仍是 0.1 → 直接 `SystemExit`，一个字节都不写。这是
  "RNG 被多抽了一次"的唯一指纹。人数也不同（多半是原始表更新了）则只打印警告。
* 找不到基线文件就在 `meta.json` 里记 `null` 并打印"跳过"，**不伪造 `True`**。

`meta.json` 新增 `split_fractions` / `split_counts`（每份的人数与行数）/ `split_layout` /
`split_matches_twoway_baseline`。另有几条断言在跑的时候就会炸而不是事后才发现：三份 projid
**两两无交集**（人级别，不是行级别 —— 同一个人的行分到两边，test 上的"泛化"其实是在背这个人的
前半段）、并集等于全部人、三份行数之和等于总行数、train 段取整后非空。

**两分请求（`test` 比例为 0）时 `ntr` 直接取 `va_start`**，不用 `int(f_tr·N)`：两者在数学上
相等（和必须为 1 已校验），但浮点取整可以差 1 个人，那个人会掉进一个"既不在 train 也不在 val"
的空段 —— 旧写法会把他写进一份 `split_fractions.test = 0.0` 的 `test.bin`，同时让他从 train 里
消失，两头都不报错。默认 `0.9,0.1,0` 下这条是 no-op（两边都是 3985，md5 已验）。

### 验证过的等价性

| 检查 | 结果 |
|---|---|
| 默认参数 vs `data_rosmap/` 的 `train.bin`/`val.bin`/`labels.csv` md5 | **三个全等**（`aa8f5c1f…` / `5957dd21…` / `5c491771…`） |
| `--split 0.8,0.1,0.1` 的 `val.bin` vs 交付版 `val.bin` | **md5 全等** |
| 三分 `train.bin` vs 交付版 `train.bin` 里那 3,542 人的行 | **逐字节相同** |
| 三分 `test.bin` vs 交付版 `train.bin` 里那 443 人的行 | **逐字节相同** |
| 三分 train + test 行数 | 126,471 + 16,057 = 142,528 = 交付版 train |
| 三份 projid 两两交集 | 0 / 0 / 0 |
| 加了基线实测检查之后重跑上面两条 | 两份输出的 md5 全部不变（`aa8f5c1f…` / `5957dd21…` / 三分 `1454672d…` / `88cd75cf…`） |
| 把 `ROSMAP_BASELINE_VAL_BIN` 指到一份人数相同但换了人的 `.bin` | 写 `.bin` **之前** `SystemExit`，输出目录为空 |

`meta.json` 会多出上面那几个键（`.bin` 和 `labels.csv` 不变）。`data_rosmap/meta.json` 本来就
比代码旧，去重那几个键也不在里面，所以 meta 的 md5 从来不是复现判据，**判据是三个数据文件**。

### 下游怎么吃 test.bin

`figure2_eval/radc_delphi/engine.py:load_split` 读的是 `f"{split}.bin"`，`run_figure2.sh`
（`SPLIT=test ./run_figure2.sh`）和各个 `probe_*.py`（`--split`，默认 `val`）因此装好 `test.bin`
就能直接用，这条路径一行都不用改。

**但 figure2_eval 里更新的那批入口没有一个认 `split`，它们把 `val` 写死在源码里。** 在三分 ckpt
上跑它们，报出来的仍然是 val 的数，而且不会有任何提示 —— 这正好是三分想解决的那个问题（val 既
选 ckpt 又报数）原封不动地留着。用之前必须先看一眼：

| 入口 | 现状 |
|---|---|
| `run_figure2.sh` / `figure2/*` / `probe_*.py` / `perturbation_eval.py` | 有 `--split`，`SPLIT=test` 可用 |
| `direct_head.py`（`JOB=dh`） | `frame_with_emb(..., "train")` / `(..., "val")` 字面量写死 |
| `lr_endpoint_baseline.py`（`JOB=lrb`） | `frame_lite(ddir, "train")` / `(ddir, "val")` 字面量写死 |
| `next_visit_auc.py`（`JOB=nva`）、`next_token_auc.py`（`JOB=nta`） | 直接 `np.fromfile(.../val.bin)` |
| `change_skill.py`（`JOB=chg`）、`window_baselines.py`（`JOB=wbl`） | 复用 `next_visit_auc.truth_table` / `score_run`，跟着它一起只看 val |
| `delphi/evaluate_auc_rosmap.py` | `val.bin` 写死，不认 `--split` |

给它们加 `--split` 是独立的一件事，本次没动（那些文件正在被别的改动动）。

另外：`delphi/train.py` 只 memmap `train.bin` / `val.bin`，选 ckpt 仍然只用 val —— 这正是我们想要
的分工（val 选模型，test 只报数）。所以 `--split` 的 val 比例必须 > 0，`build.py` 里直接拒绝 0。

### 重训之前必须换训练配置 —— 漏了不报错，而且正好毁掉三分的意义

`build.py` 会在 `$OUT/` 里顺手写一份 `train_delphi_rosmap.py`，`dataset` / `out_dir` 已经指向新
数据集（`--dataset rosmap_split811` → `dataset='rosmap_split811'`、`out_dir='Delphi-ROSMAP-split811'`）。
**但没有任何脚本会把它装进 `../delphi/config/`**，而且它的文件名不带数据集后缀 —— 直接 `cp`
过去会**覆盖交付版的 `config/train_delphi_rosmap.py`**。所以要显式改名：

```bash
cp data_rosmap_split811/train_delphi_rosmap.py ../delphi/config/train_delphi_rosmap_split811.py
# 别急着提交：先和你要对读的那份配方 diff 一遍
diff ../delphi/config/train_delphi_rosmap_nodedup_pos.py ../delphi/config/train_delphi_rosmap_split811.py
cd ../delphi && CFG=config/train_delphi_rosmap_split811.py sbatch ris/train.sbatch
```

**生成的那份配置只知道数据，不知道配方。** 它写的是 `dataset` / `vocab_size` / `block_size` /
`ignore_tokens` / `lifestyle_token_min|max`（这一项本次补上了 —— 以前不写，`train.py` 会退回
UKB 的 `(3, 11)`，ROSMAP 独有的 12..32 背景 token 就不再参与年龄抖动，静默换配方）。它**不写**
`pos_embedding` / `aux_head` / `aux_lambda` / `visit_heads`，`train.py` 对这些的默认值全是
`False`，漏了不报错。`max_iters` / `token_dropout` 也可能和交付版不同（模板是 20000 / 0.1，
`config/train_delphi_rosmap.py` 是 12000 / 0.0）。**在三分数据上重训、再去和 val 上的旧数字
对读之前，这些必须先对齐**，否则换掉的不只是 split，还有配方。

漏这一步的后果是**静默的**：`delphi/ris/train.sbatch` 只检查"配置里写的那个 dataset 的
`train.bin/val.bin/labels.csv` 在不在"，拿旧配置就是**又训了一遍两分数据**，一路不报错；而那个
ckpt 在 `test.bin` 上报出来的数**没有任何 held-out 含义**——那 443 人它训过。这和下面这条是同
一个坑的两半：

> **`test.bin` 对交付版的 `Delphi-ROSMAP/ckpt.pt` 不是 held-out。** 它是从原 `train` 里切出来
> 的，那 443 人在交付版里训过。只有用三分数据**重训出来的新 ckpt**，`SPLIT=test` 才有意义。

从 Mac 往 RIS 传数据走 `delphi/ris/deploy.sh --data <ds>`：它按**显式文件名**上传，`test.bin`
原先不在列表里（本次已补上，本地没有 test.bin 时会提示检查集群上有没有残留的旧的）。在 RIS 上用
`tokenization/ris/build.sbatch` 直接建则两种情况都已经自动处理。

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
