import time

# ROSMAP / Delphi-2M —— **认知量表不去重**的消融分支。
#
# 与 config/train_delphi_rosmap.py 逐行相同，只有三处不同，且都是数据强制的，不是调参：
#   dataset      rosmap -> rosmap_nodedup
#   out_dir      Delphi-ROSMAP -> Delphi-ROSMAP-nodedup
#   block_size   96 -> 144
# 其余超参（层数/宽度/dropout/lr/iters/seed）一律保持不变，否则两个模型的差异就不再能
# 归因到分词，而是混进了训练配方。
#
# 数据来自 ../tokenization/build.py --nodedup MMSE,COGN：MMSE 与 cogn_global 改成**每次随访
# 发射一个 token**，run-length 去重和全局 first-occurrence 去重都豁免。
# 后果：token 数 158,103 -> 212,078（+34%），认知 token 占 32.5%，人均 15.6 个；
# 序列长度中位 35 -> 46，p99 57 -> 90，max 65 -> 114。
# 换来的是交付版结构上不可能存在的往复：MMSE 2,794 次、cogn_global 2,426 次"回到曾经离开
# 过的箱"（交付版恒为 0，因为全局去重把第二次出现整条丢掉）。

out_dir = 'Delphi-ROSMAP-nodedup-pos-3w'
eval_interval = 250
eval_iters = 50
log_interval = 25
seed = 42

always_save_checkpoint = False

wandb_log = False
wandb_project = 'delphi-rosmap'
wandb_run_name = 'nodedup_pos-3w' + str(time.time())

dataset = 'rosmap_nodedup_3w'
batch_size = 128
# 最长轨迹 114 个真实 token + <=20 个注入的 no-event = 134（实测 max，见 build.py 的长度报告）。
# 144 留出和交付版相同比例的余量（96/85 = 1.13 vs 144/134 = 1.07）。**必须**盖住 max：
# get_batch 超长时从左边裁，裁掉的正是基线那一访，也就是每条轨迹信息最密的位置。
block_size = 144
data_fraction = 1.0

n_layer = 6
n_head = 6
n_embd = 96
dropout = 0.1
weight_decay = 2e-1
vocab_size = 129      # 词表完全不变——不去重只改发射次数，不新增 token

learning_rate = 6e-4
max_iters = 12000
lr_decay_iters = 12000
min_lr = 6e-5
beta2 = 0.99

warmup_iters = 500
# padding, sex and the background/lifestyle block (post-shift ids, cf. data/rosmap_nodedup/labels.csv)
ignore_tokens = [0] + list(range(2, 34))
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5

lifestyle_token_min = 3
lifestyle_token_max = 32

# ---------------------------------------------------------------- 唯一的实验变量
# 重新启用可学习的绝对位置嵌入。假设：上游把 wpe 注释掉之后，模型对时间的唯一感知是**绝对**
# 年龄的加性正弦码，于是"某个序数家族最近一次的值是什么"需要在 n_embd 维空间里做跨位置的
# 年龄大小比较，而**相对**时间 (age_q - age_k) 根本不可得。那个量恰好是单访视 logistic 基线
# （figure2_eval/window_baselines.py 的 W）的全部特征，也是实测里 transformer 唯一没能超过
# 基线的地方（skill vs W 三档全为负，见 figure2_eval/README.md 第 9 节）。
#
# .bin 的 token 流按年龄稳定排序，所以**位置下标就是新近度排名**，"最近一次" = "下标最大的
# 那个"，attention 一步可达。参数量 0.693M -> 0.707M（+144x96）。
# 其余每一个超参都和 config/train_delphi_rosmap_nodedup.py 逐行相同 —— 这是单变量实验。
pos_embedding = True

# ---------------------------------------------------------------- 三分 split
# 数据来自 tokenization/build.py --split 0.8,0.1,0.1。**val 与两分版逐字节相同**、
# train 是两分版 train 的前缀子集（build.py 沿用同一次 RNG.shuffle 再切第三段），
# 所以这一支和已有结果可逐人对读，唯一的差别是 train 少了 443 人（3985 -> 3542）。
# 目的见 figure2_eval/README.md 第 3.3 节：val 既用于选 ckpt 又用于报数，所有数字都是上界。
# 这一支的 ckpt 仍然在 val 上选，但**报数必须在 test.bin 上**。
