import time

# ROSMAP / Delphi-2M —— **完全不去重**的消融分支（"每次访视都看到完整的一次访视"）。
#
# 与 config/train_delphi_rosmap.py 逐行相同，只有三处不同，且都是数据强制的，不是调参：
#   dataset      rosmap -> rosmap_fullvisit
#   out_dir      Delphi-ROSMAP -> Delphi-ROSMAP-fullvisit
#   block_size   96 -> 320
# 其余超参（层数/宽度/dropout/lr/iters/seed）一律保持不变 —— 否则三个模型之间的差异就不再能
# 归因到分词，而是混进了训练配方。这也是为什么 block_size 必须动：它不是超参，是"能不能
# 装下一条完整轨迹"的硬约束。
#
# 数据来自 ../tokenization/build.py --nodedup all --no-global-dedup：
#   * spec.CONT 的全部 15 个纵向量都改成**每次随访发射一个 token**（run-length 去重全关）；
#   * 全局 first-occurrence 去重整体关闭 —— 这一条额外让**用药 ON/OFF 的第二次开关**回来了
#     （2,008 个 (人,token) 组合），交付版把"停了又吃"里的第二次 ON 整条删掉。
# 后果：token 数 158,103 -> 369,354（2.34x），逐访视测量占 76.6%，人均 63.9 个；
# 序列长度中位 35 -> 73，p99 57 -> 233，max 65 -> 285。
# 实测可重复 token 从 0 个变成 43 个（35 个分箱 + 8 个用药开关）；CRP/IL6/TNFA 不在其中，
# 因为那三个生物标记在 ROSMAP 里只测了一次，没有第二次可重复。

out_dir = 'Delphi-ROSMAP-fullvisit'
eval_interval = 250
eval_iters = 50
log_interval = 25
seed = 42

always_save_checkpoint = False

wandb_log = False
wandb_project = 'delphi-rosmap'
wandb_run_name = 'rosmap-fullvisit' + str(time.time())

dataset = 'rosmap_fullvisit'
batch_size = 128
# 最长轨迹 285 个真实 token + <=20 个注入的 no-event = 304（实测 max）。320 留出和另两支
# 相同比例的余量（96/85=1.13, 144/134=1.07, 320/304=1.05）。**必须**盖住 max：
# get_batch 超长时从左边裁，裁掉的正是基线那一访，也就是每条轨迹信息最密的位置。
block_size = 320
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
# padding, sex and the background/lifestyle block (post-shift ids, cf. data/rosmap_fullvisit/labels.csv)
ignore_tokens = [0] + list(range(2, 34))
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5

lifestyle_token_min = 3
lifestyle_token_max = 32
