# Figure 3 基线 1：只看"现在"的 Delphi —— 每个位置只能注意背景块 + 当前这次访视。
#
# 与 fig3_fullvisit3w **只差 attn_visits**。fullvisit 数据每次访视发射全部状态，所以"当前
# 访视"就是完整的现状。"fig3_fullvisit3w − 这个" = 历史的价值，用 transformer 自己量。
exec(open('config/train_delphi_rosmap_fig3_fullvisit3w.py').read())

out_dir = 'Delphi-ROSMAP-fig3-visit1'
# ris/train.sbatch 用正则读这一行来检查数据是否就位，exec 继承来的值它看不到，所以显式重写一遍
dataset = 'rosmap_fullvisit_3w'
wandb_run_name = 'fig3-visit1' + str(time.time())
attn_visits = 1           # model.py 改动 C；防泄漏测试见 test_scope_mask.py
