# Figure 3 基线 1 的**配对参照**：fullvisit 数据、三分 split、看全部历史。
#
# 基线 1（fig3_visit1）必须用 fullvisit 数据，否则"当前访视"只有变化的那几个量，基线被人为
# 削弱。但这样它和主模型（nodedup 数据）之间就多了一个"数据表示"的差异。这一支和基线 1 **只差
# 注意力范围**，所以 "这个 − 基线 1" 才是干净的"历史的价值"。
# 超参沿用主模型（nodedup-pos-3w），只改数据强制的项。
exec(open('config/train_delphi_rosmap_nodedup_pos_3w.py').read())

out_dir = 'Delphi-ROSMAP-fig3-fullvisit3w'
wandb_run_name = 'fig3-fullvisit3w' + str(time.time())
dataset = 'rosmap_fullvisit_3w'
block_size = 320          # 与 config/train_delphi_rosmap_fullvisit.py 相同：最长 285 真实 + <=20 no-event
pos_embedding = False     # 与基线 1 保持一致（基线 1 必须关，见那边）
