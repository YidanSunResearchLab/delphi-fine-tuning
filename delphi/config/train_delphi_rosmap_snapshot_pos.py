# Figure 3 主模型（新）：完全不去重（snapshot）+ 可学习位置嵌入。
# 与 config/train_delphi_rosmap_snapshot.py 只差 pos_embedding —— 之前的主模型 nodedup-pos
# 正是靠这一项站住的；snapshot 每次访视重发全部状态，"找到最近一次的值"最依赖它。
exec(open('config/train_delphi_rosmap_snapshot.py').read())

out_dir = 'Delphi-ROSMAP-snapshot-pos'
# ris/train.sbatch 用正则读这一行来检查数据是否就位，exec 继承来的值它看不到，所以显式写一遍
dataset = 'rosmap_snapshot'
wandb_run_name = 'rosmap-snapshot-pos' + str(time.time())
pos_embedding = True
