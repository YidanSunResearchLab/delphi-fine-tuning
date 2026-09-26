# Figure 3 基线（snapshot 数据）：只看性别 + 背景块 + 当前年龄。
exec(open('config/train_delphi_rosmap_snapshot.py').read())

out_dir = 'Delphi-ROSMAP-snapshot-static'
dataset = 'rosmap_snapshot'
wandb_run_name = 'rosmap-snapshot-static' + str(time.time())
static_only = True
static_first_only = True
pos_embedding = False
