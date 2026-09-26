# Figure 3 基线（snapshot 数据）：只看"现在"—— 背景块首份 + 当前这次访视。
# snapshot 每次访视都是完整状态，所以"当前访视"就是完整的现状。与 train_delphi_rosmap_snapshot.py
# 只差注意力范围，"snapshot − 这个"= 历史的价值。static_first_only：背景块每次访视都重发，
# 不加它的话数拷贝份数就等于数访视次数（test_scope_mask.py 的 snapshot 段钉住了这一条）。
exec(open('config/train_delphi_rosmap_snapshot.py').read())

out_dir = 'Delphi-ROSMAP-snapshot-visit1'
dataset = 'rosmap_snapshot'
wandb_run_name = 'rosmap-snapshot-visit1' + str(time.time())
attn_visits = 1
static_first_only = True
pos_embedding = False     # 位置下标 = 前面有多少 token = 历史长度，会漏
