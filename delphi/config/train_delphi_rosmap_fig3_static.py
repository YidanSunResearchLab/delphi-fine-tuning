# Figure 3 基线 2：只看性别 + 背景块 + 当前年龄的 Delphi（transformer 版人口学先验）。
#
# 先执行主模型的配置（panel A 的 nodedup-pos-3w），再只改下面几项。同数据、同 train split、
# 同配方，所以"主模型 − 这个"= transformer 从**全部临床 token**里挣到了多少。
exec(open('config/train_delphi_rosmap_nodedup_pos_3w.py').read())

out_dir = 'Delphi-ROSMAP-fig3-static'
# ris/train.sbatch 用正则读这一行来检查数据是否就位，exec 继承来的值它看不到，所以显式重写一遍
dataset = 'rosmap_nodedup_3w'
wandb_run_name = 'fig3-static' + str(time.time())
static_only = True        # 非背景 token 在输入里抹成 no-event、只能注意背景块（model.py 改动 C）
pos_embedding = False     # 位置下标 = 前面有多少个 token = 历史长度，会漏
