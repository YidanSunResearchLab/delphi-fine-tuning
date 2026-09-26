# ROSMAP / Delphi-2M —— **完全不去重**（tokenization/build.py --snapshot）：每次访视发射完整状态。
#
# 先执行 fullvisit 的配置（它与交付版 train_delphi_rosmap.py 逐行相同，只差数据强制的 dataset /
# out_dir / block_size），再只改这三项。和 dedup（Delphi-ROSMAP-gpubase）、fullvisit 同一份两分
# 划分、同一套配方，所以三者之间的差异只能归到分词。
#
# block_size：snapshot 序列最长 710 个真实 token + <=20 个注入的 no-event = 730，取 736
# （和其它几支同样的"必须盖住 max"原则 —— get_batch 超长时从左裁，裁掉的是基线那一访）。
# 已知的配方交互：lifestyle_augmentations 会给背景块 token 的年龄加抖动；snapshot 里背景块每次
# 访视都发一遍，所以每一份拷贝都会被各自抖动。配方保持不变，否则差异里会混进配方变量。
exec(open('config/train_delphi_rosmap_fullvisit.py').read())

out_dir = 'Delphi-ROSMAP-snapshot'
# ris/train.sbatch 用正则读这一行来检查数据是否就位，exec 继承来的值它看不到，所以显式写一遍
dataset = 'rosmap_snapshot'
wandb_run_name = 'rosmap-snapshot' + str(time.time())
block_size = 736
