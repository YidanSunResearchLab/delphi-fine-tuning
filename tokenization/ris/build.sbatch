#!/bin/bash
# build.sbatch -- 在 RIS compute2 上重新分词。
#
#   sbatch ris/build.sbatch --out data_rosmap --dataset rosmap
#   sbatch ris/build.sbatch --out data_rosmap_nodedup  --dataset rosmap_nodedup  --nodedup MMSE,COGN
#   sbatch ris/build.sbatch --out data_rosmap_fullvisit --dataset rosmap_fullvisit \
#            --nodedup all --no-global-dedup
#   sbatch ris/build.sbatch --out data_rosmap_split811 --dataset rosmap_split811 --split 0.8,0.1,0.1
#
# 跑完自动把 {train,val}.bin + labels.csv + meta.json 装进 ../delphi/data/<dataset>/，
# 并重建 figure2_eval 那边的 symlink —— 手工 cp 漏一个文件是静默失败（labels.csv 不匹配
# 才会被 run_figure2.sh 的前置检查抓到，meta.json 漏了则连检查都没有，no-repeat 规则直接失效）。
#
# 原始 RADC 表格在 $AD_BASE/raw/，是**受限数据**：一次性刻意上传，不随代码同步。
#SBATCH --job-name=rosmap-tok
#SBATCH --partition=general-cpu
#SBATCH --account=compute2-mdan
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/tok_%j.out
#SBATCH --error=logs/tok_%j.err
#SBATCH --exclude=c2-node-011

set -euo pipefail
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0915_rosmap_fig2}"
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ad-projection

cd "$BASE/tokenization"
mkdir -p logs
export ROSMAP_DATA_ROOT="$BASE/raw"
echo "host: $(hostname) | data_root=$ROSMAP_DATA_ROOT | args: $*"

python -u build.py "$@"

# --- 装进 delphi/data/<dataset>/ 并接好 figure2_eval 的 symlink -------------------
DS=""; OUTD=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DS="$2"; shift 2 ;;
    --out)     OUTD="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "$DS" ] && [ -n "$OUTD" ] || { echo "没给 --dataset/--out，跳过安装步骤"; exit 0; }
DEST="$BASE/delphi/data/$DS"
mkdir -p "$DEST"; chmod 700 "$DEST"
cp "$OUTD"/train.bin "$OUTD"/val.bin "$OUTD"/labels.csv "$OUTD"/meta.json "$DEST"/
# test.bin 只在 --split 的 test 比例 > 0 时存在。两边都要处理，因为两种漏法都是静默的：
#   * 生成了却不 cp —— 后面想评 held-out 时才发现 data/<ds>/ 里没有，而那时训练已经跑完；
#   * 上一轮三分留下的旧 test.bin 不删 —— 它和本轮 train **有交集**，会被当成 held-out 用。
if [ -f "$OUTD/test.bin" ]; then
  cp "$OUTD"/test.bin "$DEST"/
else
  rm -f "$DEST"/test.bin
fi
chmod 600 "$DEST"/*
ln -sfn "../../delphi/data/$DS" "$BASE/figure2_eval/data/$DS"
echo "装好 -> $DEST"
ls -la "$DEST"
python - <<PYEOF
import json
m = json.load(open("$DEST/meta.json"))
print("vocab_size", m["vocab_size"])
print("nodedup_prefixes", m["nodedup_prefixes"])
print("no_global_dedup", m.get("no_global_dedup"))
print("split_fractions", m.get("split_fractions"), m.get("split_counts"))
print("与两分基线对读", m.get("split_matches_twoway_baseline"))
ids = m["repeatable_tokens_postshift"]
print("实测可重复 token", len(ids), [m["vocab"][t-2] for t in ids])
PYEOF
