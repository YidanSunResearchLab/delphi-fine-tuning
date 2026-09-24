#!/bin/bash
# figure2.sbatch -- figure2_eval 的重计算，跑在 RIS compute2 上。
#
#   sbatch ris/figure2.sbatch                  # figure2 全量（MC + 四面板 + 合成图 x2 cohort）
#   sbatch ris/figure2.sbatch --limit 300      # 冒烟
#   JOB=pvo   sbatch ris/figure2.sbatch        # predict_vs_observe（全 token 校准）
#   JOB=sweep sbatch ris/figure2.sbatch        # 报告口径扫描
#   JOB=diag  sbatch ris/figure2.sbatch        # diagnostics
#   VISIT_SIZES=0 sbatch ris/figure2.sbatch    # 关掉访视采样（用于和本地基线做纯机器对照）
#
#   DATASET=rosmap_nodedup CKPT=../delphi/Delphi-ROSMAP-nodedup/ckpt.pt FIG2_TAG=nodedup \
#     sbatch ris/figure2.sbatch                # 认知量表不去重的那份分词 + 它自己的 ckpt
#
# DATASET / CKPT / FIG2_TAG 三个一定要**一起**给。给错组合不会崩：engine.load 只校验
# vocab_size 和 ignore_tokens，两份分词这两项完全相同（129 / [0]+2..33），所以用交付版 ckpt
# 去评不去重的数据会安安静静跑完并给出一张错图。FIG2_TAG 决定输出子目录，漏了就会覆盖
# results/figure2/ 里的权威结果。
#
# 这个文件照抄 delphi-fine-tuning 的 slurm/figure2.sbatch（同一个实验室、同一个集群的工作配置），
# 只换了路径和入口。**不要**改回 LSF/bsub：那是 compute1 的写法，这里是 compute2 + Slurm。
#SBATCH --job-name=rosmap-fig2
#SBATCH --partition=general-cpu
#SBATCH --account=compute2-mdan
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fig2_%j.out
#SBATCH --error=logs/fig2_%j.err
# c2-node-011 坏了（上游 2026-07-28 记录）：调度到它上面的任务——哪怕 --wrap='hostname'——
# 都会以 signal 53 立刻死掉且不写日志，而 Slurm 仍把它报成 idle，所以会持续吸引任务。
# 节点修好后删掉这行。
#SBATCH --exclude=c2-node-011

set -euo pipefail
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0915_rosmap_fig2}"
CODE="$BASE/figure2_eval"

# conda 装在最早那个 checkout 里，从未按 checkout 复制过，所以**不要**默认成 $BASE/miniforge3。
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ] || {
  echo "FATAL: $CONDA_ROOT 下没有 conda" >&2
  echo "  用 AD_CONDA=/path/to/miniforge3 指定（含 etc/profile.d/conda.sh 的那个目录）" >&2
  exit 1; }
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ad-projection
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1                      # 每个 MC worker 必须单线程，否则严重超订
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-32} - 2 ))

DATASET="${DATASET:-rosmap}"
CKPT="${CKPT:-../delphi/Delphi-ROSMAP/ckpt.pt}"
export DATASET CKPT
export FIG2_TAG="${FIG2_TAG:-}"

cd "$CODE"
echo "host: $(hostname) | workers=$WORKERS | JOB=${JOB:-figure2} | dataset=$DATASET | \
ckpt=$CKPT | tag=${FIG2_TAG:-<none>} | extra: ${*:-none}"

# ---------------------------------------------------------------- 前置检查
python - <<'PYEOF'
import importlib.util, sys        # 只 import importlib 不会绑定 .util 子模块
missing = [m for m in ("torch","numpy","pandas","sklearn","matplotlib","scipy") if not importlib.util.find_spec(m)]
if missing: sys.exit(f"FATAL: 缺包 {missing}")
print("deps ok" + ("  (umap 可用 -> panel d 用真 UMAP)" if importlib.util.find_spec("umap")
                   else "  (无 umap -> panel d 退回 PCA 并如实标注)"))
PYEOF
for f in "data/$DATASET/train.bin" "data/$DATASET/val.bin" "data/$DATASET/labels.csv" "$CKPT"; do
  [ -e "$f" ] || { echo "FATAL: 缺 $f —— 数据没上传。见 ris/README.md" >&2; exit 1; }
done
# visit_sizes 是从 train.bin 推出来的派生量，缺了就地重建，避免和本地那份不一致。
# 交付版落在仓库根，其余数据集落在自己的 data/<name>/ 下（engine 先找 data_dir 再退回根）。
if [ "$DATASET" = "rosmap" ]; then
  [ -f visit_sizes.npy ] || python make_visit_sizes.py
else
  [ -f "data/$DATASET/visit_sizes.npy" ] || python make_visit_sizes.py --dataset "$DATASET"
fi

# ---------------------------------------------------------------- 跑
case "${JOB:-figure2}" in
  figure2) WORKERS="$WORKERS" DEVICE="${DEVICE:-cpu}" DATA="data/$DATASET" CKPT="$CKPT" \
             ./run_figure2.sh "$@" ;;
  # 下一次访视的判别力（无 rollout）。它自己同时加载多个 ckpt，所以不走 DATASET/CKPT，
  # 而是用 RUNS="name:ckpt:dataset ..." 指定。真值一律取 --truth-dataset（必须是最不去重
  # 那份，它才是忠实记录）。
  nva)     python -u next_visit_auc.py --runs ${RUNS:-\
             dedup:../delphi/Delphi-ROSMAP-gpubase/ckpt.pt:rosmap \
             nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup} \
             --truth-dataset "${TRUTH_DS:-rosmap_nodedup}" \
             --out "results/next_visit_auc_${NVA_TAG:-default}.json" "$@" ;;
  # 最朴素的口径：预测下一个 token（训练目标本身）。每个模型在**自己的**流上，分母不同。
  nta)     python -u next_token_auc.py --runs ${RUNS:-\
             dedup:../delphi/Delphi-ROSMAP-gpubase/ckpt.pt:rosmap \
             nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup \
             fullvisit:../delphi/Delphi-ROSMAP-fullvisit/ckpt.pt:rosmap_fullvisit} \
             --out "results/next_token_auc_${NTA_TAG:-default}.json" "$@" ;;
  # 区分"复读当前状态"和"真学到变化"：对持续性 / 一阶 Markov 两个基线报 skill score
  chg)     python -u change_skill.py --runs ${RUNS:-\
             dedup:../delphi/Delphi-ROSMAP-gpubase/ckpt.pt:rosmap \
             nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup \
             fullvisit:../delphi/Delphi-ROSMAP-fullvisit/ckpt.pt:rosmap_fullvisit} \
             --out "results/change_skill_${CHG_TAG:-default}.json" "$@" ;;
  # 两个"视野极窄"的基线：完全不转变 / 只看当次访视 token 的 logistic
  wbl)     python -u window_baselines.py --runs ${RUNS:-\
             dedup:../delphi/Delphi-ROSMAP-gpubase/ckpt.pt:rosmap \
             nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup \
             fullvisit:../delphi/Delphi-ROSMAP-fullvisit/ckpt.pt:rosmap_fullvisit} \
             --out "results/window_baselines_${WBL_TAG:-default}.json" "$@" ;;
  # figure2 终点上的逻辑回归对照（不跑 MC，纯观测 + 基线前缀特征）
  lrb)     python -u lr_endpoint_baseline.py \
             --out "results/lr_endpoint_baseline_${LRB_TAG:-default}.json" "$@" ;;
  # 方案2 第一步：冻结 encoder，基线前缀隐状态 vs 原始特征，接 logistic 判别头
  dh)      python -u direct_head.py --runs ${RUNS:-\
             dedup:../delphi/Delphi-ROSMAP-gpubase/ckpt.pt:rosmap \
             nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup \
             fullvisit:../delphi/Delphi-ROSMAP-fullvisit/ckpt.pt:rosmap_fullvisit} \
             --tags ${DH_TAGS:-gpubase nodedup fullvisit} \
             --eval-split "${EVAL_SPLIT:-val}" \
             --out "results/direct_head_${DH_TAG:-default}.json" "$@" ;;
  # 扰动引擎评估：剂量反应 / 零扰动对照 / 注入对照 / 靶点-脱靶 / MC 噪声地板
  pe)      python -u perturbation_eval.py --runs ${RUNS:-\
             nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup} \
             --n-mc "${NMC:-100}" --workers "$WORKERS" \
             --out "results/perturbation_eval_${PE_TAG:-default}.json" "$@" ;;
  pvo)     python predict_vs_observe.py --workers "$WORKERS" "$@" ;;
  sweep)   python predict_vs_observe.py --sweep --workers "$WORKERS" "$@" ;;
  diag)    python diagnostics.py --threads 8 "$@" ;;
  probe)   python probe_normal.py --threads 8 "$@" ;;
  shift)   python probe_shift.py --threads 8 "$@" ;;
  statics) python probe_statics.py --threads 8 "$@" ;;
  head)    python probe_head.py --threads 8 "$@" ;;
  batch)   python probe_batch.py --threads 8 "$@" ;;
  loss)    python probe_loss.py --threads 8 "$@" ;;
  masktie) python probe_masktie.py --threads 8 "$@" ;;
  rcomp)   python probe_rollout_comp.py --threads 8 "$@" ;;
  counts)  python probe_counts.py --threads 8 "$@" ;;
  perturb) python probe_perturb.py --threads 8 "$@" ;;
  *)       echo "未知 JOB=${JOB}（figure2|pvo|sweep|diag|probe|shift）" >&2; exit 1 ;;
esac
