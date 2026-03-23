#!/usr/bin/env bash
# 在 conda 环境 torch21 下依次运行 1-Linear、2-GBDT、3-DNN、4-ORF、6-BDRL
# 使用方式: bash run_all_models_torch21.sh  或  conda run -n torch21 bash run_all_models_torch21.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$MODEL_ROOT"

# 使用 conda 环境 torch21
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate torch21
fi

echo "========== 数据路径 =========="
echo "使用 3-Replication/input_file.py 中的 input_file（默认 5-Data/simulated_data.txt）"
echo "如需更换数据，请修改: $REPO_ROOT/3-Replication/input_file.py"
echo ""

# 确保 3-Replication 在 Python 路径中，以便各子目录能 from input_file import input_file
export PYTHONPATH="${REPO_ROOT}/3-Replication:${PYTHONPATH:-}"

echo "========== 1-Linear (OLS) =========="
python 1-Linear/OLS_train_model.py
echo ""

echo "========== 2-GBDT =========="
python 2-GBDT/GBDT_train_model.py --prediction regression
echo ""

echo "========== 3-DNN =========="
python 3-DNN/NN_model_train_predict.py
echo ""

echo "========== 4-ORF =========="
python 4-ORF/ORF_train_model.py
echo ""

echo "========== 6-BDRL (ReAgent) =========="
echo "6-BDRL 需 ReAgent 依赖与 timeline 数据，在 6-BDRL/ReAgent 下执行 train.sh 或按 README 准备数据后训练。"
if [ -f "6-BDRL/ReAgent/scripts/train.sh" ]; then
  (cd 6-BDRL/ReAgent && bash scripts/train.sh) || echo "6-BDRL 未执行或未配置，请参见 6-BDRL/ReAgent/README.md"
else
  echo "跳过 6-BDRL（未找到 train.sh）"
fi

echo ""
echo "========== 全部完成 =========="
