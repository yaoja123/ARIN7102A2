#!/bin/bash
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=q4_sentiment
#SBATCH --output=q4_gpu_training_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "$HOME/arin7102_ass2"

echo "job started: $(date)"
echo "host: $(hostname)"
echo "python path: $(which python)"
python --version

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

python Q4/q4_gpu_train.py \
  --model all \
  --epochs 5 \
  --batch-size 128 \
  --num-workers 2 \
  --device cuda

echo "job finished: $(date)"
