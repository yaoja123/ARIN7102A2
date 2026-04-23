#!/bin/bash
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=q3_experiments
#SBATCH --output=q3_experiments_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "$HOME/arin7102_ass2"

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "conda not found; activate your environment manually or edit this sbatch file" >&2
  exit 1
fi

conda activate arin7102-q3

python Q3_gpu_experiments.py \
  --mode ablation \
  --baseline-model model.pth \
  --output-dir q3_ablation_models \
  --json-out q3_gpu_ablation_results.json \
  --csv-out q3_gpu_ablation_results.csv \
  --epochs 10 \
  --finetune-epochs 2 \
  --batch-size 128 \
  --num-workers 2 \
  --device cuda
