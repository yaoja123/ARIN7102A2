# Task Log

## 2026-04-22 17:03:32 CST +0800

- Role: Codex coding agent
- Task: Implemented Q3 CNN assignment code while preserving the provided notebook structure.
- Files changed: `Q3.ipynb`, `Q3.py`
- Summary:
  - Generated `Q3.py` from `Q3.ipynb` for diff review.
  - Filled Q3 placeholder sections for validation split, batch size, CNN model, training loop, hyperparameters, finetuning settings, and test evaluation.
  - Added `# Scope: Q3 ...` comments to implemented blocks.
  - Installed missing `torchinfo` in conda environment `7606`.
- Implementation:
  - Used a simple validation split setup with `val_frac = 0.1`, `rand_seed = 42`, and `batch_size = 128`.
  - Implemented `ImageClassifierNet` as a small grayscale CNN for EMNIST balanced: two convolution layers (`1 -> 16 -> 32`) with ReLU and max pooling, followed by dropout and two linear layers ending in 47 class logits.
  - Implemented `train_model` with a standard PyTorch loop: train mode, forward pass, cross-entropy loss, backpropagation, optimizer step, and epoch-level loss/accuracy tracking.
  - Added validation evaluation inside `train_model` when `val_dl` is provided, using `model.eval()` and `torch.no_grad()`.
  - Set initial training to `CrossEntropyLoss`, `Adam`, `lr = 0.001`, `num_epochs = 10`; set full-dataset finetuning to `lr = 0.0005`, `num_epochs = 2`.
  - Implemented `evaluate` to compute test accuracy over the test dataloader without gradient tracking.
- Verification:
  - Ran `conda run -n 7606 python -m py_compile Q3.py`.
  - Confirmed no executable `YOUR CODE HERE` placeholders remain.
  - Confirmed notebook outputs were not added.
  - Confirmed model parameter count is 56,559, under the 100,000 limit.
- Not run:
  - Did not run `Q3.ipynb`.
  - Did not run dataset download or model training.

## 2026-04-23 17:46:53 HKT

- Role: Codex coding agent
- Task: Closed the Q3 GPU experiment line and recorded the final selected result in the notebook.
- Files changed: `Q3.ipynb`, `Q3.py`, `Q3_gpu_experiments.py`, `run_q3_gpu_experiments.sbatch`, `q3_gpu_results.json`, `q3_gpu_results.csv`, `q3_gpu_ablation_results.json`, `q3_gpu_ablation_results.csv`, `q3_experiment_models/`, `q3_ablation_models/`
- Summary:
  - Added existing HKU GPU Farm experiment results into `Q3.ipynb` and regenerated `Q3.py` for diff review.
  - Selected `TinyResNet AvgPool variant` as the final best model.
  - Marked this Q3 improvement line as complete and ready to switch away from Q3.
- Implementation:
  - Kept all trained models under the 100,000-parameter assignment limit.
  - Compared the baseline checkpoint, AdamW/label-smoothing baseline, TinyResNet variants, dropout ablations, label-smoothing ablations, and AvgPool downsampling variant.
  - Final selected checkpoint: `q3_ablation_models/tiny_resnet_avgpool_variant.pth`.
  - Final selected result: 89,615 parameters, validation accuracy `0.8848`, test accuracy `0.8834`.
  - Improvement over original baseline checkpoint: test accuracy `0.8307` to `0.8834`, about `+5.26` absolute percentage points.
  - Noted that the mild augmentation run underperformed, so the final selected model uses no augmentation.
- Verification:
  - Confirmed `Q3.ipynb` is valid JSON.
  - Regenerated `Q3.py` from `Q3.ipynb` using conda environment `7606`.
  - Ran `conda run -n 7606 python -m py_compile Q3.py`.
  - Confirmed the final checkpoint path is present in the local result files.
- Not run:
  - Did not rerun `Q3.ipynb`.
  - Did not retrain models after recording the completed GPU Farm results.
