# Q3 GPU Farm Training Rules

## 目标

在 HKU GPU Farm 上手动运行 `Q3.ipynb`，完成 EMNIST balanced CNN 训练、loss/accuracy 图、test accuracy，并保存 `model.pth`。

## 必须遵守

- 先连 HKUVPN，再 SSH 到 GPU Farm gateway。
- 不要在 gateway node 上训练；必须先进入 GPU compute node。
- 用 `nvidia-smi` 确认当前节点有 GPU。
- 不要直接长期空挂 Jupyter；跑完后 shutdown Jupyter 并退出 GPU node。
- Q3 以 `Q3.ipynb` 为主，`Q3.py` 只用于 diff/review。

## 上传文件

本地 repo 根目录：

```bash
scp -F /dev/null -o ProxyCommand=none -o ProxyJump=none \
  Q3.ipynb Q3.py TASK_LOG.md \
  u3651420@gpu2gate1.cs.hku.hk:~/arin7102_ass2/
```

如果 `scp` 被本地代理拦截，先关 Clash/TUN/System Proxy，保留 HKUVPN。

## 登录与申请 GPU

```bash
ssh u3651420@gpu2gate1.cs.hku.hk
tmux new -s q3
gpu-interactive
nvidia-smi
```

看到 GPU 信息后才继续。

## 配 Python 环境

在 GPU node 上：

```bash
cd ~/arin7102_ass2

conda create -n arin7102-q3 python=3.11 -y
conda activate arin7102-q3

pip install numpy matplotlib torchinfo jupyterlab ipykernel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -m ipykernel install --user --name arin7102-q3 --display-name "arin7102-q3"
```

检查 CUDA：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

`torch.cuda.is_available()` 必须是 `True`。如果是 `False`，先确认你不是在 gateway。

## 启动 Jupyter

在 GPU node 上：

```bash
cd ~/arin7102_ass2
jupyter-lab --no-browser --FileContentsManager.delete_to_trash=False
```

记下 Jupyter 输出的端口和 token，例如：

```text
http://localhost:8888/lab?token=...
```

再查 GPU node IP：

```bash
hostname -I
```

## iPad Termius 端口转发

在 Termius 新建 Local Port Forwarding：

```text
Type: Local
Local host: 127.0.0.1
Local port: 8888

Destination host: <GPU node IP from hostname -I>
Destination port: 8888

SSH server:
Host: gpu2gate1.cs.hku.hk
User: u3651420
Port: 22
```

如果 Jupyter 端口是 `8889`，所有 `8888` 都改成 `8889`。

iPad Safari 打开：

```text
http://127.0.0.1:8888/lab?token=...
```

## Notebook 运行规则

打开 `Q3.ipynb`，kernel 选择 `arin7102-q3`。

按顺序运行：

1. Data preparation / dataset download cells。
2. Data exploration cells。
3. Train/validation split cells。
4. Model definition and `summary(...)` cell。
5. GPU device wrapper cells。
6. `train_model(...)` definition。
7. Initial training cells。
8. Loss/accuracy plot cells。
9. Full dataset finetuning cells。
10. Save model cell。
11. Test dataset and `evaluate(...)` cells。

如果 `view_prediction` 可视化 cell 报错，可以先跳过；核心是训练、plot、保存模型和 test accuracy。

## 当前 Q3 默认设置

- Validation fraction: `0.1`
- Random seed: `42`
- Batch size: `128`
- Model: small CNN, `1 -> 16 -> 32` conv layers, dropout `0.2`, output `47` classes
- Params: `56,559`, under the `100,000` limit
- Initial training: `num_epochs = 10`, `Adam`, `lr = 0.001`
- Full-dataset finetune: `num_epochs = 2`, `lr = 0.0005`
- Loss: `CrossEntropyLoss`

## 结束清理

Jupyter 跑完后：

1. Jupyter 页面里 shutdown server。
2. GPU node terminal 按 `Ctrl+C` 停 Jupyter。
3. 退出 GPU node：

```bash
exit
```

4. 如果在 tmux 里：

```bash
exit
```

不要让 GPU session 空挂。

## 常见问题

- `Connection closed by 127.0.0.1 port 7897`: 本地 SSH/SCP 被代理拦截，关 Clash/TUN/System Proxy，只保留 HKUVPN。
- `torch.cuda.is_available() == False`: 你可能还在 gateway，先运行 `gpu-interactive`。
- Safari 打不开 Jupyter: 检查 Termius tunnel 端口、GPU node IP、HKUVPN、Jupyter 是否仍在运行。
- `view_prediction` 报错: 跳过该 cell，继续保存模型和测试 accuracy。
