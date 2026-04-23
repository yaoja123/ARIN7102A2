#!/usr/bin/env python
"""Run Q3 EMNIST GPU experiments outside Jupyter.

This script is intended for HKU GPU Farm runs. It trains stronger candidates
under the assignment's 100,000-parameter limit and writes result files that can
be merged back into Q3.ipynb later.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, Normalize, RandomAffine, RandomRotation, ToTensor


NUM_CLASSES = 47


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n, val_frac, seed):
    n_val = int(val_frac * n)
    np.random.seed(seed)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield tuple(x.to(self.device, non_blocking=True) for x in batch)

    def __len__(self):
        return len(self.dl)


class ImageClassifierNet(nn.Module):
    def __init__(self, n_channels=3):
        super(ImageClassifierNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.05):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class TinyResNet(nn.Module):
    def __init__(self, n_channels=3):
        super(TinyResNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block = ResidualBlock(36, dropout=0.05)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(36 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv2(x)
        x = self.block(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def make_transforms():
    basic_transform = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ])
    mild_aug_transform = Compose([
        RandomRotation(10),
        RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ])
    return basic_transform, mild_aug_transform


def make_loaders(args, train_transform, eval_transform, device):
    train_dataset = EMNIST(args.data_dir, download=True, train=True, split="balanced", transform=train_transform)
    eval_train_dataset = EMNIST(args.data_dir, download=True, train=True, split="balanced", transform=eval_transform)
    test_dataset = EMNIST(args.data_dir, download=True, train=False, split="balanced", transform=eval_transform)

    train_indices, val_indices = split_indices(len(eval_train_dataset), args.val_frac, args.seed)
    full_indices, _ = split_indices(len(eval_train_dataset), 0, args.seed)

    train_dl = DataLoader(
        train_dataset,
        args.batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_dl = DataLoader(
        eval_train_dataset,
        args.batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    full_dl = DataLoader(
        train_dataset,
        args.batch_size,
        sampler=SubsetRandomSampler(full_indices),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_dl = DataLoader(
        test_dataset,
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    return (
        DeviceDataLoader(train_dl, device),
        DeviceDataLoader(val_dl, device),
        DeviceDataLoader(full_dl, device),
        DeviceDataLoader(test_dl, device),
    )


def accuracy_from_outputs(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return (preds == labels).sum().item(), labels.size(0)


def evaluate(model, data_loader, loss_fn=None):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            if loss_fn is not None:
                total_loss += loss_fn(outputs, labels).item() * images.size(0)
            correct, count = accuracy_from_outputs(outputs, labels)
            total_correct += correct
            total += count

    avg_loss = total_loss / total if loss_fn is not None and total else None
    accuracy = total_correct / total if total else 0.0
    return avg_loss, accuracy


def train_epochs(model, train_dl, val_dl, loss_fn, optimizer, n_epochs, label):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(n_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_dl:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct, count = accuracy_from_outputs(outputs, labels)
            train_correct += correct
            train_total += count

        train_losses.append(train_loss / train_total)
        train_accs.append(train_correct / train_total)

        if val_dl is not None:
            val_loss, val_acc = evaluate(model, val_dl, loss_fn)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(
                f"{label} Epoch [{epoch + 1}/{n_epochs}], "
                f"train_loss: {train_losses[-1]:.4f}, val_loss: {val_loss:.4f}, "
                f"train_acc: {train_accs[-1]:.4f}, val_acc: {val_acc:.4f}",
                flush=True,
            )
        else:
            print(
                f"{label} Epoch [{epoch + 1}/{n_epochs}], "
                f"train_loss: {train_losses[-1]:.4f}, train_acc: {train_accs[-1]:.4f}",
                flush=True,
            )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accs,
        "val_accuracies": val_accs,
    }


def make_adamw(model, lr, weight_decay):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def load_baseline_checkpoint(path, device):
    try:
        loaded = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location=device)

    if isinstance(loaded, nn.Module):
        return loaded.to(device)

    model = ImageClassifierNet().to(device)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        model.load_state_dict(loaded["state_dict"])
    elif isinstance(loaded, dict):
        model.load_state_dict(loaded)
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(loaded)}")
    return model


def run_training_experiment(args, config, loaders, device, output_dir):
    train_dl, val_dl, full_dl, test_dl = loaders
    set_seed(args.seed)

    model = config["model_class"]().to(device)
    params = parameter_count(model)
    if params > 100000:
        raise ValueError(f"{config['name']} has {params} params, above the 100000 limit")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = make_adamw(model, config["lr"], config["weight_decay"])

    start_time = time.time()
    print(f"\n=== {config['name']} ===", flush=True)
    print(f"params: {params}", flush=True)
    print(
        f"optimizer: AdamW lr={config['lr']} weight_decay={config['weight_decay']} "
        f"label_smoothing={config['label_smoothing']} augmentation={config['augmentation']}",
        flush=True,
    )

    history = train_epochs(model, train_dl, val_dl, loss_fn, optimizer, args.epochs, config["name"])

    if args.finetune_epochs > 0:
        finetune_optimizer = make_adamw(model, config["finetune_lr"], config["weight_decay"])
        finetune_history = train_epochs(
            model,
            full_dl,
            None,
            loss_fn,
            finetune_optimizer,
            args.finetune_epochs,
            config["name"] + " finetune",
        )
    else:
        finetune_history = {"train_losses": [], "val_losses": [], "train_accuracies": [], "val_accuracies": []}

    test_loss, test_acc = evaluate(model, test_dl, loss_fn)
    elapsed = time.time() - start_time

    checkpoint_path = output_dir / f"{config['name']}.pth"
    checkpoint_config = {
        "name": config["name"],
        "model_class": config["model_class"].__name__,
        "augmentation": config["augmentation"],
        "lr": config["lr"],
        "finetune_lr": config["finetune_lr"],
        "weight_decay": config["weight_decay"],
        "label_smoothing": config["label_smoothing"],
    }
    torch.save(
        {
            "model_class": config["model_class"].__name__,
            "state_dict": model.state_dict(),
            "config": checkpoint_config,
            "params": params,
            "test_accuracy": test_acc,
        },
        checkpoint_path,
    )

    val_accs = history["val_accuracies"]
    result = {
        "name": config["name"],
        "model_class": config["model_class"].__name__,
        "params": params,
        "augmentation": config["augmentation"],
        "optimizer": "AdamW",
        "lr": config["lr"],
        "finetune_lr": config["finetune_lr"],
        "weight_decay": config["weight_decay"],
        "label_smoothing": config["label_smoothing"],
        "epochs": args.epochs,
        "finetune_epochs": args.finetune_epochs,
        "final_val_accuracy": val_accs[-1] if val_accs else None,
        "best_val_accuracy": max(val_accs) if val_accs else None,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": elapsed,
        "history": history,
        "finetune_history": finetune_history,
    }

    print(f"{config['name']} Test Accuracy = {test_acc:.4f}", flush=True)
    return result


def write_results(results, json_path, csv_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fields = [
        "name",
        "model_class",
        "params",
        "augmentation",
        "optimizer",
        "lr",
        "finetune_lr",
        "weight_decay",
        "label_smoothing",
        "epochs",
        "finetune_epochs",
        "final_val_accuracy",
        "best_val_accuracy",
        "test_accuracy",
        "checkpoint",
        "elapsed_seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field) for field in fields})


def parse_args():
    parser = argparse.ArgumentParser(description="Run Q3 EMNIST GPU experiments.")
    parser.add_argument("--data-dir", default="MNIST_data/")
    parser.add_argument("--baseline-model", default="model.pth")
    parser.add_argument("--output-dir", default="q3_experiment_models")
    parser.add_argument("--json-out", default="q3_gpu_results.json")
    parser.add_argument("--csv-out", default="q3_gpu_results.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Run one short epoch and no finetune.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.finetune_epochs = 0

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    basic_transform, mild_aug_transform = make_transforms()
    basic_loaders = make_loaders(args, basic_transform, basic_transform, device)
    aug_loaders = make_loaders(args, mild_aug_transform, basic_transform, device)

    results = []

    baseline_path = Path(args.baseline_model)
    if baseline_path.exists():
        print(f"\n=== baseline_checkpoint_eval ({baseline_path}) ===", flush=True)
        baseline_model = load_baseline_checkpoint(baseline_path, device)
        params = parameter_count(baseline_model)
        test_loss, test_acc = evaluate(baseline_model, basic_loaders[3], nn.CrossEntropyLoss())
        print(f"baseline checkpoint params: {params}", flush=True)
        print(f"baseline checkpoint Test Accuracy = {test_acc:.4f}", flush=True)
        results.append({
            "name": "baseline_checkpoint_eval",
            "model_class": baseline_model.__class__.__name__,
            "params": params,
            "augmentation": "none",
            "optimizer": "checkpoint",
            "lr": None,
            "finetune_lr": None,
            "weight_decay": None,
            "label_smoothing": None,
            "epochs": 0,
            "finetune_epochs": 0,
            "final_val_accuracy": None,
            "best_val_accuracy": None,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "checkpoint": str(baseline_path),
            "elapsed_seconds": 0.0,
            "history": {},
            "finetune_history": {},
        })
    else:
        print(f"baseline checkpoint not found: {baseline_path}", flush=True)

    configs = [
        {
            "name": "baseline_adamw_label_smoothing",
            "model_class": ImageClassifierNet,
            "augmentation": "none",
            "lr": 0.001,
            "finetune_lr": 0.0005,
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
            "loaders": basic_loaders,
        },
        {
            "name": "tiny_resnet_adamw_label_smoothing",
            "model_class": TinyResNet,
            "augmentation": "none",
            "lr": 0.001,
            "finetune_lr": 0.0005,
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
            "loaders": basic_loaders,
        },
        {
            "name": "tiny_resnet_aug_adamw_label_smoothing",
            "model_class": TinyResNet,
            "augmentation": "rotation10_translate008_scale095_105",
            "lr": 0.001,
            "finetune_lr": 0.0005,
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
            "loaders": aug_loaders,
        },
    ]

    for config in configs:
        result = run_training_experiment(args, config, config["loaders"], device, output_dir)
        result.pop("loaders", None)
        results.append(result)
        write_results(results, args.json_out, args.csv_out)

    print("\n=== Summary ===", flush=True)
    for result in results:
        print(
            f"{result['name']}: params={result['params']} "
            f"best_val={result.get('best_val_accuracy')} test={result['test_accuracy']:.4f}",
            flush=True,
        )
    print(f"wrote {args.json_out} and {args.csv_out}", flush=True)


if __name__ == "__main__":
    main()
