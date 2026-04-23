#!/usr/bin/env python
"""Run Q4 sentiment models on a GPU Farm batch node.

This script mirrors the completed Q4 notebook logic, but writes checkpoints,
metrics, and plots for a non-interactive GPU Farm run.
"""

import argparse
import collections
import csv
import json
import os
import random
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"
PREDICTION_SAMPLES = [
    "this movie is so great",
    "this movie is so bad",
]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_file(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"using existing file: {path}", flush=True)
        return path

    tmp_path = path.with_suffix(path.suffix + ".part")
    print(f"downloading {url} -> {path}", flush=True)
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp_path.replace(path)
    return path


def ensure_imdb(data_dir):
    data_dir = Path(data_dir)
    imdb_dir = data_dir / "aclImdb"
    if imdb_dir.exists():
        print(f"using existing IMDb data: {imdb_dir}", flush=True)
        return imdb_dir

    archive = download_file(IMDB_URL, data_dir / "aclImdb_v1.tar.gz")
    print(f"extracting {archive}", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)
    return imdb_dir


def ensure_glove(data_dir):
    data_dir = Path(data_dir)
    glove_path = data_dir / GLOVE_FILE
    if glove_path.exists():
        print(f"using existing GloVe vectors: {glove_path}", flush=True)
        return glove_path

    archive = download_file(GLOVE_URL, data_dir / "glove.6B.zip")
    print(f"extracting {GLOVE_FILE} from {archive}", flush=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extract(GLOVE_FILE, data_dir)
    return glove_path


def read_imdb(data_dir, is_train):
    data, labels = [], []
    split = "train" if is_train else "test"
    for label in ("pos", "neg"):
        folder = Path(data_dir) / split / label
        for path in sorted(folder.iterdir()):
            with open(path, "rb") as f:
                review = f.read().decode("utf-8").replace("\n", "")
                data.append(review)
                labels.append(1 if label == "pos" else 0)
    return data, labels


def tokenize(lines):
    # Scope: Q4 GPU runner - match notebook word tokenization.
    return [line.split() for line in lines]


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        tokens = tokens or []
        reserved_tokens = reserved_tokens or []
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = list(sorted(set(
            ["<unk>"] + reserved_tokens +
            [token for token, freq in self.token_freqs if freq >= min_freq]
        )))
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)
        }

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    @property
    def unk(self):
        return self.token_to_idx["<unk>"]


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def make_loaders(data_dir, batch_size, num_steps, min_freq, num_workers, device):
    train_data = read_imdb(data_dir, is_train=True)
    test_data = read_imdb(data_dir, is_train=False)
    train_tokens = tokenize(train_data[0])
    test_tokens = tokenize(test_data[0])
    vocab = Vocab(train_tokens, min_freq=min_freq, reserved_tokens=["<pad>"])

    pad_idx = vocab["<pad>"]
    train_features = torch.tensor([
        truncate_pad(vocab[line], num_steps, pad_idx) for line in train_tokens
    ])
    test_features = torch.tensor([
        truncate_pad(vocab[line], num_steps, pad_idx) for line in test_tokens
    ])
    train_labels = torch.tensor(train_data[1])
    test_labels = torch.tensor(test_data[1])

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader, vocab


def load_glove_embeddings(glove_path, vocab, embed_size):
    vectors = torch.zeros((len(vocab), embed_size))
    token_to_idx = vocab.token_to_idx
    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            elems = line.rstrip().split(" ")
            token = elems[0]
            if token not in token_to_idx:
                continue
            values = elems[1:]
            if len(values) != embed_size:
                continue
            vectors[token_to_idx[token]] = torch.tensor(
                [float(value) for value in values]
            )
            found += 1
    print(f"loaded GloVe vectors for {found}/{len(vocab)} vocab tokens", flush=True)
    return vectors


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(
            embed_size,
            num_hiddens,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[:, 0, :], outputs[:, -1, :]), dim=1)
        return self.decoder(encoding)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList([
            nn.Conv1d(2 * embed_size, channels, kernel_size)
            for channels, kernel_size in zip(num_channels, kernel_sizes)
        ])

    def forward(self, inputs):
        embeddings = torch.cat((
            self.embedding(inputs),
            self.constant_embedding(inputs),
        ), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        convolved = [self.relu(conv(embeddings)) for conv in self.convs]
        encoding = torch.cat([
            torch.squeeze(F.max_pool1d(output, kernel_size=output.shape[-1]), dim=-1)
            for output in convolved
        ], dim=1)
        return self.decoder(self.dropout(encoding))


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
    if isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)


def apply_embeddings(model, embeds, freeze_primary):
    model.embedding.weight.data.copy_(embeds)
    model.embedding.weight.requires_grad = not freeze_primary
    if hasattr(model, "constant_embedding"):
        model.constant_embedding.weight.data.copy_(embeds)
        model.constant_embedding.weight.requires_grad = False


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).sum().item(), labels.numel()


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * y.numel()
            correct, count = accuracy(logits, y)
            total_correct += correct
            total += count
    return total_loss / total, total_correct / total


def train_model(model, train_loader, test_loader, lr, epochs, device, label):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(X)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            correct, count = accuracy(logits, y)
            total_correct += correct
            total += count

        train_loss = total_loss / total
        train_acc = total_correct / total
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)
        print(
            f"{label} epoch {epoch + 1}/{epochs}: "
            f"loss {train_loss:.4f}, train_acc {train_acc:.4f}, "
            f"test_acc {test_acc:.4f}",
            flush=True,
        )
    return history


def plot_history(history, path, title):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["train_accuracy"], label="train accuracy")
    plt.plot(epochs, history["test_accuracy"], label="test accuracy")
    plt.xlabel("Epoch")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def predict_sentiment(model, vocab, sequence, device):
    token_indices = vocab[tokenize([sequence])[0]]
    if hasattr(model, "convs"):
        min_len = max(conv.kernel_size[0] for conv in model.convs)
        if len(token_indices) < min_len:
            token_indices += [vocab["<pad>"]] * (min_len - len(token_indices))
    X = torch.tensor(token_indices, device=device).reshape(1, -1)
    model.eval()
    with torch.no_grad():
        label = torch.argmax(model(X), dim=1).item()
    return "positive" if label == 1 else "negative"


def make_model(name, vocab_size, embed_size):
    if name == "birnn":
        return BiRNN(vocab_size, embed_size, num_hiddens=100, num_layers=2), 0.01
    if name == "textcnn":
        model = TextCNN(
            vocab_size,
            embed_size,
            kernel_sizes=[3, 4, 5],
            num_channels=[100, 100, 100],
        )
        return model, 0.001
    raise ValueError(f"unknown model: {name}")


def run_one_model(name, args, vocab, embeds, train_loader, test_loader, device):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, lr = make_model(name, len(vocab), args.embed_size)
    model.apply(init_weights)
    apply_embeddings(model, embeds, freeze_primary=(name == "birnn"))
    model.to(device)

    start = time.time()
    history = train_model(model, train_loader, test_loader, lr, args.epochs, device, name)
    elapsed = time.time() - start

    predictions = {
        text: predict_sentiment(model, vocab, text, device)
        for text in PREDICTION_SAMPLES
    }
    checkpoint_path = output_dir / f"{name}_sentiment.pth"
    plot_path = output_dir / f"{name}_training.png"
    plot_history(history, plot_path, f"Q4 {name} training")

    torch.save({
        "model": name,
        "state_dict": model.state_dict(),
        "vocab_idx_to_token": vocab.idx_to_token,
        "config": {
            "embed_size": args.embed_size,
            "num_steps": args.num_steps,
            "min_freq": args.min_freq,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": lr,
        },
        "history": history,
        "predictions": predictions,
    }, checkpoint_path)

    result = {
        "model": name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": lr,
        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_test_accuracy": history["test_accuracy"][-1],
        "checkpoint": str(checkpoint_path),
        "plot": str(plot_path),
        "elapsed_seconds": elapsed,
        "predictions": predictions,
    }
    print(f"{name} final test accuracy: {result['final_test_accuracy']:.4f}", flush=True)
    return result


def write_results(results, output_dir):
    output_dir = Path(output_dir)
    json_path = output_dir / "q4_gpu_results.json"
    csv_path = output_dir / "q4_gpu_results.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fields = [
        "model",
        "epochs",
        "batch_size",
        "lr",
        "final_train_loss",
        "final_train_accuracy",
        "final_test_loss",
        "final_test_accuracy",
        "checkpoint",
        "plot",
        "elapsed_seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fields})

    print(f"wrote {json_path} and {csv_path}", flush=True)


def select_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Q4 sentiment analysis on GPU Farm.")
    parser.add_argument("--model", choices=["birnn", "textcnn", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", default="Q4/data")
    parser.add_argument("--output-dir", default="Q4/q4_gpu_outputs")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--embed-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Run one epoch for a short remote check.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1

    set_seed(args.seed)
    device = select_device(args.device)
    data_dir = Path(args.data_dir)

    print(f"python: {sys.version.split()[0]}", flush=True)
    print(f"torch: {torch.__version__}", flush=True)
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    imdb_dir = ensure_imdb(data_dir)
    glove_path = ensure_glove(data_dir)
    train_loader, test_loader, vocab = make_loaders(
        imdb_dir,
        args.batch_size,
        args.num_steps,
        args.min_freq,
        args.num_workers,
        device,
    )
    print(f"vocab size: {len(vocab)}", flush=True)
    embeds = load_glove_embeddings(glove_path, vocab, args.embed_size)

    model_names = ["birnn", "textcnn"] if args.model == "all" else [args.model]
    results = []
    for name in model_names:
        print(f"\n=== training {name} ===", flush=True)
        result = run_one_model(name, args, vocab, embeds, train_loader, test_loader, device)
        results.append(result)
        write_results(results, args.output_dir)

    print("\n=== summary ===", flush=True)
    for result in results:
        print(
            f"{result['model']}: test_acc={result['final_test_accuracy']:.4f} "
            f"checkpoint={result['checkpoint']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
