import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from robot_ai.common.device import get_device
from robot_ai.ml.crack_dataset import CrackDataset, build_transforms


def set_seed(seed: int) -> None:
    """
    Fix the random seeds of Python, NumPy and PyTorch.

    :param seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet18 classifier with two outputs (no crack / crack).

    :param pretrained: Whether to load the ImageNet weights.
    :return: The model.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    # ImageNet の 1000 クラスの出力層を、亀裂なし／ありの 2 クラスに付け替える
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def compute_metrics(labels, preds) -> dict:
    """
    Compute classification metrics with crack (label 1) as the positive class.

    :param labels: True labels.
    :param preds: Predicted labels.
    :return: Dictionary with accuracy, precision, recall, f1 and the confusion matrix.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        # 行が正解、列が予測（[[TN, FP], [FN, TP]]）
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """
    Run one pass over the loader, training the model when an optimizer is given.

    :param model: The model.
    :param loader: DataLoader to iterate over.
    :param criterion: Loss function.
    :param device: Device to run on.
    :param optimizer: Optimizer; evaluation only when ``None``.
    :param scaler: Gradient scaler for mixed precision training.
    :return: Tuple (mean loss, true labels, predicted labels).
    """
    train = optimizer is not None
    model.train(train)
    total_loss, labels, preds = 0.0, [], []
    with torch.set_grad_enabled(train):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            # 半精度で計算して VRAM と時間を節約する
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(targets)
            labels += targets.tolist()
            preds += outputs.argmax(dim=1).tolist()
    return total_loss / len(loader.dataset), labels, preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 on SDNET2018 walls.")
    parser.add_argument("--csv", type=Path, default=Path("data/splits/sdnet_wall.csv"))
    parser.add_argument("--root", type=Path, default=Path("data/raw/SDNET2018"))
    parser.add_argument("--out", type=Path, default=Path("models/crack_resnet18"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    args.out.mkdir(parents=True, exist_ok=True)

    train_set = CrackDataset(args.csv, args.root, "train", build_transforms(train=True))
    val_set = CrackDataset(args.csv, args.root, "val", build_transforms(train=False))
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)

    model = build_model().to(device)
    # 亀裂ありは全体の約 2 割なので、少ないクラスの誤りを重く数える
    counts = train_set.items["label"].value_counts().sort_index().to_numpy()
    class_weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, labels, preds = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        metrics = compute_metrics(labels, preds)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        row.update({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        history.append(row)
        print(
            f"epoch {epoch:2d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"val_f1 {metrics['f1']:.4f}  ({time.time() - start:.0f} s)"
        )

        # 亀裂は少ないので、正解率ではなく F1 で一番よいエポックを選ぶ
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), args.out / "best.pt")

    pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    if device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print(f"best val f1 {best_f1:.4f}, saved to {args.out}")


if __name__ == "__main__":
    main()
