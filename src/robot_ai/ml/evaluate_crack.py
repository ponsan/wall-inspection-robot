import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from robot_ai.common.device import get_device
from robot_ai.ml.crack_dataset import CrackDataset, build_transforms
from robot_ai.ml.train_crack import build_model, compute_metrics


def predict_crack_prob(model, loader, device) -> list[float]:
    """
    Predict the probability of a crack for every image in the loader.

    :param model: Trained two-class model.
    :param loader: DataLoader that yields (images, labels).
    :param device: Device to run on.
    :return: Crack probabilities in the order of the loader.
    """
    model.eval()
    probs = []
    with torch.no_grad():
        for images, _ in loader:
            outputs = model(images.to(device))
            # 列 1 が亀裂ありの確率
            probs += torch.softmax(outputs, dim=1)[:, 1].tolist()
    return probs


def save_confusion_matrix(matrix, path: Path) -> None:
    """
    Save a confusion matrix as a PNG image.

    :param matrix: Confusion matrix as ``[[TN, FP], [FN, TP]]``.
    :param path: Output file path.
    """
    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot()
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.array(matrix), display_labels=["no crack", "crack"]
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the crack classifier.")
    parser.add_argument("--csv", type=Path, default=Path("data/splits/sdnet_wall.csv"))
    parser.add_argument("--root", type=Path, default=Path("data/raw/SDNET2018"))
    parser.add_argument("--weights", type=Path, default=Path("models/crack_resnet18/best.pt"))
    parser.add_argument("--out", type=Path, default=Path("results/crack_resnet18"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    args.out.mkdir(parents=True, exist_ok=True)

    dataset = CrackDataset(args.csv, args.root, args.split, build_transforms(train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))

    probs = predict_crack_prob(model, loader, device)
    labels = dataset.items["label"].tolist()
    preds = [int(p >= args.threshold) for p in probs]
    metrics = compute_metrics(labels, preds)
    metrics.update({"split": args.split, "threshold": args.threshold, "num_images": len(labels)})

    prefix = args.out / args.split
    Path(f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
    predictions = dataset.items[["path", "photo_id", "label"]].copy()
    predictions["crack_prob"] = probs
    predictions["pred"] = preds
    predictions.to_csv(f"{prefix}_predictions.csv", index=False)
    save_confusion_matrix(metrics["confusion_matrix"], Path(f"{prefix}_confusion_matrix.png"))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
