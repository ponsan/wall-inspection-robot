from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

# ImageNet で学習済みの重みは、この平均と標準偏差で正規化した画像を前提にしている
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool, size: int = 224) -> v2.Compose:
    """
    Build the image transforms for training or evaluation.

    :param train: Whether to add random augmentation.
    :param size: Output image size in pixels.
    :return: Composed transform that returns a normalized float tensor.
    """
    steps = [v2.ToImage(), v2.Resize((size, size), antialias=True)]
    if train:
        # 亀裂の向きや明るさは撮り方で変わるので、反転と明るさの変化で増やす
        steps += [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(steps)


class CrackDataset(Dataset):
    """Wall images of SDNET2018 for one split, read from the split CSV."""

    def __init__(self, csv_path: Path, root: Path, split: str, transform=None):
        """
        Load the list of images for the given split.

        :param csv_path: CSV written by :mod:`robot_ai.ml.split_sdnet`.
        :param root: Dataset root that the paths in the CSV are relative to.
        :param split: One of ``train``, ``val`` or ``test``.
        :param transform: Transform applied to each image.
        """
        table = pd.read_csv(csv_path, dtype={"photo_id": str})
        self.items = table[table["split"] == split].reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        row = self.items.iloc[index]
        # グレースケールの画像が混じっていても 3 チャンネルにそろえる
        image = Image.open(self.root / row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])
