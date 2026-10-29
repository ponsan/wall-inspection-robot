from pathlib import Path

import numpy as np
import torch

from robot_ai.common.device import get_device
from robot_ai.ml.crack_dataset import build_transforms
from robot_ai.ml.train_crack import build_model


class CrackClassifier:
    """Crack classifier that predicts on one camera image at a time."""

    def __init__(self, weights: Path, device: torch.device | None = None):
        """
        Load the trained weights.

        :param weights: Path to the weights saved by :mod:`robot_ai.ml.train_crack`.
        :param device: Device to run on. Use CUDA when available if omitted.
        """
        self.device = device or get_device()
        self.model = build_model(pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        self.model.eval()
        # 学習時の評価と同じ前処理にしないと、確率の意味が変わる
        self.transform = build_transforms(train=False)

    def predict(self, image: np.ndarray) -> float:
        """
        Predict the probability of a crack in an RGB image.

        :param image: Array with shape (height, width, 3) and dtype uint8.
        :return: Probability of a crack between 0 and 1.
        """
        batch = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch)
        return torch.softmax(logits, dim=1)[0, 1].item()
