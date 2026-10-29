import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from robot_ai.ml.crack_classifier import CrackClassifier  # noqa: E402
from robot_ai.ml.train_crack import build_model  # noqa: E402


@pytest.fixture
def weights(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "weights.pt"
    torch.save(build_model(pretrained=False).state_dict(), path)
    return path


@pytest.mark.ml
def test_predict_returns_probability(weights):
    classifier = CrackClassifier(weights, torch.device("cpu"))
    image = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)

    prob = classifier.predict(image)

    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


@pytest.mark.ml
def test_predict_is_repeatable(weights):
    # 評価モードなので、同じ画像なら何度でも同じ確率になる
    classifier = CrackClassifier(weights, torch.device("cpu"))
    image = np.full((256, 256, 3), 128, dtype=np.uint8)

    assert classifier.predict(image) == classifier.predict(image)
