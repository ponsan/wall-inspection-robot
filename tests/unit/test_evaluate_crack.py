import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from robot_ai.ml.evaluate_crack import predict_crack_prob, save_confusion_matrix  # noqa: E402


class FixedModel(torch.nn.Module):
    """Model that always returns the same logits."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits)

    def forward(self, images):
        return self.logits.repeat(len(images), 1)


@pytest.mark.ml
def test_predict_crack_prob_returns_softmax_of_crack_column():
    # ロジットが等しいとき、亀裂の確率は 0.5 になる
    model = FixedModel([0.0, 0.0])
    loader = DataLoader(TensorDataset(torch.zeros(3, 3, 8, 8), torch.zeros(3)), batch_size=2)

    probs = predict_crack_prob(model, loader, torch.device("cpu"))

    assert probs == pytest.approx([0.5, 0.5, 0.5])


@pytest.mark.ml
def test_save_confusion_matrix_writes_png(tmp_path):
    path = tmp_path / "cm.png"

    save_confusion_matrix([[3, 1], [0, 2]], path)

    # PNG ファイルの先頭 8 バイトは決まっている
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.ml
def test_predict_crack_prob_is_high_when_crack_logit_is_large():
    # 亀裂ありのロジットがずっと大きいとき、確率はほぼ 1 になる
    model = FixedModel([0.0, 10.0])
    loader = DataLoader(TensorDataset(torch.zeros(3, 3, 8, 8), torch.zeros(3)), batch_size=2)

    probs = predict_crack_prob(model, loader, torch.device("cpu"))

    assert len(probs) == 3
    assert all(p > 0.99 for p in probs)
