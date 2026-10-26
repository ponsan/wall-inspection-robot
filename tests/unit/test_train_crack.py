import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("sklearn")
from robot_ai.ml.train_crack import build_model, compute_metrics  # noqa: E402


@pytest.mark.ml
def test_model_has_two_outputs():
    # 重みのダウンロードを避けるため、学習済みの重みは読み込まない
    model = build_model(pretrained=False).eval()

    with torch.no_grad():
        outputs = model(torch.zeros(2, 3, 224, 224))

    assert outputs.shape == (2, 2)


@pytest.mark.ml
def test_compute_metrics():
    labels = [0, 0, 1, 1]
    preds = [0, 1, 1, 0]

    metrics = compute_metrics(labels, preds)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


@pytest.mark.ml
def test_metrics_when_predicting_no_crack_only():
    labels = [0, 0, 0, 1]
    preds = [0, 0, 0, 0]

    metrics = compute_metrics(labels, preds)

    # 正解率は高く見えるが、亀裂を1つも見つけていない
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)
    assert metrics["confusion_matrix"] == [[3, 0], [1, 0]]
