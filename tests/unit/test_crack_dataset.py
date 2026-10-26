import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image  # noqa: E402

from robot_ai.ml.crack_dataset import CrackDataset, build_transforms  # noqa: E402


@pytest.mark.ml
def test_transform_output_shape():
    image = Image.new("RGB", (256, 256), color=(128, 128, 128))

    tensor = build_transforms(train=True)(image)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


@pytest.mark.ml
def test_dataset_reads_only_its_split(tmp_path):
    (tmp_path / "W" / "CW").mkdir(parents=True)
    Image.new("RGB", (256, 256)).save(tmp_path / "W" / "CW" / "7001-1.jpg")
    # グレースケールでも 3 チャンネルになるか確かめる
    Image.new("L", (256, 256)).save(tmp_path / "W" / "CW" / "7002-1.jpg")
    csv_path = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "path": ["W/CW/7001-1.jpg", "W/CW/7002-1.jpg"],
            "label": [1, 1],
            "photo_id": ["7001", "7002"],
            "split": ["train", "test"],
        }
    ).to_csv(csv_path, index=False)

    dataset = CrackDataset(csv_path, tmp_path, "test", build_transforms(train=False))
    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 224, 224)
    assert label == 1


@pytest.mark.ml
def test_eval_transform_is_deterministic():
    # 左右で色が違う画像にして、反転すれば違いが出るようにする
    image = Image.new("RGB", (256, 256), color=(255, 255, 255))
    image.paste((0, 0, 0), (0, 0, 128, 256))
    transform = build_transforms(train=False)

    # ランダムな処理が紛れ込んでいれば、どこかで違いが出るように何度も比べる
    first = transform(image)
    for _ in range(10):
        assert torch.equal(first, transform(image))
