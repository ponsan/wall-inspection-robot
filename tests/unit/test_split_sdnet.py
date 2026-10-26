import pandas as pd
import pytest

pytest.importorskip("sklearn")
from robot_ai.ml.split_sdnet import list_wall_images, split_by_photo  # noqa: E402


def make_images(num_photos: int = 21, per_photo: int = 10) -> pd.DataFrame:
    rows = []
    for p in range(num_photos):
        for i in range(per_photo):
            rows.append({"path": f"W/X/{p}-{i}.jpg", "label": int(i < 3), "photo_id": str(p)})
    return pd.DataFrame(rows)


def test_list_wall_images(tmp_path):
    for class_dir, name in [("CW", "7001-1.jpg"), ("UW", "7001-2.jpg"), ("UW", "7002-1.jpg")]:
        (tmp_path / "W" / class_dir).mkdir(parents=True, exist_ok=True)
        (tmp_path / "W" / class_dir / name).touch()

    images = list_wall_images(tmp_path)

    assert images["path"].tolist() == ["W/CW/7001-1.jpg", "W/UW/7001-2.jpg", "W/UW/7002-1.jpg"]
    assert images["label"].tolist() == [1, 0, 0]
    assert images["photo_id"].tolist() == ["7001", "7001", "7002"]


def test_no_photo_in_two_splits():
    images = split_by_photo(make_images(), seed=0)

    assert set(images["split"]) == {"train", "val", "test"}
    assert (images.groupby("photo_id")["split"].nunique() == 1).all()


def test_same_seed_gives_same_split():
    a = split_by_photo(make_images(), seed=0)
    b = split_by_photo(make_images(), seed=0)

    assert a["split"].tolist() == b["split"].tolist()


def test_every_split_has_both_classes():
    images = split_by_photo(make_images(), seed=0)

    assert (images.groupby("split")["label"].nunique() == 2).all()
