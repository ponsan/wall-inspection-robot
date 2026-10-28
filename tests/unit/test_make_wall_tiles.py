import pytest

pd = pytest.importorskip("pandas")
from robot_ai.ml.make_wall_tiles import choose_tiles  # noqa: E402


def make_images():
    rows = []
    for split in ["train", "test"]:
        for i in range(20):
            rows.append({"path": f"{split}-{i}.jpg", "label": int(i < 5), "split": split})
    return pd.DataFrame(rows)


def test_uses_only_test_images():
    cells = [(u, 0.5) for u in range(6)]

    tiles = choose_tiles(make_images(), cells, num_crack=2, seed=0)

    assert tiles["source"].str.startswith("test-").all()
    assert tiles["source"].is_unique


def test_places_requested_number_of_cracks():
    cells = [(u, 0.5) for u in range(6)]

    tiles = choose_tiles(make_images(), cells, num_crack=2, seed=0)

    assert len(tiles) == 6
    assert tiles["label"].sum() == 2
    assert list(tiles["u"]) == [0, 1, 2, 3, 4, 5]
