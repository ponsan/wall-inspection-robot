import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
from robot_ai.analysis.compare_wall_map import match_cells  # noqa: E402


def make_ground_truth():
    return pd.DataFrame({"cell": [0, 1], "u": [-0.5, 0.0], "v": [0.5, 0.5], "label": [1, 0]})


def test_match_cells_joins_by_rounded_goal():
    # 目標の座標に小さな誤差があっても同じマスとして突き合わせる
    captures = pd.DataFrame(
        {
            "goal_u": [0.0000001, -0.4999999],
            "goal_v": [0.5, 0.5000001],
            "crack_prob": [0.1, 0.9],
            "crack": [0, 1],
        }
    )

    cells = match_cells(captures, make_ground_truth())

    assert cells["pred"].tolist() == [1, 0]
    assert cells["crack_prob"].tolist() == [0.9, 0.1]


def test_match_cells_raises_when_cell_is_missing():
    captures = pd.DataFrame({"goal_u": [0.0], "goal_v": [0.5], "crack_prob": [0.1], "crack": [0]})

    with pytest.raises(ValueError, match=r"\[0\]"):
        match_cells(captures, make_ground_truth())


def test_match_cells_labels_outcomes():
    ground_truth = pd.DataFrame(
        {"cell": [0, 1, 2, 3], "u": [0.0, 0.5, 1.0, 1.5], "v": [0.5] * 4, "label": [1, 0, 0, 1]}
    )
    captures = pd.DataFrame(
        {
            "goal_u": [0.0, 0.5, 1.0, 1.5],
            "goal_v": [0.5] * 4,
            "crack_prob": [0.9, 0.1, 0.8, 0.2],
            "crack": [1, 0, 1, 0],
        }
    )

    cells = match_cells(captures, ground_truth)

    assert cells["outcome"].tolist() == ["TP", "TN", "FP", "FN"]
