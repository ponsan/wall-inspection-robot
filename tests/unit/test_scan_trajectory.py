import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
from robot_ai.analysis.scan_trajectory import row_deviation, trim_before_first_goal  # noqa: E402


def make_row(row_v, offset):
    u = np.linspace(-1.5, 1.5, 31)
    return pd.DataFrame({"u": u, "v": np.full_like(u, row_v + offset)})


def test_row_deviation_measures_constant_sag():
    # 行全体が 1 cm 下がっていれば mean は -0.01 になる
    trajectory = make_row(1.0, -0.01)

    (result,) = row_deviation(trajectory, [1.0], u_min=-1.5, u_max=1.5)

    assert result["mean"] == pytest.approx(-0.01)
    assert result["max_abs"] == pytest.approx(0.01)


def test_row_deviation_ignores_both_ends_of_row():
    # 行の端は次の行への移動と重なるので数えない
    trajectory = pd.concat([make_row(1.0, 0.0), pd.DataFrame({"u": [1.45], "v": [1.08]})])

    (result,) = row_deviation(trajectory, [1.0], u_min=-1.5, u_max=1.5)

    assert result["max_abs"] == pytest.approx(0.0)


def test_trim_before_first_goal_starts_at_first_goal():
    trajectory = pd.DataFrame({"u": [0.0, -0.75, -1.47, -1.5], "v": [1.0, 0.75, 0.5, 0.5]})

    trimmed = trim_before_first_goal(trajectory, (-1.5, 0.5))

    assert len(trimmed) == 2
    assert trimmed["u"].iloc[0] == pytest.approx(-1.47)
