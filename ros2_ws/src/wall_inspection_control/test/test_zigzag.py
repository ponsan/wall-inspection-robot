import math

import pytest
from wall_inspection_control.zigzag import make_zigzag_goals


def test_number_of_goals():
    goals = make_zigzag_goals(-1.5, 1.5, 0.5, 2.5, 0.5)
    assert len(goals) == 7 * 5


def test_rows_alternate_direction():
    goals = make_zigzag_goals(-1.5, 1.5, 0.5, 2.5, 0.5)
    assert goals[0] == (-1.5, 0.5)
    assert goals[6] == (1.5, 0.5)
    assert goals[7] == (1.5, 1.0)
    assert goals[13] == (-1.5, 1.0)


def test_no_long_jumps():
    goals = make_zigzag_goals(-1.5, 1.5, 0.5, 2.5, 0.5)
    for (u1, v1), (u2, v2) in zip(goals, goals[1:]):
        assert math.hypot(u2 - u1, v2 - v1) == pytest.approx(0.5)
