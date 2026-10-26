import math

from wall_inspection_control.goal_follower import GoalFollower


def test_arrived_when_close():
    follower = GoalFollower()
    linear, angular, arrived = follower.update(0.0, 1.0, math.pi / 2, 0.01, 1.01)
    assert arrived
    assert linear == 0.0
    assert angular == 0.0


def test_turns_in_place_first():
    follower = GoalFollower()
    # 上向きで、目標は左（-u 方向）にある
    linear, angular, arrived = follower.update(0.0, 1.0, math.pi / 2, -1.0, 1.0)
    assert not arrived
    assert linear == 0.0
    assert angular > 0.0


def test_drives_when_facing_goal():
    follower = GoalFollower()
    linear, angular, arrived = follower.update(0.0, 1.0, math.pi / 2, 0.0, 2.0)
    assert not arrived
    assert linear > 0.0
    assert abs(angular) < 1e-9


def test_turns_again_when_heading_error_grows():
    follower = GoalFollower()
    linear, angular, arrived = follower.update(0.0, 1.0, math.pi / 2, 0.0, 2.0)
    assert linear > 0.0
    assert not follower.turning

    # 目標が右に変わる：ずれは -90° なので、その場で右回り（負）に回る
    linear, angular, arrived = follower.update(0.0, 1.0, math.pi / 2, 1.0, 1.0)
    assert not arrived
    assert linear == 0.0
    assert angular < 0.0
    assert follower.turning
