import math

from geometry_msgs.msg import Point, Quaternion
import pytest
from wall_inspection_control.wall_geometry import normalize_angle, wall_pose_from_world


def test_initial_pose_faces_up():
    # ワールドの include と同じ姿勢（ピッチ −90°）
    position = Point(x=1.928, y=0.0, z=1.0)
    orientation = Quaternion(x=0.0, y=-0.7071068, z=0.0, w=0.7071068)
    u, v, theta = wall_pose_from_world(position, orientation)
    assert u == pytest.approx(0.0)
    assert v == pytest.approx(1.0)
    assert theta == pytest.approx(math.pi / 2)


def test_facing_right():
    position = Point(x=1.928, y=-0.5, z=2.0)
    orientation = Quaternion(x=0.5, y=-0.5, z=-0.5, w=0.5)
    u, v, theta = wall_pose_from_world(position, orientation)
    assert u == pytest.approx(0.5)
    assert v == pytest.approx(2.0)
    assert theta == pytest.approx(0.0)


def test_normalize_angle():
    assert normalize_angle(0) == pytest.approx(0)
    assert normalize_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
