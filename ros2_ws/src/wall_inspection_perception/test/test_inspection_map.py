import pytest
from visualization_msgs.msg import Marker
from wall_inspection_perception.inspection_map import make_map_markers


def test_makes_square_and_label_per_cell():
    markers = make_map_markers(
        [(0.0, 0.5, 0.1), (0.5, 1.0, 0.9)], 0.5, 'world', 2.0, 0.4).markers

    assert [m.type for m in markers] == [
        Marker.CUBE, Marker.TEXT_VIEW_FACING, Marker.CUBE, Marker.TEXT_VIEW_FACING]
    assert [m.text for m in markers[1::2]] == ['0.10', '0.90']


def test_places_cell_in_world_coordinates():
    # 壁座標の u は右向き、ワールドの y は左向きなので符号が逆になる
    square = make_map_markers([(0.5, 1.5, 0.1)], 0.5, 'world', 2.0, 0.4).markers[0]

    assert square.header.frame_id == 'world'
    assert square.pose.position.y == pytest.approx(-0.5)
    assert square.pose.position.z == pytest.approx(1.5)
    assert square.pose.position.x < 2.0


def test_colors_crack_red_and_ok_green():
    ok, _, crack, _ = make_map_markers(
        [(0.0, 0.5, 0.49), (0.5, 0.5, 0.5)], 0.5, 'world', 2.0, 0.4).markers

    assert ok.color.g > ok.color.r
    assert crack.color.r > crack.color.g
