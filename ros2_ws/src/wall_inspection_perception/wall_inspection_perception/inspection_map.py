from visualization_msgs.msg import Marker, MarkerArray

# 健全は緑、亀裂は赤（RGBA）
OK_COLOR = (0.2, 0.8, 0.2, 0.8)
CRACK_COLOR = (0.9, 0.1, 0.1, 0.8)


def make_map_markers(results, threshold, frame_id, wall_x, cell_size):
    """
    Build markers that show the crack judgement of every inspected cell.

    :param results: List of (u, v, crack_prob) in wall coordinates.
    :param threshold: Crack probability at or above which a cell is a crack.
    :param frame_id: Frame of the world coordinates.
    :param wall_x: World x of the wall surface the markers are drawn on.
    :param cell_size: Side length of a cell marker in meters.
    :return: MarkerArray with one square and one label per cell.
    """
    markers = MarkerArray()
    for i, (u, v, crack_prob) in enumerate(results):
        square = Marker()
        square.header.frame_id = frame_id
        square.ns = 'cells'
        square.id = i
        square.type = Marker.CUBE
        square.action = Marker.ADD
        # 壁座標 (u, v) はワールドの (y, z) = (-u, v)
        # ロボットの側から見えるように壁の表面より少し手前に置く
        square.pose.position.x = wall_x - 0.01
        square.pose.position.y = -u
        square.pose.position.z = v
        square.pose.orientation.w = 1.0
        square.scale.x = 0.005
        square.scale.y = cell_size
        square.scale.z = cell_size
        color = CRACK_COLOR if crack_prob >= threshold else OK_COLOR
        square.color.r, square.color.g, square.color.b, square.color.a = color

        label = Marker()
        label.header.frame_id = frame_id
        label.ns = 'labels'
        label.id = i
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = wall_x - 0.05
        label.pose.position.y = -u
        label.pose.position.z = v
        label.pose.orientation.w = 1.0
        label.scale.z = 0.08
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = f'{crack_prob:.2f}'

        markers.markers += [square, label]
    return markers
