import math


def normalize_angle(angle):
    """Wrap an angle to the range [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def wall_pose_from_world(position, orientation):
    """
    Convert a world pose into a 2D pose on the wall surface.

    The wall coordinates are defined as seen from the front of the wall:
    +u points right (world -y), +v points up (world +z), and theta is
    measured counterclockwise from the +u axis.

    :param position: Position in the world frame (geometry_msgs/Point).
    :param orientation: Orientation in the world frame
        (geometry_msgs/Quaternion).
    :return: Tuple (u, v, theta) in meters and radians.
    """
    # 壁を正面から見て右を +u、上を +v にし、左回りを正の向きにそろえる
    u = -position.y
    v = position.z

    # 回転行列の1列目：車体の前方向（x 軸）をワールド座標で表したもの
    q = orientation
    forward_y = 2.0 * (q.x * q.y + q.w * q.z)
    forward_z = 2.0 * (q.x * q.z - q.w * q.y)
    theta = math.atan2(forward_z, -forward_y)
    return u, v, theta
