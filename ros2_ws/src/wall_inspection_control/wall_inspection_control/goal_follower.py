import math

from wall_inspection_control.wall_geometry import normalize_angle


def _clamp(value, limit):
    return max(-limit, min(limit, value))


class GoalFollower:
    """Compute velocity commands that drive the robot to a goal on the wall."""

    def __init__(self, max_linear=0.2, max_angular=0.3, k_linear=0.8,
                 k_angular=1.0, arrive_tolerance=0.03,
                 turn_enter=0.5, turn_exit=0.05):
        """
        Initialize the follower with its limits and gains.

        :param max_linear: Maximum forward speed [m/s].
        :param max_angular: Maximum turning speed [rad/s].
        :param k_linear: Proportional gain for the distance to the goal.
        :param k_angular: Proportional gain for the heading error.
        :param arrive_tolerance: Distance at which the goal counts as reached [m].
        :param turn_enter: Heading error that switches back to turning in place [rad].
        :param turn_exit: Heading error that switches from turning to driving [rad].
        """
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.k_linear = k_linear
        self.k_angular = k_angular
        self.arrive_tolerance = arrive_tolerance
        self.turn_enter = turn_enter
        self.turn_exit = turn_exit
        self.turning = True

    def update(self, u, v, theta, goal_u, goal_v):
        """
        Compute one velocity command toward the goal.

        :param u: Current position along the wall's horizontal axis [m].
        :param v: Current position along the wall's vertical axis [m].
        :param theta: Current heading on the wall [rad].
        :param goal_u: Goal position along the horizontal axis [m].
        :param goal_v: Goal position along the vertical axis [m].
        :return: Tuple (linear, angular, arrived).
        """
        du = goal_u - u
        dv = goal_v - v
        distance = math.hypot(du, dv)
        if distance < self.arrive_tolerance:
            return 0.0, 0.0, True

        heading_error = normalize_angle(math.atan2(dv, du) - theta)
        # 閾値を2つに分け、境目で回転と前進を細かく行き来しないようにする
        if self.turning and abs(heading_error) < self.turn_exit:
            self.turning = False
        elif not self.turning and abs(heading_error) > self.turn_enter:
            self.turning = True

        angular = _clamp(self.k_angular * heading_error, self.max_angular)
        if self.turning:
            return 0.0, angular, False
        linear = min(self.max_linear, self.k_linear * distance)
        return linear, angular, False
