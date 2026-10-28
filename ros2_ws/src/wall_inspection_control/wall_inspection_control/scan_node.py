from collections import deque
import signal
import threading

from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from wall_inspection_control.goal_follower import GoalFollower
from wall_inspection_control.wall_geometry import wall_pose_from_world
from wall_inspection_control.zigzag import make_zigzag_goals


class ScanNode(Node):
    """Scan the wall by visiting zigzag goals one by one."""

    def __init__(self):
        """Set up parameters, the goal queue, the publisher, the subscriber and the timer."""
        super().__init__('scan_node')
        self.declare_parameter('u_min', -1.5)
        self.declare_parameter('u_max', 1.5)
        self.declare_parameter('v_min', 0.5)
        self.declare_parameter('v_max', 2.5)
        self.declare_parameter('step', 0.5)
        self.declare_parameter('dwell_time', 1.0)

        goals = make_zigzag_goals(
            self.get_parameter('u_min').value,
            self.get_parameter('u_max').value,
            self.get_parameter('v_min').value,
            self.get_parameter('v_max').value,
            self.get_parameter('step').value)
        self.goals = deque(goals)
        self.num_goals = len(goals)
        self.dwell_time = Duration(seconds=self.get_parameter('dwell_time').value)

        self.follower = GoalFollower()
        self.pose = None
        self.state = 'MOVE'
        self.dwell_start = None

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.inspect_pub = self.create_publisher(PointStamped, 'inspect_request', 10)
        self.create_subscription(Odometry, 'ground_truth', self.on_ground_truth, 10)
        self.create_timer(0.05, self.on_timer)
        self.get_logger().info(f'Scanning {self.num_goals} goals')

    def on_ground_truth(self, msg):
        """Store the latest pose converted into wall coordinates."""
        pose = msg.pose.pose
        self.pose = wall_pose_from_world(pose.position, pose.orientation)

    def on_timer(self):
        """Advance the scan by one control step."""
        # 真値が届く前は位置がわからないので動かさない
        if self.state == 'DONE' or self.pose is None:
            return
        if self.state == 'MOVE':
            self.move_to_goal()
        elif self.state == 'DWELL':
            self.dwell()

    def move_to_goal(self):
        """Publish one velocity command toward the current goal."""
        u, v, theta = self.pose
        goal_u, goal_v = self.goals[0]
        linear, angular, arrived = self.follower.update(u, v, theta, goal_u, goal_v)
        # TrackedVehicle は最後の指令を保持するので、到達時の 0 も必ず送る
        self.publish_cmd(linear, angular)

        if arrived:
            index = self.num_goals - len(self.goals) + 1
            self.get_logger().info(
                f'Goal {index}/{self.num_goals} ({goal_u:.2f}, {goal_v:.2f}) '
                f'reached at ({u:.3f}, {v:.3f})')
            self.state = 'DWELL'
            self.dwell_start = self.get_clock().now()
            self.request_inspection(goal_u, goal_v)

    def dwell(self):
        """Wait at the goal, then move on to the next goal."""
        if self.get_clock().now() - self.dwell_start < self.dwell_time:
            return
        self.goals.popleft()
        if not self.goals:
            self.state = 'DONE'
            self.get_logger().info('Scan finished')
            return
        # 前の目標での回転・前進の状態を持ち越さない
        self.follower = GoalFollower()
        self.state = 'MOVE'

    def request_inspection(self, goal_u, goal_v):
        """Ask the inspector to inspect the goal where the robot has stopped."""
        msg = PointStamped()
        # 止まった時刻を入れ、検査側がこれより後の画像を使えるようにする
        msg.header.stamp = self.dwell_start.to_msg()
        msg.header.frame_id = 'wall'
        msg.point.x = goal_u
        msg.point.y = goal_v
        self.inspect_pub.publish(msg)

    def publish_cmd(self, linear, angular):
        """Publish a velocity command."""
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)


def main(args=None):
    # 既定の SIGINT 処理はコンテキストを先に閉じ、停止指令を送れなくなるので自分で受け取る
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ScanNode()
    stop = threading.Event()
    # Ctrl+C はターミナルと launch の両方から届くことがあるので、何回届いても印を付けるだけにする
    signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    try:
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
        node.publish_cmd(0.0, 0.0)
        node.get_logger().info('Stopped')
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
