from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from wall_inspection_control.goal_follower import GoalFollower
from wall_inspection_control.wall_geometry import wall_pose_from_world


class ScanNode(Node):
    """Move the crawler to a single goal on the wall."""

    def __init__(self):
        """Set up parameters, the publisher, the subscriber and the timer."""
        super().__init__('scan_node')
        self.declare_parameter('goal_u', 0.5)
        self.declare_parameter('goal_v', 1.5)
        self.goal_u = self.get_parameter('goal_u').value
        self.goal_v = self.get_parameter('goal_v').value

        self.follower = GoalFollower()
        self.pose = None
        self.done = False

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Odometry, 'ground_truth', self.on_ground_truth, 10)
        self.create_timer(0.05, self.on_timer)

    def on_ground_truth(self, msg):
        """Store the latest pose converted into wall coordinates."""
        pose = msg.pose.pose
        self.pose = wall_pose_from_world(pose.position, pose.orientation)

    def on_timer(self):
        """Publish one velocity command toward the goal."""
        # 真値が届く前は位置がわからないので動かさない
        if self.done or self.pose is None:
            return

        u, v, theta = self.pose
        linear, angular, arrived = self.follower.update(
            u, v, theta, self.goal_u, self.goal_v)

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        # TrackedVehicle は最後の指令を保持するので、到達時の 0 を必ず1回送る
        self.cmd_pub.publish(cmd)

        if arrived:
            self.done = True
            self.get_logger().info(f'Arrived at u={u:.3f}, v={v:.3f}')


def main(args=None):
    rclpy.init(args=args)
    node = ScanNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
