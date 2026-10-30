import csv
import math
from pathlib import Path
import signal
import threading

from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray
from wall_inspection_control.wall_geometry import wall_pose_from_world
from wall_inspection_perception.inspection_map import make_map_markers


def image_to_array(msg):
    """
    Convert an rgb8 image message into a NumPy array.

    :param msg: sensor_msgs/msg/Image with rgb8 encoding.
    :return: Array with shape (height, width, 3).
    """
    if msg.encoding != 'rgb8':
        raise ValueError(f'Unsupported encoding: {msg.encoding}')
    # 行の末尾に詰め物（step > width * 3）があっても正しく切り出す
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    return rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)


class InspectorNode(Node):
    """Save and classify a camera image at every inspection request."""

    def __init__(self):
        """Set up parameters, the output directory and the subscribers."""
        super().__init__('inspector_node')
        self.declare_parameter('output_dir', 'data/inspection')
        self.declare_parameter('settle_time', 0.3)
        self.declare_parameter('weights', 'models/crack_resnet18/best.pt')
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('wall_x', 2.0)
        self.declare_parameter('cell_size', 0.4)

        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.settle_time = Duration(seconds=self.get_parameter('settle_time').value)
        self.threshold = self.get_parameter('threshold').value
        self.csv_path = self.output_dir / 'captures.csv'
        with self.csv_path.open('w', newline='') as f:
            csv.writer(f).writerow(
                ['goal_u', 'goal_v', 'u', 'v', 'theta', 'image', 'crack_prob', 'crack'])

        # torch は venv にしかないので、image_to_array のテストを system の Python で通すためにここで import する
        from robot_ai.ml.crack_classifier import CrackClassifier
        self.classifier = CrackClassifier(Path(self.get_parameter('weights').value))
        self.get_logger().info(f'Crack classifier on {self.classifier.device}')

        self.pose = None
        self.request = None
        self.results = []

        # 後から起動した RViz にも最新の地図が届くように、最後の 1 件を残す
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(MarkerArray, 'inspection_map', map_qos)

        self.create_subscription(PointStamped, 'inspect_request', self.on_request, 10)
        self.create_subscription(Odometry, 'ground_truth', self.on_ground_truth, 10)
        self.create_subscription(Image, 'camera/image_raw', self.on_image, 10)

    def on_ground_truth(self, msg):
        """Store the latest pose converted into wall coordinates."""
        pose = msg.pose.pose
        self.pose = wall_pose_from_world(pose.position, pose.orientation)

    def on_request(self, msg):
        """Remember the goal to inspect and when the robot stopped there."""
        self.request = msg

    def on_image(self, msg):
        """Save the first image taken after the robot has settled."""
        if self.request is None or self.pose is None:
            return
        # 止まった直後の画像は動いている途中のものがあるので、少し待ってから使う
        ready = Time.from_msg(self.request.header.stamp) + self.settle_time
        if Time.from_msg(msg.header.stamp) < ready:
            return

        goal_u, goal_v = self.request.point.x, self.request.point.y
        self.request = None
        name = f'u{goal_u:+.2f}_v{goal_v:.2f}.png'
        image = image_to_array(msg)
        PILImage.fromarray(image).save(self.output_dir / name)
        crack_prob = self.classifier.predict(image)
        crack = int(crack_prob >= self.threshold)

        u, v, theta = self.pose
        with self.csv_path.open('a', newline='') as f:
            csv.writer(f).writerow(
                [goal_u, goal_v, f'{u:.4f}', f'{v:.4f}', f'{theta:.4f}', name,
                 f'{crack_prob:.4f}', crack])
        self.get_logger().info(
            f'{name} at ({u:.3f}, {v:.3f}, {math.degrees(theta):.0f} deg): '
            f'crack_prob {crack_prob:.3f} -> {"CRACK" if crack else "ok"}')

        self.results.append((goal_u, goal_v, crack_prob))
        self.map_pub.publish(make_map_markers(
            self.results, self.threshold,
            self.get_parameter('frame_id').value,
            self.get_parameter('wall_x').value,
            self.get_parameter('cell_size').value))


def main(args=None):
    # scan_node と同じく、Ctrl+C が 2 回届いても Traceback を出さずに終わる
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = InspectorNode()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    try:
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
