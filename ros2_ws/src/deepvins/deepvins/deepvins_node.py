"""ROS 2 inference node for the DeepVINS TorchScript model."""

from collections import deque
import time

from cv_bridge import CvBridge
from deepvins.pose_math import compose_pose
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu
import torch


IMU_SAMPLES_PER_IMAGE = 10
IMAGE_MODEL_SHAPE = (752, 480)


def stamp_to_nanoseconds(stamp):
    """Convert a builtin_interfaces/Time message to integer nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class DeepVINSNode(Node):
    """Run DeepVINS on FIFO-coupled camera and IMU samples."""

    def __init__(self):
        super().__init__('deepvins_node')

        self.declare_parameter('image_topic', '/cam0/image_raw')
        self.declare_parameter('imu_topic', '/imu0')
        self.declare_parameter('pose_topic', '/deepvins/pose')
        self.declare_parameter('model_path', '')
        self.declare_parameter('sequence_length', 8)
        self.declare_parameter('inference_rate_hz', 20.0)
        self.declare_parameter('output_frame_id', 'odom')
        self.declare_parameter('max_alignment_error_sec', 0.025)
        self.declare_parameter('backlog_warning_pairs', 2)

        self.sequence_length = self.get_parameter(
            'sequence_length').get_parameter_value().integer_value
        inference_rate_hz = self.get_parameter(
            'inference_rate_hz').get_parameter_value().double_value
        self.output_frame_id = self.get_parameter(
            'output_frame_id').get_parameter_value().string_value
        self.max_alignment_error_ns = int(
            self.get_parameter('max_alignment_error_sec')
            .get_parameter_value().double_value * 1e9)
        self.backlog_warning_pairs = self.get_parameter(
            'backlog_warning_pairs').get_parameter_value().integer_value

        if self.sequence_length <= 0:
            raise ValueError('sequence_length must be greater than zero')
        if inference_rate_hz <= 0.0:
            raise ValueError('inference_rate_hz must be greater than zero')

        model_path = self.get_parameter(
            'model_path').get_parameter_value().string_value
        if not model_path:
            raise ValueError('model_path must point to a TorchScript model')

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
        except Exception as exc:
            self.get_logger().fatal(
                f'Could not load TorchScript model from {model_path}: {exc}')
            raise

        self.bridge = CvBridge()
        self.image_queue = deque()
        self.imu_queue = deque()
        self.pending_pairs = deque()
        self.image_sequence = deque(maxlen=self.sequence_length)
        self.imu_sequence = deque(maxlen=self.sequence_length)
        self.pose = np.array(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.pose_initialized = False
        self.last_warning_times = {}

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        imu_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)

        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.image_callback,
            image_qos)
        self.imu_sub = self.create_subscription(
            Imu,
            self.get_parameter('imu_topic').value,
            self.imu_callback,
            imu_qos)
        self.pose_pub = self.create_publisher(
            PoseStamped, self.get_parameter('pose_topic').value, 10)

        self.target_period = 1.0 / inference_rate_hz
        self.timer = self.create_timer(self.target_period, self.timer_callback)
        self.get_logger().info(
            f'Loaded DeepVINS on {self.device}; inference target is '
            f'{inference_rate_hz:.2f} Hz ({self.target_period * 1e3:.1f} ms)')

    def _warn_throttled(self, key, message, interval_sec=5.0):
        now = time.monotonic()
        if now - self.last_warning_times.get(key, float('-inf')) >= interval_sec:
            self.get_logger().warning(message)
            self.last_warning_times[key] = now

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            if cv_image.size != IMAGE_MODEL_SHAPE[0] * IMAGE_MODEL_SHAPE[1]:
                self._warn_throttled(
                    'image_shape',
                    f'Ignoring image with shape {cv_image.shape}; the model expects '
                    '480x752 grayscale EuRoC images')
                return

            # Training reshapes the 480x752 image storage to 752x480 inside the
            # encoder. Do the same here while constructing the advertised input.
            model_image = np.ascontiguousarray(cv_image).reshape(IMAGE_MODEL_SHAPE)
            self.image_queue.append(
                (model_image, stamp_to_nanoseconds(msg.header.stamp), msg.header.stamp))
            self._couple_available_samples()
        except Exception as exc:
            self.get_logger().error(f'Error converting image: {exc}')

    def imu_callback(self, msg):
        stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
        sample = np.array([
            stamp_ns,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=np.float64)
        self.imu_queue.append((sample, stamp_ns))
        self._couple_available_samples()

    def _couple_available_samples(self):
        """FIFO-couple each consecutive image pair with the next ten IMUs."""
        while (len(self.image_queue) >= 2 and
               len(self.imu_queue) >= IMU_SAMPLES_PER_IMAGE):
            older_image, older_ns, _ = self.image_queue.popleft()
            newer_image, newer_ns, newer_stamp = self.image_queue[0]
            imu_entries = [
                self.imu_queue.popleft()
                for _ in range(IMU_SAMPLES_PER_IMAGE)
            ]

            imu_samples = np.stack([entry[0] for entry in imu_entries])
            alignment_error_ns = max(
                abs(imu_entries[0][1] - older_ns),
                abs(imu_entries[-1][1] - newer_ns))
            if alignment_error_ns > self.max_alignment_error_ns:
                self._warn_throttled(
                    'alignment',
                    'Camera/IMU FIFO alignment error is '
                    f'{alignment_error_ns / 1e6:.1f} ms (limit: '
                    f'{self.max_alignment_error_ns / 1e6:.1f} ms). Check that '
                    'camera and IMU playback start together and run near 20/200 Hz.')

            image_pair = np.stack((older_image, newer_image))
            self.pending_pairs.append(
                (image_pair, imu_samples.reshape(-1), newer_stamp))

        if len(self.pending_pairs) > self.backlog_warning_pairs:
            self._warn_throttled(
                'backlog',
                f'Inference backlog is {len(self.pending_pairs)} image pairs; '
                'real-time processing is falling behind the camera stream')

    def timer_callback(self):
        if not self.pending_pairs:
            return

        image_pair, imu_group, stamp = self.pending_pairs.popleft()
        self.image_sequence.append(image_pair)
        self.imu_sequence.append(imu_group)

        if len(self.image_sequence) < self.sequence_length:
            return

        image_array = np.stack(self.image_sequence).astype(np.float32) / 255.0
        imu_array = np.stack(self.imu_sequence).astype(np.float32)
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).to(self.device)
        imu_tensor = torch.from_numpy(imu_array).unsqueeze(0).to(self.device)

        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self.model(image_tensor, imu_tensor)
            relative_poses = output[0].detach().cpu().numpy()
        except Exception as exc:
            self.get_logger().error(f'Model inference failed: {exc}')
            return
        elapsed = time.perf_counter() - started

        if elapsed > self.target_period:
            self._warn_throttled(
                'inference_speed',
                f'Inference took {elapsed * 1e3:.1f} ms, exceeding the '
                f'{self.target_period * 1e3:.1f} ms EuRoC camera period')

        expected_shape = (self.sequence_length, 7)
        if (relative_poses.shape != expected_shape or
                not np.all(np.isfinite(relative_poses))):
            self.get_logger().error(
                f'Model returned invalid pose sequence with shape '
                f'{relative_poses.shape}; expected {expected_shape}')
            return

        try:
            # The first full window contains all motion since the identity pose.
            # Later rolling windows overlap it, so only their newest step is new.
            poses_to_apply = (relative_poses if not self.pose_initialized
                              else relative_poses[-1:])
            accumulated_pose = self.pose
            for relative_pose in poses_to_apply:
                accumulated_pose = compose_pose(
                    accumulated_pose, relative_pose)
        except ValueError as exc:
            self.get_logger().error(f'Invalid model quaternion: {exc}')
            return
        self.pose = accumulated_pose
        self.pose_initialized = True
        self._publish_pose(stamp)

    def _publish_pose(self, stamp):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.output_frame_id
        pose_msg.pose.position.x = float(self.pose[0])
        pose_msg.pose.position.y = float(self.pose[1])
        pose_msg.pose.position.z = float(self.pose[2])
        # The model uses wxyz; ROS geometry messages use xyzw.
        pose_msg.pose.orientation.w = float(self.pose[3])
        pose_msg.pose.orientation.x = float(self.pose[4])
        pose_msg.pose.orientation.y = float(self.pose[5])
        pose_msg.pose.orientation.z = float(self.pose[6])
        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DeepVINSNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
