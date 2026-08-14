import rclpy 

from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import torch
import numpy as np

SEQUENCE_LENGTH = 12
IMU_PER_IMG = 7

class DeepVINSNode(Node):

    def __init__(self):
        super().__init__('DeepVINSNode')

        self.declare_parameter('image_topic', 'image_topic')
        self.declare_parameter('imu_topic', 'imu_topic')
        self.declare_parameter('pose_topic', 'pose_topic')
        self.declare_parameter('timer_period', 1.0)
        # self.declare_parameter('model_path', 'path_to_model')
        self.declare_parameter('sequence_length', 10) 
        
        self.img_sub = self.create_subscription(
            Image,
            self.get_parameter('image_topic').get_parameter_value().string_value,
            self.image_callback,
            10)

        self.imu_sub = self.create_subscription(
            Imu,
            self.get_parameter('imu_topic').get_parameter_value().string_value,
            self.imu_callback,
            10)

        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter('pose_topic').get_parameter_value().string_value,
            10)

        self.timer = self.create_timer(
            self.get_parameter('timer_period').get_parameter_value().double_value,
            self.timer_callback
        )
        
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = torch.jit.load(self.get_parameter('model_path').get_parameter_value().string_value, map_location=self.device)

        self.imu_buffer = None
        self.image_buffer = None
        self.callback_count = 0


    def timer_callback(self):
        # sequence_length = self.get_parameter('sequence_length').get_parameter_value().integer_value

        if self.image_buffer is not None and self.image_buffer.shape[0] > 2 and self.imu_buffer is not None:

            image_frame1 = self.image_buffer[-1,:,:]
            image_frame2 = self.image_buffer[-2,:,:]

            image = np.stack((image_frame2, image_frame1), axis=0)

            self.get_logger().info(f'Image Shape: {image.shape}, IMU shape {self.imu_buffer.shape}')

        '''
        

            img_tensor = torch.from_numpy(img_buffer[-sequence_length:]).float().to(self.device)
            imu_tensor = torch.from_numpy(imu_buffer[-sequence_length:]).float().to(self.device)

            with torch.no_grad():
                pose_output = self.model(img_tensor, imu_tensor)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'

            pose_msg.pose.position.x = pose_output[0].item()
            pose_msg.pose.position.y = pose_output[1].item()
            pose_msg.pose.position.z = pose_output[2].item()

            pose_msg.pose.orientation.x = pose_output[3].item()
            pose_msg.pose.orientation.y = pose_output[4].item()
            pose_msg.pose.orientation.z = pose_output[5].item()
            pose_msg.pose.orientation.w = pose_output[6].item()

            self.pose_pub.publish(pose_msg)
        '''


    def image_callback(self, msg):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            cv_image = np.expand_dims(cv_image, axis=0) 

            if self.image_buffer is None:
                self.image_buffer = cv_image
            else:
                self.image_buffer = np.vstack((self.image_buffer, cv_image))

        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')


    def imu_callback(self, msg):

        linear_acceleration = msg.linear_acceleration
        angular_velocity = msg.angular_velocity
        time_obg = Time.from_msg(msg.header.stamp)
        stamp_nanoseconds = time_obg.nanoseconds

        imu_arr = np.array([
            stamp_nanoseconds,
            angular_velocity.x,
            angular_velocity.y,
            angular_velocity.z,
            linear_acceleration.x,
            linear_acceleration.y,
            linear_acceleration.z])

        if self.imu_buffer is None:
            self.imu_buffer = imu_arr
        else:
            self.imu_buffer = np.vstack((self.imu_buffer, imu_arr))
        

def main(args=None):
    rclpy.init(args=args)
    vins = DeepVINSNode()

    rclpy.spin(vins)
    vins.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
