from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

launch_args = [
    DeclareLaunchArgument(
        'image_topic',
        default_value='/cam0/image_raw',
        description='Image topic'
    ),
    DeclareLaunchArgument(
        'imu_topic',
        default_value='/imu0',
        description='IMU topic'
    ),
    DeclareLaunchArgument(
        'pose_topic',
        default_value='/deepvins/pose',
        description='Output Pose topic'
    ),
    DeclareLaunchArgument(
        'model_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('deepvins'), 'models', 'best.pth'
        ]),
        description='Absolute path to the TorchScript DeepVINS model'
    ),
    DeclareLaunchArgument(
        'sequence_length',
        default_value='8',
        description='Number of image/IMU timesteps supplied to the model'
    ),
    DeclareLaunchArgument(
        'inference_rate_hz',
        default_value='20.0',
        description='Inference cadence; EuRoC cameras publish at 20 Hz'
    ),
    DeclareLaunchArgument(
        'output_frame_id',
        default_value='odom',
        description='Frame ID for the accumulated absolute pose'
    ),
    DeclareLaunchArgument(
        'max_alignment_error_sec',
        default_value='0.025',
        description='Warn when FIFO camera/IMU alignment exceeds this value'
    ),
    DeclareLaunchArgument(
        'backlog_warning_pairs',
        default_value='2',
        description='Warn when this many paired samples await inference'
    )
]


def generate_launch_description():
    return LaunchDescription(
        launch_args + [
            Node(
                package='deepvins',
                executable='deepvins_node',
                name='deepvins_node',
                output='screen',
                parameters=[{
                    'image_topic': LaunchConfiguration('image_topic'),
                    'imu_topic': LaunchConfiguration('imu_topic'),
                    'pose_topic': LaunchConfiguration('pose_topic'),
                    'model_path': LaunchConfiguration('model_path'),
                    'sequence_length': LaunchConfiguration('sequence_length'),
                    'inference_rate_hz': LaunchConfiguration('inference_rate_hz'),
                    'output_frame_id': LaunchConfiguration('output_frame_id'),
                    'max_alignment_error_sec': LaunchConfiguration(
                        'max_alignment_error_sec'),
                    'backlog_warning_pairs': LaunchConfiguration(
                        'backlog_warning_pairs')
                }]
            )
        ]
    )
