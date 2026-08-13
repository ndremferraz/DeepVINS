from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

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
        default_value='/pose',
        description='Output Pose topic'
    ),
    DeclareLaunchArgument(
        'timer_period',
        default_value='1.0',
        description='Timer period in seconds'
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
                    'timer_period': LaunchConfiguration('timer_period')
                }]
            )
        ]
    )