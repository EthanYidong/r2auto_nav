from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ts_client',
            executable='nfc',
        ),
        Node(
            package='ts_client',
            executable='button',
        ),
        Node(
            package='ts_client',
            executable='thermal',
        ),
        Node(
            package='ts_client',
            executable='motors',
        ),
    ])
