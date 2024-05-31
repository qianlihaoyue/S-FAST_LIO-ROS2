import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_dir = get_package_share_directory('fast_lio')
    # siasun 
    config_file = os.path.join(package_dir, 'config','mid360.yaml')
    rviz_file = os.path.join(package_dir, 'config', 'lio.rviz')

    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        parameters=[config_file],
        output='screen'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_file],
    )

    ld = LaunchDescription()
    ld.add_action(fast_lio_node)
    ld.add_action(rviz_node)

    return ld
