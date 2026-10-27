import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    gazebo_share = get_package_share_directory('wall_inspection_gazebo')

    wall_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, 'launch', 'wall_world.launch.py')
        )
    )

    scan_node = Node(
        package='wall_inspection_control',
        executable='scan_node',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription([wall_world, scan_node])
