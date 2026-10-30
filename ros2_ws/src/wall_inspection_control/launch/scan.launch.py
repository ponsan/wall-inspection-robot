import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    gazebo_share = get_package_share_directory('wall_inspection_gazebo')
    control_share = get_package_share_directory('wall_inspection_control')

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

    inspector_node = Node(
        package='wall_inspection_perception',
        executable='inspector_node',
        prefix='python3',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # /ground_truth はワールド座標の真値なので、crawler/odom は world と同じ位置に置く
    world_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--frame-id', 'world', '--child-frame-id', 'crawler/odom'],
        parameters=[{'use_sim_time': True}],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(control_share, 'rviz', 'inspection.rviz')],
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription(
        [wall_world, scan_node, inspector_node, world_to_odom, rviz])
