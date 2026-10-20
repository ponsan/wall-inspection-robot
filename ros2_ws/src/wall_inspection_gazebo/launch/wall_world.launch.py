import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # インストール先（install/.../share/wall_inspection_gazebo）からワールドを探す
    pkg_share = get_package_share_directory('wall_inspection_gazebo')
    world = os.path.join(pkg_share, 'worlds', 'wall.sdf')

    # ros_gz_sim が用意している Gazebo 起動用の launch を読み込む
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
            )
        ),
        # -r：起動と同時にシミュレーションを開始する
        launch_arguments={'gz_args': f'-r {world}'}.items(),
    )

    return LaunchDescription([gz_sim])