import os
import launch
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_dir = get_package_share_directory('go2_description')
    world_path = os.path.join(package_dir, 'worlds', 'empty.wbt')

    # Gọi trực tiếp Webots của Ubuntu bằng ExecuteProcess thay vì dùng WebotsLauncher
    webots = ExecuteProcess(
        cmd=['webots', world_path],
        output='screen'
    )

    return launch.LaunchDescription([
        webots
    ])