import os
import launch
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_controller import WebotsController

def generate_launch_description():
    desc_dir = get_package_share_directory('go2_description')
    world_path = os.path.join(desc_dir, 'worlds', 'empty.wbt')

    # Mở trình mô phỏng Webots
    webots = ExecuteProcess(
        cmd=['webots', world_path],
        output='screen'
    )

    # Gọi Plugin điều khiển Python và cắm thẳng vào cổng <extern> của robot
    go2_controller = WebotsController(
        robot_name='Go2Description', # Tên này được trích xuất trực tiếp từ khai báo <robot name="go2_description">
        parameters=[
            {'robot_description': '<robot><webots><plugin type="go2_control.smc_plugin.SMCControllerPlugin" /></webots></robot>'},
        ],
        respawn=True
    )

    return launch.LaunchDescription([
        webots,
        go2_controller,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])
