Lệnh thực hiện tuần tự:

cd ~/ros2_ws

colcon build --packages-select go2_control

source install/setup.bash

ros2 launch go2_control control.launch.py
