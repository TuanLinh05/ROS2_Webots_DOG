# 🐶 Unitree Go2 – Sliding Mode Gait Control in ROS 2 + Webots

![ROS2](https://img.shields.io/badge/ROS%202-Humble-22314E?logo=ros&logoColor=white)
![Webots](https://img.shields.io/badge/Webots-webots__ros2-C62828)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Control](https://img.shields.io/badge/Control-Sliding%20Mode-6A1B9A)

Gait control for the **Unitree Go2** quadruped model in **Webots**, driven from **ROS 2** through a `webots_ros2_driver` Python plugin.
Each joint is controlled by a **Sliding Mode Controller with gravity compensation**, and foot trajectories come from a **Bézier-curve gait planner**. Every run is logged to CSV for offline analysis.

---

## ✨ Features

- **Joint-level SMC:** `τ = G(q) + Kp·e + Kd·ė + K_smc·sat(s/Φ)`, where the boundary-layer saturation reduces chattering. Gains can be set per joint.
- **Gait planner:** `trot`, `walk` and `pronk` gaits with configurable frequency, stride length and swing height; the swing phase uses 4th-order Bézier curves.
- **Kinematics:** leg inverse kinematics for the Go2 thigh and calf joints, with smooth transitions from standing to walking.
- **Data logging & plotting:** desired vs actual joint angles, torques, sliding surfaces and foot positions are saved to CSV. `plot_results.py` plots tracking error, torques, sliding surfaces and x–z foot trajectories.

## 📂 Project structure

```text
ros2_ws/src/
├── go2_control/                 # Control package (Python)
│   ├── go2_control/
│   │   ├── smc_plugin.py        # webots_ros2 plugin: main control loop
│   │   ├── smc_controller.py    # Sliding Mode Controller (single & multi-joint)
│   │   ├── gait_planner.py      # Trot / walk / pronk + Bézier swing trajectories
│   │   ├── go2_kinematics.py    # Leg kinematics
│   │   ├── data_logger.py       # CSV logger
│   │   └── plot_results.py      # Offline plots
│   └── launch/control.launch.py # Starts Webots + the controller plugin
└── go2_description/             # Go2 URDF / xacro, Webots PROTO and world
```

## 🚀 Getting started

> Requirements: Ubuntu 22.04, ROS 2 Humble, Webots, `webots_ros2`.

```bash
cd ~/ros2_ws
colcon build --packages-select go2_description go2_control
source install/setup.bash
ros2 launch go2_control control.launch.py
```

Plot the latest log after a run:

```bash
python3 src/go2_control/go2_control/plot_results.py            # newest CSV
python3 src/go2_control/go2_control/plot_results.py path/to/smc_log.csv
```

## 🙏 Credits

The Go2 robot model (`go2_description`) is based on Unitree's open-source Go2 URDF.

## 🔗 Related

- [Quadruped-robot-12DOF](https://github.com/TuanLinh05/Quadruped-robot-12DOF) – custom 12-DOF quadruped with a 1 kHz C-based SMC controller and MATLAB telemetry.

---

<p align="center">Made by <a href="https://github.com/TuanLinh05">Vu Tuan Linh</a> · HCMUT</p>
