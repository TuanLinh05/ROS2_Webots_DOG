# 🐶 Unitree Go2 – Sliding Mode Gait Control in ROS 2 + Webots

![ROS2](https://img.shields.io/badge/ROS%202-Humble-22314E?logo=ros&logoColor=white)
![Webots](https://img.shields.io/badge/Webots-webots__ros2-C62828)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Control](https://img.shields.io/badge/Control-Sliding%20Mode-6A1B9A)

<a id="english"></a>**🇬🇧 English** · [🇻🇳 Tiếng Việt](#tieng-viet)

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

<a id="tieng-viet"></a>

## 🇻🇳 Tiếng Việt

[🇬🇧 English](#english) · **🇻🇳 Tiếng Việt**

Điều khiển dáng đi cho mô hình robot bốn chân **Unitree Go2** trong **Webots**, chạy từ **ROS 2** qua một plugin Python của `webots_ros2_driver`.
Mỗi khớp được điều khiển bằng **bộ điều khiển trượt có bù trọng lực**, quỹ đạo bàn chân lấy từ **bộ lập dáng đi dùng đường cong Bézier**. Mỗi lần chạy đều được ghi ra CSV để phân tích sau.

### ✨ Tính năng

- **SMC cấp khớp:** `τ = G(q) + Kp·e + Kd·ė + K_smc·sat(s/Φ)`. Hàm bão hòa có lớp biên giúp giảm chattering. Có thể đặt hệ số riêng cho từng khớp.
- **Lập dáng đi:** các dáng `trot`, `walk`, `pronk`, chỉnh được tần số, độ dài bước và độ cao nhấc chân. Pha nhấc chân dùng Bézier bậc 4.
- **Động học:** động học ngược cho khớp đùi và khớp gối của Go2, chuyển mượt từ đứng sang đi.
- **Ghi và vẽ dữ liệu:** góc khớp mong muốn / thực tế, mô-men, mặt trượt và vị trí bàn chân được lưu ra CSV. `plot_results.py` vẽ sai số bám, mô-men, mặt trượt và quỹ đạo bàn chân trong mặt phẳng x–z.

Cấu trúc thư mục: xem phần tiếng Anh ở trên.

### 🚀 Hướng dẫn sử dụng

> Yêu cầu: Ubuntu 22.04, ROS 2 Humble, Webots, `webots_ros2`.

1. Build: `colcon build --packages-select go2_description go2_control`
2. Nạp môi trường: `source install/setup.bash`
3. Chạy: `ros2 launch go2_control control.launch.py`
4. Sau khi chạy, vẽ log mới nhất: `python3 src/go2_control/go2_control/plot_results.py`

### 🙏 Ghi nhận

Mô hình robot Go2 (`go2_description`) dựa trên URDF mã nguồn mở của Unitree.

---

<p align="center">Made by <a href="https://github.com/TuanLinh05">Vu Tuan Linh</a> · HCMUT</p>
