import rclpy
import math
import numpy as np
import os

from go2_control.go2_kinematics import Go2Kinematics
from go2_control.smc_controller import SMCControllerMultiJoint
from go2_control.data_logger import DataLogger


def smoothstep(t):
    """Nội suy mượt: 3t² - 2t³ (Hermite smoothstep).
    Input t trong [0, 1], output trong [0, 1].
    Đạo hàm = 0 tại t=0 và t=1 → không giật.
    """
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class SMCControllerPlugin:

    # ==================== CẤU HÌNH ====================
    CONTROL_MODE = 'smc'        # 'position' hoặc 'smc'
    ENABLE_LOGGING = True       # Bật/tắt ghi dữ liệu

    # Thời gian các pha (giây)
    T_STAND = 2.0               # Pha đứng ổn định
    T_TRANSITION = 3.0          # Thời gian chuyển đứng → ngồi (chậm = mượt hơn)
    # =====================================================

    # --- TƯ THẾ ĐỨNG (Standing Pose) ---
    # Tất cả chân giống nhau: đứng thẳng
    STAND_POSE = {
        # Chân trước
        'FL_hip_joint': 0.0, 'FL_thigh_joint': 0.854, 'FL_calf_joint': -1.707,
        'FR_hip_joint': 0.0, 'FR_thigh_joint': 0.854, 'FR_calf_joint': -1.707,
        # Chân sau
        'RL_hip_joint': 0.0, 'RL_thigh_joint': 0.854, 'RL_calf_joint': -1.707,
        'RR_hip_joint': 0.0, 'RR_thigh_joint': 0.854, 'RR_calf_joint': -1.707,
    }

    # --- TƯ THẾ NGỒI (Sitting Pose) ---
    # Chân trước: hơi duỗi thẳng hơn (chống đỡ trọng lượng phía trước)
    # Chân sau: gập sâu (hạ mông xuống đất)
    SIT_POSE = {
        # Chân trước — hơi duỗi thẳng, hạ thấp nhẹ
        'FL_hip_joint': 0.0, 'FL_thigh_joint': 0.5,  'FL_calf_joint': -1.0,
        'FR_hip_joint': 0.0, 'FR_thigh_joint': 0.5,  'FR_calf_joint': -1.0,
        # Chân sau — gập sâu, đùi xoay mạnh, ống chân gập lại
        'RL_hip_joint': 0.0, 'RL_thigh_joint': 2.0,  'RL_calf_joint': -2.5,
        'RR_hip_joint': 0.0, 'RR_thigh_joint': 2.0,  'RR_calf_joint': -2.5,
    }

    def init(self, webots_node, properties):
        # --- 1. KHỞI TẠO BIẾN HỆ THỐNG ---
        self._t = 0.0
        self.__robot = webots_node.robot
        self.__node = webots_node
        self.__timestep = int(self.__robot.getBasicTimeStep())
        self.dt = self.__timestep / 1000.0
        self._step_count = 0

        self.joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]

        self.motors = {}
        self.sensors = {}

        # --- 2. KẾT NỐI THIẾT BỊ WEBOTS ---
        for name in self.joint_names:
            motor = self.__robot.getDevice(name)
            if motor:
                self.motors[name] = motor

            sensor = self.__robot.getDevice(name + '_sensor')
            if sensor:
                sensor.enable(self.__timestep)
                self.sensors[name] = sensor

        # --- 3. KHỞI TẠO MODULES ---
        self.kin = Go2Kinematics()

        # Danh sách driven joints (thigh + calf, không hip)
        self.driven_joints = [n for n in self.joint_names if 'hip' not in n]

        # SMC Controller — gain bảo thủ hơn cho quỹ đạo chậm
        self.smc = SMCControllerMultiJoint(
            joint_names=self.driven_joints,
            kinematics=self.kin,
            thigh_gains={
                'K_p': 80.0, 'K_d': 4.0, 'Lambda': 5.0,
                'K_smc': 10.0, 'Phi': 0.2, 'max_torque': 23.7,
                'vel_filter_alpha': 0.1,
            },
            calf_gains={
                'K_p': 50.0, 'K_d': 3.0, 'Lambda': 8.0,
                'K_smc': 6.0, 'Phi': 0.15, 'max_torque': 45.43,
                'vel_filter_alpha': 0.1,
            },
        )

        self.logger = DataLogger(
            joint_names=self.driven_joints,
            enabled=self.ENABLE_LOGGING,
        )

        self.torque_mode_enabled = False
        self._current_phase = 'init'

        print(f'\n[SMC Plugin] mode={self.CONTROL_MODE} action=STAND→SIT\n', flush=True)

    def _get_target_pose(self, t):
        """Tính tư thế mục tiêu tại thời điểm t.

        Returns:
            dict {joint_name: target_angle}, str phase_name
        """
        if t < self.T_STAND:
            # Pha 1: Đứng yên
            return dict(self.STAND_POSE), 'stand'

        elif t < self.T_STAND + self.T_TRANSITION:
            # Pha 2: Chuyển đứng → ngồi (nội suy mượt)
            progress = (t - self.T_STAND) / self.T_TRANSITION
            alpha = smoothstep(progress)

            pose = {}
            for name in self.joint_names:
                q_stand = self.STAND_POSE[name]
                q_sit = self.SIT_POSE[name]
                pose[name] = q_stand + alpha * (q_sit - q_stand)

            return pose, 'transition'

        else:
            # Pha 3: Giữ tư thế ngồi
            return dict(self.SIT_POSE), 'sit'

    def step(self):
        self._t += self.dt
        self._step_count += 1

        # --- Tính tư thế mục tiêu ---
        target_pose, phase = self._get_target_pose(self._t)

        # Lấy các góc khớp driven (thigh + calf)
        q_des = {n: target_pose[n] for n in self.driven_joints}

        # --- ĐIỀU KHIỂN ---
        torques = None
        surfaces = None

        if self.CONTROL_MODE == 'position':
            # === POSITION CONTROL ===
            for name in self.joint_names:
                if name in self.motors:
                    self.motors[name].setPosition(target_pose[name])

        elif self.CONTROL_MODE == 'smc':
            # === SMC TORQUE CONTROL ===

            if phase == 'stand' and self._t < self.T_STAND - 0.1:
                # Giai đoạn đầu: dùng position control để ổn định
                for name in self.joint_names:
                    if name in self.motors:
                        self.motors[name].setPosition(target_pose[name])

                # Cập nhật trạng thái SMC liên tục
                q_act = {n: self.sensors[n].getValue()
                         for n in self.driven_joints if n in self.sensors}
                self.smc.initialize_state(q_act, q_des)

            else:
                # Chuyển sang torque mode (một lần duy nhất)
                if not self.torque_mode_enabled:
                    for name in self.driven_joints:
                        if name in self.motors:
                            self.motors[name].setVelocity(float('inf'))
                            self.motors[name].setPosition(float('inf'))
                            self.motors[name].setTorque(0.0)
                    self.torque_mode_enabled = True
                    print(f'[t={self._t:.2f}] Torque mode ENABLED', flush=True)

                # Hip vẫn dùng position control
                for leg in ['FL', 'FR', 'RL', 'RR']:
                    hip_name = f'{leg}_hip_joint'
                    if hip_name in self.motors:
                        self.motors[hip_name].setPosition(target_pose[hip_name])

                # SMC cho thigh + calf
                q_act = {n: self.sensors[n].getValue()
                         for n in self.driven_joints if n in self.sensors}
                torques, surfaces = self.smc.compute_all_torques(q_des, q_act, self.dt)

                for name, tau in torques.items():
                    if name in self.motors:
                        self.motors[name].setTorque(tau)

        # --- LOGGING ---
        if self.ENABLE_LOGGING and self._step_count % 5 == 0:
            q_act = {n: self.sensors[n].getValue()
                     for n in self.driven_joints if n in self.sensors}
            self.logger.log(
                t=self._t, q_des=q_des, q_act=q_act,
                torques=torques, surfaces=surfaces,
                gait_type=phase,
            )

        # --- DEBUG PRINT mỗi 1 giây ---
        if self._step_count % int(1.0 / self.dt) == 0:
            phase_str = f'[{phase.upper():>10}]'

            if self.CONTROL_MODE == 'smc' and torques:
                q_act = {n: self.sensors[n].getValue()
                         for n in self.driven_joints if n in self.sensors}
                # Sai số lớn nhất
                max_err = max(abs(q_des[n] - q_act.get(n, 0))
                              for n in self.driven_joints)
                # Torque lớn nhất
                max_tau = max(abs(v) for v in torques.values())
                print(f'[t={self._t:.1f}] {phase_str} '
                      f'max_err={max_err:.4f} rad  max_tau={max_tau:.2f} Nm',
                      flush=True)
            else:
                print(f'[t={self._t:.1f}] {phase_str} position_ctrl', flush=True)

        # Auto-save log mỗi 30 giây
        if self.ENABLE_LOGGING and self._t % 30.0 < self.dt:
            self.logger.save()