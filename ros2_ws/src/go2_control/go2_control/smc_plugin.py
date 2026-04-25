import rclpy
import math
import numpy as np
import os

from go2_control.go2_kinematics import Go2Kinematics
from go2_control.gait_planner import GaitPlanner
from go2_control.smc_controller import SMCControllerMultiJoint
from go2_control.data_logger import DataLogger


class SMCControllerPlugin:

    # ==================== CẤU HÌNH ====================
    CONTROL_MODE = 'smc'   # 'position' hoặc 'smc'
    GAIT_TYPE = 'trot'          # 'trot', 'walk', hoặc 'pronk'
    ENABLE_LOGGING = True       # Bật/tắt ghi dữ liệu
    # =====================================================

    def init(self, webots_node, properties):
        # --- 1. KHỞI TẠO BIẾN HỆ THỐNG ---
        self.L1, self.L2 = 0.213, 0.213
        self.default_z = -0.28
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

        self.gait = GaitPlanner(
            gait_type=self.GAIT_TYPE,
            freq=2.5,
            stride_length=0.10,
            swing_height=0.05,
            default_z=self.default_z,
        )

        # Danh sách driven joints (thigh + calf, không hip)
        self.driven_joints = [n for n in self.joint_names if 'hip' not in n]

        # SMC Controller với per-joint gains và gravity compensation
        self.smc = SMCControllerMultiJoint(
            joint_names=self.driven_joints,
            kinematics=self.kin,
            thigh_gains={
                'K_p': 50.0, 'K_d': 2.0, 'Lambda': 8.0,
                'K_smc': 5.0, 'Phi': 0.15, 'max_torque': 23.7,
                'vel_filter_alpha': 0.15,
            },
            calf_gains={
                'K_p': 30.0, 'K_d': 1.5, 'Lambda': 12.0,
                'K_smc': 3.0, 'Phi': 0.10, 'max_torque': 45.43,
                'vel_filter_alpha': 0.15,
            },
        )

        self.logger = DataLogger(
            joint_names=self.driven_joints,
            enabled=self.ENABLE_LOGGING,
        )

        self.torque_mode_enabled = False

        # --- 4. NẠP MÔ HÌNH VẬT LÝ (PINOCCHIO - tùy chọn) ---
        self.model = None
        self.data = None
        try:
            import pinocchio as pin
            urdf_path = os.path.expanduser(
                '~/ros2_ws/src/go2_description/urdf/go2_description.urdf')
            if os.path.exists(urdf_path):
                self.model = pin.buildModelFromUrdf(urdf_path)
                self.data = self.model.createData()
                print('[OK] Pinocchio dynamics loaded (monitoring only)', flush=True)
        except Exception:
            pass

        print(f'\n[SMC Plugin] mode={self.CONTROL_MODE} gait={self.GAIT_TYPE}\n', flush=True)

    def inverse_kinematics(self, x, z):
        """IK giống hệt code gốc đã kiểm chứng: atan2(x, -z)"""
        return self.kin.inverse_kinematics(x, z)

    def step(self):
        self._t += self.dt
        self._step_count += 1

        # --- 1. TÍNH TOÁN ĐỘNG LỰC HỌC (giám sát, không dùng để điều khiển) ---
        if self.model is not None:
            try:
                import pinocchio as pin
                q_current = pin.neutral(self.model)
                for name in self.joint_names:
                    if self.model.existJointName(name) and name in self.sensors:
                        joint_id = self.model.getJointId(name)
                        q_idx = self.model.joints[joint_id].idx_q
                        q_current[q_idx] = self.sensors[name].getValue()

                pin.crba(self.model, self.data, q_current)
                G = pin.computeGeneralizedGravity(self.model, self.data, q_current)

                if self._step_count % 50 == 0:
                    print(f'[DYNAMICS] G(5): {G[:5]}', flush=True)
            except Exception as e:
                if self._step_count % 50 == 0:
                    print(f'[DYNAMICS ERROR] {e}', flush=True)

        # --- 2. GAIT PLANNING ---
        freq = self.gait.freq
        stride_len = self.gait.stride_length
        swing_height = self.gait.swing_height
        T = self.gait.T

        # Pha 1: Đáp đất ổn định (0s - 1s)
        if self._t < 1.0:
            thigh, calf = self.inverse_kinematics(0.0, self.default_z)
            for leg in ['FL', 'FR', 'RL', 'RR']:
                if f'{leg}_hip_joint' in self.motors:
                    self.motors[f'{leg}_hip_joint'].setPosition(0.0)
                    self.motors[f'{leg}_thigh_joint'].setPosition(thigh)
                    self.motors[f'{leg}_calf_joint'].setPosition(calf)
            return

        # Pha 2: Lấy đà (1s - 1.5s)
        if self._t < 1.5:
            lerp = (self._t - 1.0) / 0.5
            for leg in ['FL', 'FR', 'RL', 'RR']:
                x, z = self.gait.get_transition_target(leg, lerp)
                thigh, calf = self.inverse_kinematics(x, z)
                if f'{leg}_thigh_joint' in self.motors:
                    self.motors[f'{leg}_hip_joint'].setPosition(0.0)
                    self.motors[f'{leg}_thigh_joint'].setPosition(thigh)
                    self.motors[f'{leg}_calf_joint'].setPosition(calf)

            # Cập nhật trạng thái SMC nếu cần
            if self.CONTROL_MODE == 'smc':
                q_act = {n: self.sensors[n].getValue() for n in self.driven_joints if n in self.sensors}
                q_des = {}
                for leg in ['FL', 'FR', 'RL', 'RR']:
                    x, z = self.gait.get_transition_target(leg, lerp)
                    th, ca = self.inverse_kinematics(x, z)
                    q_des[f'{leg}_thigh_joint'] = th
                    q_des[f'{leg}_calf_joint'] = ca
                self.smc.initialize_state(q_act, q_des)
            return

        # Pha 3: Chu kỳ dáng đi liên tục
        q_des = {}
        foot_pos = {}

        for leg in ['FL', 'FR', 'RL', 'RR']:
            x, z = self.gait.get_foot_target(leg, self._t, start_time=1.5)
            thigh, calf = self.inverse_kinematics(x, z)

            q_des[f'{leg}_thigh_joint'] = thigh
            q_des[f'{leg}_calf_joint'] = calf
            foot_pos[leg] = (x, z)

        # --- 3. ĐIỀU KHIỂN ---
        torques = None
        surfaces = None

        if self.CONTROL_MODE == 'position':
            # POSITION CONTROL — giống hệt code gốc đã kiểm chứng
            for leg in ['FL', 'FR', 'RL', 'RR']:
                if f'{leg}_hip_joint' in self.motors:
                    self.motors[f'{leg}_hip_joint'].setPosition(0.0)
                    self.motors[f'{leg}_thigh_joint'].setPosition(q_des[f'{leg}_thigh_joint'])
                    self.motors[f'{leg}_calf_joint'].setPosition(q_des[f'{leg}_calf_joint'])

        elif self.CONTROL_MODE == 'smc':
            # SMC TORQUE CONTROL
            if not self.torque_mode_enabled:
                for name in self.driven_joints:
                    if name in self.motors:
                        # Bước 1: Đặt vận tốc motor = infinity (cho phép quay tự do)
                        self.motors[name].setVelocity(float('inf'))
                        # Bước 2: Đặt position = infinity (chuyển sang torque mode)
                        self.motors[name].setPosition(float('inf'))
                        # Bước 3: Khởi tạo torque = 0 (giữ ổn định)
                        self.motors[name].setTorque(0.0)
                self.torque_mode_enabled = True
                print('[SMC] Torque mode enabled for driven joints', flush=True)

            # Hip vẫn dùng position control
            for leg in ['FL', 'FR', 'RL', 'RR']:
                if f'{leg}_hip_joint' in self.motors:
                    self.motors[f'{leg}_hip_joint'].setPosition(0.0)

            # SMC cho thigh + calf (với gravity compensation)
            q_act = {n: self.sensors[n].getValue() for n in self.driven_joints if n in self.sensors}
            torques, surfaces = self.smc.compute_all_torques(q_des, q_act, self.dt)

            for name, tau in torques.items():
                if name in self.motors:
                    self.motors[name].setTorque(tau)

        # --- 4. LOGGING ---
        if self.ENABLE_LOGGING and self._step_count % 5 == 0:
            q_act = {n: self.sensors[n].getValue() for n in self.driven_joints if n in self.sensors}
            self.logger.log(
                t=self._t, q_des=q_des, q_act=q_act,
                torques=torques, surfaces=surfaces,
                foot_positions=foot_pos, gait_type=self.GAIT_TYPE,
            )

        # --- 5. DEBUG PRINT mỗi 2 giây ---
        if self._step_count % int(2.0 / self.dt) == 0:
            fl = foot_pos.get('FL', (0, 0))
            th_d = q_des.get('FL_thigh_joint', 0)
            ca_d = q_des.get('FL_calf_joint', 0)
            fk_x, fk_z = self.kin.forward_kinematics(th_d, ca_d)

            # Thêm thông tin torque và error vào debug
            debug_extra = ''
            if torques and self.CONTROL_MODE == 'smc':
                fl_thigh_tau = torques.get('FL_thigh_joint', 0)
                fl_calf_tau = torques.get('FL_calf_joint', 0)
                q_act_dbg = {n: self.sensors[n].getValue() for n in self.driven_joints if n in self.sensors}
                fl_th_err = q_des.get('FL_thigh_joint', 0) - q_act_dbg.get('FL_thigh_joint', 0)
                fl_ca_err = q_des.get('FL_calf_joint', 0) - q_act_dbg.get('FL_calf_joint', 0)
                debug_extra = (f' tau=({fl_thigh_tau:.1f},{fl_calf_tau:.1f})'
                               f' err=({fl_th_err:.3f},{fl_ca_err:.3f})')

            print(f'[t={self._t:.1f}] {self.CONTROL_MODE}/{self.GAIT_TYPE} '
                  f'FL=({fl[0]:.3f},{fl[1]:.3f}) FK=({fk_x:.3f},{fk_z:.3f})'
                  f'{debug_extra}', flush=True)

        # Auto-save log mỗi 30 giây
        if self.ENABLE_LOGGING and self._t % 30.0 < self.dt:
            self.logger.save()