import numpy as np
import math
from go2_control.go2_kinematics import Go2Kinematics


class SMCController:
    """Sliding Mode Controller cho 1 khớp (joint-level) với bù trọng lực."""

    def __init__(self, K_p=40.0, K_d=1.0, Lambda=10.0,
                 K_smc=4.0, Phi=0.1, max_torque=23.7,
                 vel_filter_alpha=0.2):
        """
        Args:
            K_p: Gain tỷ lệ (proportional)
            K_d: Gain vi phân (derivative)
            Lambda: Hệ số mặt trượt (sliding surface slope)
            K_smc: Gain SMC (switching gain)
            Phi: Lớp biên saturation (boundary layer thickness)
            max_torque: Moment xoắn tối đa (Nm)
            vel_filter_alpha: Hệ số lọc thông thấp cho vận tốc (0-1, nhỏ = mượt hơn)
        """
        self.K_p = K_p
        self.K_d = K_d
        self.Lambda = Lambda
        self.K_smc = K_smc
        self.Phi = Phi
        self.max_torque = max_torque
        self.vel_filter_alpha = vel_filter_alpha

    def sat(self, s):
        """Hàm saturation: giảm chattering.

        sat(s) = s/Φ  nếu |s| < Φ
               = sign(s) nếu |s| >= Φ
        """
        return np.clip(s / self.Phi, -1.0, 1.0)

    def compute_torque(self, e, de, G_comp=0.0):
        """Tính torque điều khiển cho 1 khớp.

        Luật điều khiển SMC với bù trọng lực:
            τ = G(q) + K_p·e + K_d·ė + K_smc·sat(s/Φ)

        Args:
            e: sai số vị trí (q_des - q_act)
            de: sai số vận tốc (dq_des - dq_act), đã lọc
            G_comp: torque bù trọng lực (Nm)

        Returns:
            tau: torque điều khiển (Nm)
            s: giá trị mặt trượt (để logging)
        """
        # Mặt trượt: s = ė + λ·e
        s = de + self.Lambda * e

        # Luật điều khiển: τ = G(q) + PD + SMC switching
        tau_pd = self.K_p * e + self.K_d * de
        tau_smc = self.K_smc * self.sat(s)
        tau = G_comp + tau_pd + tau_smc

        # Giới hạn torque
        tau = float(np.clip(tau, -self.max_torque, self.max_torque))

        return tau, s


class SMCControllerMultiJoint:
    """SMC Controller cho nhiều khớp cùng lúc với bù trọng lực."""

    # Gain mặc định phân biệt giữa thigh và calf
    DEFAULT_THIGH_GAINS = {
        'K_p': 50.0, 'K_d': 2.0, 'Lambda': 8.0,
        'K_smc': 5.0, 'Phi': 0.15, 'max_torque': 23.7,
        'vel_filter_alpha': 0.15,
    }
    DEFAULT_CALF_GAINS = {
        'K_p': 30.0, 'K_d': 1.5, 'Lambda': 12.0,
        'K_smc': 3.0, 'Phi': 0.10, 'max_torque': 45.43,
        'vel_filter_alpha': 0.15,
    }

    def __init__(self, joint_names, kinematics=None,
                 thigh_gains=None, calf_gains=None):
        """
        Args:
            joint_names: danh sách tên khớp (thigh + calf, không hip)
            kinematics: Go2Kinematics instance (cho tính G(q))
            thigh_gains: dict tham số cho khớp thigh (None = dùng mặc định)
            calf_gains: dict tham số cho khớp calf (None = dùng mặc định)
        """
        self.joint_names = joint_names
        self.kin = kinematics or Go2Kinematics()

        # Tạo controller riêng cho mỗi khớp với gain phù hợp
        tg = thigh_gains or self.DEFAULT_THIGH_GAINS
        cg = calf_gains or self.DEFAULT_CALF_GAINS

        self.controllers = {}
        for name in joint_names:
            if 'thigh' in name:
                self.controllers[name] = SMCController(**tg)
            elif 'calf' in name:
                self.controllers[name] = SMCController(**cg)

        # Lưu trữ trạng thái trước để tính vận tốc
        self.q_prev = {name: 0.0 for name in joint_names}
        self.q_des_prev = {name: 0.0 for name in joint_names}

        # Vận tốc đã lọc (low-pass filtered)
        self.v_act_filtered = {name: 0.0 for name in joint_names}
        self.v_des_filtered = {name: 0.0 for name in joint_names}

        self._initialized = False

    def _get_leg_prefix(self, joint_name):
        """Trích xuất tên chân từ tên khớp: 'FL_thigh_joint' → 'FL'"""
        return joint_name.split('_')[0]

    def _compute_gravity_for_leg(self, leg, q_act):
        """Tính G(q) cho 1 chân.

        Args:
            leg: 'FL', 'FR', 'RL', 'RR'
            q_act: dict {joint_name: position}

        Returns:
            dict {thigh_joint: G_thigh, calf_joint: G_calf}
        """
        thigh_name = f'{leg}_thigh_joint'
        calf_name = f'{leg}_calf_joint'

        theta_thigh = q_act.get(thigh_name, 0.0)
        theta_calf = q_act.get(calf_name, 0.0)

        G_thigh, G_calf = self.kin.compute_gravity_torque(theta_thigh, theta_calf)

        return {thigh_name: G_thigh, calf_name: G_calf}

    def compute_all_torques(self, q_des, q_act, dt):
        """Tính torque cho tất cả khớp với bù trọng lực.

        Args:
            q_des: dict {joint_name: desired_position}
            q_act: dict {joint_name: actual_position}
            dt: time step (s)

        Returns:
            torques: dict {joint_name: torque}
            surfaces: dict {joint_name: sliding_surface_value}
        """
        torques = {}
        surfaces = {}

        # Tính gravity compensation cho mỗi chân
        gravity_comp = {}
        for leg in ['FL', 'FR', 'RL', 'RR']:
            gravity_comp.update(self._compute_gravity_for_leg(leg, q_act))

        for name in self.joint_names:
            if name not in q_des or name not in q_act:
                continue

            ctrl = self.controllers[name]
            alpha = ctrl.vel_filter_alpha

            # Sai số vị trí
            e = q_des[name] - q_act[name]

            # Vận tốc thực tế (sai phân)
            v_act_raw = (q_act[name] - self.q_prev[name]) / dt

            # Vận tốc mong muốn (sai phân quỹ đạo)
            v_des_raw = (q_des[name] - self.q_des_prev[name]) / dt

            # Lọc thông thấp (EMA filter): giảm noise và spike
            self.v_act_filtered[name] = (
                alpha * v_act_raw + (1.0 - alpha) * self.v_act_filtered[name]
            )
            self.v_des_filtered[name] = (
                alpha * v_des_raw + (1.0 - alpha) * self.v_des_filtered[name]
            )

            # Sai số vận tốc (dùng vận tốc đã lọc)
            de = self.v_des_filtered[name] - self.v_act_filtered[name]

            # Clip sai số vận tốc tránh extreme values
            de = np.clip(de, -10.0, 10.0)

            # Lấy torque bù trọng lực
            G_comp = gravity_comp.get(name, 0.0)

            # Tính torque
            tau, s = ctrl.compute_torque(e, de, G_comp)

            torques[name] = tau
            surfaces[name] = s

            # Cập nhật trạng thái trước
            self.q_prev[name] = q_act[name]
            self.q_des_prev[name] = q_des[name]

        return torques, surfaces

    def initialize_state(self, q_act, q_des):
        """Khởi tạo trạng thái trước khi bắt đầu điều khiển torque.

        Gọi hàm này ở cuối giai đoạn position control.
        CHỈ KHỞI TẠO MỘT LẦN để tránh ghi đè liên tục.
        """
        if self._initialized:
            # Vẫn cập nhật q_prev để giữ vận tốc chính xác
            for name in self.joint_names:
                if name in q_act:
                    self.q_prev[name] = q_act[name]
                if name in q_des:
                    self.q_des_prev[name] = q_des[name]
            return

        for name in self.joint_names:
            if name in q_act:
                self.q_prev[name] = q_act[name]
            if name in q_des:
                self.q_des_prev[name] = q_des[name]
            # Reset filtered velocities to 0
            self.v_act_filtered[name] = 0.0
            self.v_des_filtered[name] = 0.0

        self._initialized = True
        print('[SMC] State initialized for torque control transition', flush=True)
