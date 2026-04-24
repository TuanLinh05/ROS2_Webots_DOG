import rclpy
import math

class SMCControllerPlugin:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__node = webots_node
        self.__timestep = int(self.__robot.getBasicTimeStep())
        self._t = 0.0

        self.L1 = 0.213
        self.L2 = 0.213
        self.default_z = -0.28  

        self.joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        self.motors = {name: self.__robot.getDevice(name) for name in self.joint_names}
        for name in self.joint_names:
            sensor = self.__robot.getDevice(name + '_sensor')
            if sensor: sensor.enable(self.__timestep)

        self.__node.get_logger().info('Bộ điều khiển 8-DOF (Đã đảo chiều Tiến lên) sẵn sàng!')

    def inverse_kinematics(self, x, z):
        d2 = x**2 + z**2
        d = math.sqrt(d2)
        
        cos_calf = (d2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_calf = max(-1.0, min(1.0, cos_calf))
        theta_calf = -math.acos(cos_calf)
        
        cos_thigh = (self.L1**2 + d2 - self.L2**2) / (2 * self.L1 * d)
        cos_thigh = max(-1.0, min(1.0, cos_thigh))
        theta_thigh = math.atan2(x, -z) + math.acos(cos_thigh)
        
        return theta_thigh, theta_calf

    def step(self):
        self._t += self.__timestep / 1000.0
        
        freq = 2.5          
        stride_len = 0.10   
        swing_height = 0.05 
        T = 1.0 / freq

        # TRẠNG THÁI 1: RƠI VÀ ỔN ĐỊNH
        if self._t < 1.0:
            thigh, calf = self.inverse_kinematics(0.0, self.default_z)
            for leg in ['FL', 'FR', 'RL', 'RR']:
                self.motors[f'{leg}_hip_joint'].setPosition(0.0)
                self.motors[f'{leg}_thigh_joint'].setPosition(thigh)
                self.motors[f'{leg}_calf_joint'].setPosition(calf)
            return

        # TRẠNG THÁI 2: VÀO ĐÀ CHUẨN BỊ
        if self._t < 1.5:
            lerp = (self._t - 1.0) / 0.5
            for leg in ['FL', 'FR', 'RL', 'RR']:
                # [Đã đảo chiều]: Chân FL, RR lùi về sau (-). Chân FR, RL tiến tới trước (+)
                target_x = (-stride_len / 2.0) if leg in ['FL', 'RR'] else (stride_len / 2.0)
                current_x = 0.0 + lerp * target_x
                
                thigh, calf = self.inverse_kinematics(current_x, self.default_z)
                self.motors[f'{leg}_hip_joint'].setPosition(0.0)
                self.motors[f'{leg}_thigh_joint'].setPosition(thigh)
                self.motors[f'{leg}_calf_joint'].setPosition(calf)
            return

        # TRẠNG THÁI 3: VÀO CHU KỲ TROT
        t_phase = ((self._t - 1.5) % T) / T

        phases = {
            'FL': t_phase,
            'RR': t_phase,
            'FR': (t_phase + 0.5) % 1.0,
            'RL': (t_phase + 0.5) % 1.0
        }

        for leg in ['FL', 'FR', 'RL', 'RR']:
            phase = phases[leg]
            
            if phase < 0.5:
                # PHA ĐỨNG (Stance): [Đã đảo chiều] Đạp từ sau (-) lên trước (+)
                p = phase / 0.5
                x = (-stride_len / 2.0) + p * stride_len
                z = self.default_z
            else:
                # PHA VUNG (Swing): [Đã đảo chiều] Nhấc lên và kéo từ trước (+) về sau (-)
                p = (phase - 0.5) / 0.5
                x = (stride_len / 2.0) - p * stride_len
                z = self.default_z + swing_height * math.sin(p * math.pi)

            thigh, calf = self.inverse_kinematics(x, z)
            self.motors[f'{leg}_hip_joint'].setPosition(0.0)
            self.motors[f'{leg}_thigh_joint'].setPosition(thigh)
            self.motors[f'{leg}_calf_joint'].setPosition(calf)