import math
import numpy as np


class GaitPlanner:
    """Bộ lập kế hoạch dáng đi cho robot bốn chân."""

    # Định nghĩa các dáng đi: phase_offset cho mỗi chân và duty factor
    GAITS = {
        'trot': {
            'offsets': {'FL': 0.0, 'RR': 0.0, 'FR': 0.5, 'RL': 0.5},
            'duty': 0.5,
        },
        'walk': {
            'offsets': {'FL': 0.0, 'FR': 0.5, 'RL': 0.75, 'RR': 0.25},
            'duty': 0.75,
        },
        'pronk': {
            'offsets': {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0},
            'duty': 0.5,
        },
    }

    def __init__(self, gait_type='trot', freq=2.5, stride_length=0.10,
                 swing_height=0.05, default_z=-0.28):
        """
        Args:
            gait_type: 'trot', 'walk', hoặc 'pronk'
            freq: tần số bước (Hz)
            stride_length: chiều dài sải chân (m)
            swing_height: chiều cao nhấc chân (m)
            default_z: chiều cao đứng mặc định (m, âm)
        """
        self.freq = freq
        self.stride_length = stride_length
        self.swing_height = swing_height
        self.default_z = default_z
        self.T = 1.0 / freq
        self.set_gait(gait_type)

    def set_gait(self, gait_type):
        """Chuyển đổi dáng đi."""
        if gait_type not in self.GAITS:
            raise ValueError(f"Gait '{gait_type}' not supported. Use: {list(self.GAITS.keys())}")
        self.gait_type = gait_type
        gait = self.GAITS[gait_type]
        self.offsets = gait['offsets']
        self.duty_factor = gait['duty']

    def _bezier4(self, t, P):
        """Bézier bậc 4: B(t) = Σ C(4,i)·(1-t)^(4-i)·t^i·P_i

        Args:
            t: tham số [0, 1]
            P: list 5 điểm kiểm soát [P0, P1, P2, P3, P4]

        Returns:
            Giá trị nội suy tại t
        """
        u = 1.0 - t
        return (u**4 * P[0]
                + 4 * u**3 * t * P[1]
                + 6 * u**2 * t**2 * P[2]
                + 4 * u * t**3 * P[3]
                + t**4 * P[4])

    def _bezier4_deriv(self, t, P):
        """Đạo hàm Bézier bậc 4: dB/dt.

        Args:
            t: tham số [0, 1]
            P: list 5 điểm kiểm soát

        Returns:
            Đạo hàm tại t
        """
        u = 1.0 - t
        return (4 * (-u**3 * P[0]
                     + (u**2 * (1 - 3*t)) * P[1]
                     + (u * t * (2 - 3*t)) * P[2]
                     + (t**2 * (3*u - t)) * P[3]  # Sửa lại: 3u*t^2 - t^3... 
                     + t**3 * P[4]))

    def swing_trajectory(self, t, x_start, x_end):
        """Quỹ đạo Bézier bậc 4 cho pha vung (swing).

        Args:
            t: tiến trình pha vung [0, 1]
            x_start: vị trí x bắt đầu (cuối stance = +stride/2)
            x_end: vị trí x kết thúc (đầu stance = -stride/2)

        Returns:
            (x, z) vị trí foot
        """
        h = self.swing_height
        x_mid = (x_start + x_end) / 2.0

        # 5 điểm kiểm soát cho x
        Px = [x_start, x_start, x_mid, x_end, x_end]
        # 5 điểm kiểm soát cho z (offset từ default_z)
        Pz = [0.0, 0.7 * h, h, 0.7 * h, 0.0]

        x = self._bezier4(t, Px)
        z = self.default_z + self._bezier4(t, Pz)

        return x, z

    def get_foot_target(self, leg, t_global, start_time=1.5):
        """Tính vị trí mục tiêu cho 1 chân tại thời điểm t.

        Args:
            leg: 'FL', 'FR', 'RL', 'RR'
            t_global: thời gian simulation hiện tại (s)
            start_time: thời điểm bắt đầu dáng đi (s)

        Returns:
            (x, z) vị trí foot trong convention IK
        """
        sl = self.stride_length
        offset = self.offsets[leg]

        # Tính pha [0, 1) có offset
        t_phase = (((t_global - start_time) % self.T) / self.T + offset) % 1.0

        if t_phase < self.duty_factor:
            # PHA ĐỨNG (Stance): chân trên mặt đất, tuyến tính
            stance_progress = t_phase / self.duty_factor
            x = (-sl / 2.0) + stance_progress * sl
            z = self.default_z
        else:
            # PHA VUNG (Swing): nhấc chân, Bézier
            swing_progress = (t_phase - self.duty_factor) / (1.0 - self.duty_factor)
            x, z = self.swing_trajectory(swing_progress, sl / 2.0, -sl / 2.0)

        return x, z

    def get_transition_target(self, leg, lerp):
        """Tính vị trí chuyển tiếp từ đứng yên sang dáng đi.

        Nội suy tuyến tính từ (0, default_z) đến vị trí bắt đầu của chân
        tại pha tương ứng.

        Args:
            leg: 'FL', 'FR', 'RL', 'RR'
            lerp: hệ số nội suy [0, 1]

        Returns:
            (x, z) vị trí foot
        """
        sl = self.stride_length
        phase = self.offsets[leg]

        # Tính vị trí x mà chân sẽ ở tại phase offset
        if phase < self.duty_factor:
            stance_progress = phase / self.duty_factor
            target_x = (-sl / 2.0) + stance_progress * sl
        else:
            # Chân bắt đầu ở pha swing → vị trí cuối stance
            target_x = sl / 2.0

        return lerp * target_x, self.default_z

