import os
import csv
import time


class DataLogger:
    """Ghi dữ liệu real-time từ simulation."""

    def __init__(self, log_dir=None, joint_names=None, enabled=True):
        """
        Args:
            log_dir: thư mục lưu file CSV (mặc định ~/ros2_ws/logs/)
            joint_names: danh sách tên khớp cần ghi
            enabled: bật/tắt logging
        """
        self.enabled = enabled
        self.joint_names = joint_names or []
        self.data = []

        if log_dir is None:
            log_dir = os.path.expanduser('~/ros2_ws/logs')
        self.log_dir = log_dir

        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)

    def log(self, t, q_des=None, q_act=None, torques=None,
            surfaces=None, foot_positions=None, gait_type=None):
        """Ghi 1 dòng dữ liệu.

        Args:
            t: thời gian simulation (s)
            q_des: dict {joint: desired_position}
            q_act: dict {joint: actual_position}
            torques: dict {joint: torque} (None nếu position control)
            surfaces: dict {joint: sliding_surface} (None nếu position control)
            foot_positions: dict {leg: (x, z)} vị trí foot
            gait_type: tên dáng đi hiện tại
        """
        if not self.enabled:
            return

        row = {'time': t, 'gait': gait_type or ''}

        for name in self.joint_names:
            if q_des and name in q_des:
                row[f'{name}_des'] = q_des[name]
            if q_act and name in q_act:
                row[f'{name}_act'] = q_act[name]
            if q_des and q_act and name in q_des and name in q_act:
                row[f'{name}_err'] = q_des[name] - q_act[name]
            if torques and name in torques:
                row[f'{name}_tau'] = torques[name]
            if surfaces and name in surfaces:
                row[f'{name}_s'] = surfaces[name]

        if foot_positions:
            for leg, (x, z) in foot_positions.items():
                row[f'{leg}_foot_x'] = x
                row[f'{leg}_foot_z'] = z

        self.data.append(row)

    def save(self, filename=None):
        """Lưu dữ liệu ra file CSV.

        Args:
            filename: tên file (mặc định: smc_log_<timestamp>.csv)

        Returns:
            đường dẫn file đã lưu
        """
        if not self.enabled or not self.data:
            return None

        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'smc_log_{timestamp}.csv'

        filepath = os.path.join(self.log_dir, filename)

        # Lấy tất cả field names từ dữ liệu
        fieldnames = []
        for row in self.data:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data)

        print(f'[LOG] Saved {len(self.data)} rows to: {filepath}', flush=True)
        return filepath

    def clear(self):
        """Xóa dữ liệu đã ghi."""
        self.data.clear()

    @property
    def count(self):
        """Số dòng dữ liệu đã ghi."""
        return len(self.data)
