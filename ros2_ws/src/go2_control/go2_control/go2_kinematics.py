import math
import numpy as np


# Hằng số vật lý
GRAVITY = 9.81  # m/s²


class Go2Kinematics:
    """Kinematics và Dynamics cho 1 chân Go2 (2-link: thigh + calf)."""

    # Tham số vật lý từ URDF
    L1 = 0.213          # Chiều dài thigh (m)
    L2 = 0.213          # Chiều dài calf (m)
    M_THIGH = 1.152     # Khối lượng thigh (kg)
    M_CALF = 0.154      # Khối lượng calf (kg)
    M_FOOT = 0.04       # Khối lượng foot (kg)

    # Vị trí trọng tâm (CoM) trên mỗi link (khoảng cách từ khớp, dọc link)
    # Từ URDF: FL_thigh CoM ≈ (0, 0, -0.0327) → ~0.0327m từ khớp hip
    # Nhưng trong mô hình 2-link planar, ta dùng tỷ lệ trên chiều dài link
    LC1 = 0.1065        # Khoảng cách từ hip đến CoM thigh (≈ L1/2)
    LC2 = 0.115         # Khoảng cách từ knee đến CoM calf (từ URDF)

    def __init__(self, L1=None, L2=None):
        if L1 is not None:
            self.L1 = L1
        if L2 is not None:
            self.L2 = L2

    def inverse_kinematics(self, x, z):
        """IK: vị trí foot (x, z) → góc khớp (theta_thigh, theta_calf).

        Args:
            x: vị trí ngang foot (convention IK)
            z: vị trí dọc foot (âm = xuống dưới)

        Returns:
            (theta_thigh, theta_calf) trong radian
        """
        d2 = x**2 + z**2
        d = math.sqrt(d2)

        cos_calf = (d2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_calf = max(-1.0, min(1.0, cos_calf))
        theta_calf = -math.acos(cos_calf)

        cos_thigh = (self.L1**2 + d2 - self.L2**2) / (2 * self.L1 * d)
        cos_thigh = max(-1.0, min(1.0, cos_thigh))
        theta_thigh = math.atan2(x, -z) + math.acos(cos_thigh)

        return theta_thigh, theta_calf

    def forward_kinematics(self, theta_thigh, theta_calf):
        """FK: góc khớp (theta_thigh, theta_calf) → vị trí foot (x, z).

        Nhất quán với convention IK: FK(IK(x,z)) ≈ (x,z).

        Returns:
            (x, z) vị trí foot trong convention IK
        """
        x = (self.L1 * math.sin(theta_thigh)
             + self.L2 * math.sin(theta_thigh + theta_calf))
        z = (-self.L1 * math.cos(theta_thigh)
             - self.L2 * math.cos(theta_thigh + theta_calf))
        return x, z

    def jacobian(self, theta_thigh, theta_calf):
        """Jacobian J(q) [2x2]: dx/dq.

        J = [[dx/d_thigh, dx/d_calf],
             [dz/d_thigh, dz/d_calf]]

        Returns:
            np.ndarray shape (2, 2)
        """
        c1 = math.cos(theta_thigh)
        c12 = math.cos(theta_thigh + theta_calf)
        s1 = math.sin(theta_thigh)
        s12 = math.sin(theta_thigh + theta_calf)

        return np.array([
            [self.L1 * c1 + self.L2 * c12,  self.L2 * c12],
            [self.L1 * s1 + self.L2 * s12,  self.L2 * s12]
        ])

    def compute_gravity_torque(self, theta_thigh, theta_calf):
        """Tính vector trọng lực G(q) cho mô hình 2-link planar.

        Sử dụng đạo hàm thế năng V theo góc khớp: tau_g = dV/dq.
        Chiều dương của góc làm chân vung về phía sau (ngược hướng tiến).
        Trọng lực kéo chân xuống (về góc 0), do đó cần moment dương để giữ chân ở góc dương.

        Ghi chú: m2 ở đây bao gồm cả khối lượng calf + foot.

        Args:
            theta_thigh: góc khớp thigh (rad)
            theta_calf: góc khớp calf (rad)

        Returns:
            (G_thigh, G_calf): torque bù trọng lực (Nm)
        """
        g = GRAVITY
        m1 = self.M_THIGH
        m2 = self.M_CALF + self.M_FOOT  # Calf + foot
        Lc1 = self.LC1
        Lc2 = self.LC2
        L1 = self.L1

        s1 = math.sin(theta_thigh)
        s12 = math.sin(theta_thigh + theta_calf)

        # Torque bù trọng lực (bù lại dV/dq)
        G_thigh = (m1 * Lc1 + m2 * L1) * g * s1 + m2 * Lc2 * g * s12
        G_calf = m2 * Lc2 * g * s12

        return G_thigh, G_calf

    def compute_inertia_matrix(self, theta_calf):
        """Tính ma trận quán tính M(q) cho mô hình 2-link planar.

        Args:
            theta_calf: góc khớp calf (rad)

        Returns:
            np.ndarray shape (2, 2): ma trận quán tính
        """
        m1 = self.M_THIGH
        m2 = self.M_CALF + self.M_FOOT
        Lc1 = self.LC1
        Lc2 = self.LC2
        L1 = self.L1
        c2 = math.cos(theta_calf)

        # Moments of inertia (xấp xỉ thanh mảnh)
        I1 = m1 * Lc1**2
        I2 = m2 * Lc2**2

        M11 = I1 + I2 + m2 * L1**2 + 2 * m2 * L1 * Lc2 * c2
        M12 = I2 + m2 * L1 * Lc2 * c2
        M22 = I2

        return np.array([[M11, M12], [M12, M22]])

    def verify_ik(self, x, z, tol=1e-3):
        """Kiểm tra IK bằng FK. Trả về True nếu sai số < tol."""
        th, ca = self.inverse_kinematics(x, z)
        x_fk, z_fk = self.forward_kinematics(th, ca)
        err = math.sqrt((x - x_fk)**2 + (z - z_fk)**2)
        return err < tol, err
