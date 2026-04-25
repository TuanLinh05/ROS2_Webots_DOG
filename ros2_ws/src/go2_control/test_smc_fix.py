#!/usr/bin/env python3
"""Kiểm tra nhanh các module đã sửa."""
import sys
sys.path.insert(0, '/home/blingxyanua/ros2_ws/src/go2_control')

from go2_control.go2_kinematics import Go2Kinematics
from go2_control.gait_planner import GaitPlanner
from go2_control.smc_controller import SMCControllerMultiJoint

kin = Go2Kinematics()
th, ca = kin.inverse_kinematics(0.0, -0.28)
print(f"IK(0, -0.28): thigh={th:.4f}, calf={ca:.4f}")

x, z = kin.forward_kinematics(th, ca)
print(f"FK check: x={x:.4f}, z={z:.4f}")

G_th, G_ca = kin.compute_gravity_torque(th, ca)
print(f"Gravity torque: G_thigh={G_th:.4f} Nm, G_calf={G_ca:.4f} Nm")

M = kin.compute_inertia_matrix(ca)
print(f"Inertia matrix:\n{M}")

gait = GaitPlanner(gait_type='trot', freq=2.5, stride_length=0.10, default_z=-0.28)
x0, z0 = gait.get_foot_target('FL', 1.5, start_time=1.5)
x1, z1 = gait.get_foot_target('FL', 1.6, start_time=1.5)
print(f"\nFL stance start: x={x0:.4f} (should be +0.05)")
print(f"FL stance mid:   x={x1:.4f} (should be decreasing)")
print(f"Stance moves backward: {x1 < x0} (MUST be True)")
assert x1 < x0, "FAIL: stance direction not reversed!"

joints = ['FL_thigh_joint', 'FL_calf_joint']
smc = SMCControllerMultiJoint(joint_names=joints, kinematics=kin)
q_des = {'FL_thigh_joint': th, 'FL_calf_joint': ca}
q_act = {'FL_thigh_joint': th + 0.01, 'FL_calf_joint': ca - 0.01}
smc.initialize_state(q_act, q_des)
torques, surfaces = smc.compute_all_torques(q_des, q_act, 0.001)
print(f"\nSMC torques: thigh={torques['FL_thigh_joint']:.3f} Nm, calf={torques['FL_calf_joint']:.3f} Nm")
print(f"SMC surfaces: thigh={surfaces['FL_thigh_joint']:.3f}, calf={surfaces['FL_calf_joint']:.3f}")

print("\n=== ALL TESTS PASSED ===")
