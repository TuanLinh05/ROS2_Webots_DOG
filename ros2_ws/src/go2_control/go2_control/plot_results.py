import sys
import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def load_csv(filepath):
    """Đọc file CSV → dict of lists."""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value))
                except (ValueError, TypeError):
                    data[key].append(value)
    return data


def find_latest_log(log_dir=None):
    """Tìm file CSV mới nhất trong thư mục log."""
    if log_dir is None:
        log_dir = os.path.expanduser('~/ros2_ws/logs')
    if not os.path.exists(log_dir):
        return None
    files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    if not files:
        return None
    files.sort(reverse=True)
    return os.path.join(log_dir, files[0])


def plot_tracking_error(data, joints=None):
    """Vẽ sai số bám quỹ đạo theo thời gian."""
    t = data.get('time', [])
    if not t:
        return

    if joints is None:
        joints = [k.replace('_err', '') for k in data.keys() if k.endswith('_err')]

    fig, axes = plt.subplots(len(joints), 1, figsize=(12, 3 * len(joints)), sharex=True)
    if len(joints) == 1:
        axes = [axes]

    for ax, joint in zip(axes, joints):
        err_key = f'{joint}_err'
        if err_key in data:
            ax.plot(t, data[err_key], linewidth=0.8)
            ax.set_ylabel(f'{joint}\nerror (rad)')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Tracking Error', fontsize=14)
    plt.tight_layout()
    return fig


def plot_torques(data, joints=None):
    """Vẽ torque theo thời gian."""
    t = data.get('time', [])
    if not t:
        return

    if joints is None:
        joints = [k.replace('_tau', '') for k in data.keys() if k.endswith('_tau')]

    if not joints:
        print('[PLOT] No torque data found (position control mode?)')
        return None

    fig, axes = plt.subplots(len(joints), 1, figsize=(12, 3 * len(joints)), sharex=True)
    if len(joints) == 1:
        axes = [axes]

    for ax, joint in zip(axes, joints):
        tau_key = f'{joint}_tau'
        if tau_key in data:
            ax.plot(t, data[tau_key], linewidth=0.8, color='tab:red')
            ax.set_ylabel(f'{joint}\ntorque (Nm)')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Control Torques', fontsize=14)
    plt.tight_layout()
    return fig


def plot_sliding_surface(data, joints=None):
    """Vẽ mặt trượt SMC theo thời gian."""
    t = data.get('time', [])
    if not t:
        return

    if joints is None:
        joints = [k.replace('_s', '') for k in data.keys() if k.endswith('_s')]

    if not joints:
        print('[PLOT] No sliding surface data found')
        return None

    fig, axes = plt.subplots(len(joints), 1, figsize=(12, 3 * len(joints)), sharex=True)
    if len(joints) == 1:
        axes = [axes]

    for ax, joint in zip(axes, joints):
        s_key = f'{joint}_s'
        if s_key in data:
            ax.plot(t, data[s_key], linewidth=0.8, color='tab:green')
            ax.set_ylabel(f'{joint}\ns')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Sliding Surface Convergence', fontsize=14)
    plt.tight_layout()
    return fig


def plot_foot_trajectory(data, legs=None):
    """Vẽ quỹ đạo bàn chân trong mặt phẳng x-z."""
    if legs is None:
        legs = ['FL', 'FR', 'RL', 'RR']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, leg in zip(axes, legs):
        x_key = f'{leg}_foot_x'
        z_key = f'{leg}_foot_z'
        if x_key in data and z_key in data:
            ax.plot(data[x_key], data[z_key], linewidth=0.5, alpha=0.7)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('z (m)')
            ax.set_title(f'{leg} Foot Trajectory')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            # Đánh dấu điểm bắt đầu
            ax.plot(data[x_key][0], data[z_key][0], 'go', markersize=8, label='Start')
            ax.legend()

    fig.suptitle('Foot Trajectories (x-z plane)', fontsize=14)
    plt.tight_layout()
    return fig


def main():
    # Tìm file CSV
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = find_latest_log()

    if filepath is None or not os.path.exists(filepath):
        print('No log file found. Run simulation first.')
        print('Usage: python3 plot_results.py [path/to/csv]')
        return

    print(f'Loading: {filepath}')
    data = load_csv(filepath)
    print(f'Loaded {len(data.get("time", []))} data points')

    # Vẽ biểu đồ
    fl_joints = ['FL_thigh_joint', 'FL_calf_joint']

    plot_tracking_error(data, fl_joints)
    plot_torques(data, fl_joints)
    plot_sliding_surface(data, fl_joints)
    plot_foot_trajectory(data)

    plt.show()


if __name__ == '__main__':
    main()
