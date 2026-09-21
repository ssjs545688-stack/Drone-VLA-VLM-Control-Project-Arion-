import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction

def generate_launch_description():
    home_dir = os.path.expanduser('~')

    # 1. QGroundControl (새 터미널 창에서 실행)
    qgc_path = os.path.join(home_dir, 'QGroundControl-x86_64.AppImage')
    qgc_cmd = ExecuteProcess(
        cmd=['gnome-terminal', '--title=QGroundControl', '--', 'bash', '-c', f'{qgc_path}; exec bash'],
        output='screen'
    )

    # 2. MicroXRCEAgent (새 터미널 창에서 실행)
    agent_cmd = ExecuteProcess(
        cmd=['gnome-terminal', '--title=MicroXRCEAgent', '--', 'bash', '-c', 'MicroXRCEAgent udp4 -p 8888; exec bash'],
        output='screen'
    )

    # 3. PX4 SITL & Gazebo (새 터미널 창에서 환경변수와 함께 실행)
    px4_binary = os.path.join(home_dir, 'PX4-Autopilot/build/px4_sitl_default/bin/px4')
    px4_env_cmd = (
        f"PX4_SYS_AUTOSTART=4010 "
        f"PX4_SIM_MODEL=gz_x500_mono_cam "
        f"PX4_GZ_MODEL_POSE='1,1,0.1,0,0,0.9' "
        f"PX4_GZ_WORLD=test_world "
        f"{px4_binary}; exec bash"
    )

    px4_cmd = ExecuteProcess(
        cmd=['gnome-terminal', '--title=PX4_SITL', '--', 'bash', '-c', px4_env_cmd],
        output='screen'
    )

    # 지연 실행 설정 (QGC 3초 후 Agent -> 2초 후 PX4)
    delayed_agent = TimerAction(period=5.0, actions=[agent_cmd])
    delayed_px4 = TimerAction(period=10.0, actions=[px4_cmd])

    return LaunchDescription([
        qgc_cmd,
        delayed_agent,
        delayed_px4
    ])