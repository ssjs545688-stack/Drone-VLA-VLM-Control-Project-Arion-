#!/usr/bin/env python3
"""
drone_control.launch.py

llm_service와 flight_controller 노드를 각각 별도의 새 gnome-terminal 창으로 띄우는 launch 파일.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # 1. LLM 추론 노드 (새 터미널 창)
        Node(
            package='llm_drone_control',
            executable='llm_service',
            name='llm_service',
            output='screen',
            prefix=['gnome-terminal --title="[Node 1] LLM Service" -- '],
        ),

        # 2. PX4 비행 제어 노드 (새 터미널 창)
        Node(
            package='llm_drone_control',
            executable='flight_controller',
            name='flight_controller',
            output='screen',
            prefix=['gnome-terminal --title="[Node 2] Flight Controller" -- '],
        ),
    ])
