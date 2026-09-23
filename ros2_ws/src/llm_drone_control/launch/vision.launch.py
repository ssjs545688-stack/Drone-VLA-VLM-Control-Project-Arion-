#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('llm_drone_control')
    default_config_path = os.path.join(pkg_share, 'config', 'detector.yaml')

    use_sim = LaunchConfiguration('use_sim', default='true')
    target_color = LaunchConfiguration('target_color', default='red')
    min_area = LaunchConfiguration('min_area', default='400')

    rgb_topic = LaunchConfiguration('rgb_topic', default='/camera')
    depth_topic = LaunchConfiguration('depth_topic', default='/depth_camera')
    camera_info_topic = LaunchConfiguration('camera_info_topic', default='/camera_info')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim', default_value='true'),
        DeclareLaunchArgument('target_color', default_value='red'),
        DeclareLaunchArgument('min_area', default_value='400'),
        DeclareLaunchArgument('rgb_topic', default_value='/camera'),
        DeclareLaunchArgument('depth_topic', default_value='/depth_camera'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera_info'),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='camera_bridge',
            output='screen',
            condition=IfCondition(use_sim),
            arguments=[
                # 월드 전용 clock 토픽 연결
                '/world/red_target/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/camera@sensor_msgs/msg/Image[gz.msgs.Image',
                '/depth_camera@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            ],
            remappings=[
                # /world/red_target/clock 을 ROS2 표준인 /clock 으로 변경
                ('/world/red_target/clock', '/clock'),
            ]
        ),

        Node(
            package='llm_drone_control',
            executable='object_detector',
            name='object_detector',
            output='screen',
            parameters=[
                default_config_path if os.path.exists(default_config_path) else {},
                {
                    'use_sim_time': True,
                    'target_color': target_color,
                    'min_area': min_area,
                    'rgb_topic': rgb_topic,
                    'depth_topic': depth_topic,
                    'camera_info_topic': camera_info_topic,
                }
            ],
        ),
    ])