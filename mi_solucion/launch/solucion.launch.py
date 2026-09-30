import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mi_solucion',
            executable='auto_docker',
            name='auto_docker',
            output='screen',
            parameters=[{'use_sim_time': True}]
        )
    ])
