from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Path to camera parameters YAML
    config_file = os.path.join(
        FindPackageShare("hikrobot_aravis_ros_driver").find("hikrobot_aravis_ros_driver"),
        "config",
        "camera_params.yaml"
    )

    # Define nodes
    aravis_camera_node = Node(
        package="hikrobot_aravis_ros_driver",
        executable="aravis_camera_node",
        name="aravis_camera_node",
        respawn=True,
        output="screen",
        parameters=[config_file],
    )

    snapshot_node = Node(
        package="hikrobot_aravis_ros_driver",
        executable="snapshot_node",
        name="snapshot_node",
        respawn=False,
        output="screen",
    )

    # LaunchDescription holds the list of nodes/actions to start
    return LaunchDescription([
        aravis_camera_node,
        snapshot_node,
    ])
