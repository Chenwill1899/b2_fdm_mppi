from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("b2_fdm_mppi"), "config", "fdm_mppi.yaml"]
    )

    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to the FDM MPPI simulation config YAML.",
            ),
            Node(
                package="b2_fdm_mppi",
                executable="fdm_mppi_sim_node",
                name="b2_fdm_mppi",
                output="screen",
                parameters=[{"config_file": config_file}],
            ),
        ]
    )
