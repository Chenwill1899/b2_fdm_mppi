from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_profile = PathJoinSubstitution(
        [FindPackageShare("b2_fdm_mppi"), "configs", "mujoco_scout.yaml"]
    )

    profile = LaunchConfiguration("profile")
    controller = LaunchConfiguration("controller")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "profile",
                default_value=default_profile,
                description="Path to the MuJoCo Scout closed-loop profile YAML.",
            ),
            DeclareLaunchArgument(
                "controller",
                default_value="nominal_numpy",
                description="Controller name from the profile.",
            ),
            Node(
                package="b2_fdm_mppi",
                executable="fdm_mppi",
                name="fdm_mppi_mujoco_closed_loop",
                output="screen",
                arguments=["mujoco-closed-loop", "--profile", profile, "--controller", controller],
            ),
        ]
    )
