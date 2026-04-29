from glob import glob
import sys

from setuptools import Command, find_packages, setup

package_name = "b2_fdm_mppi"


class PyTestCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import pytest

        sys.exit(pytest.main(["-q", "tests"]))


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    cmdclass={"test": PyTestCommand},
    maintainer="Chenwill1899",
    maintainer_email="chenwill1899@example.com",
    description="ROS 2 Humble package for the FDM MPPI internal simulation node.",
    license="Vim",
    entry_points={
        "console_scripts": [
            "fdm_mppi_sim_node = b2_fdm_mppi.fdm_mppi_sim_node:main",
        ],
    },
)
