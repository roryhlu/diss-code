"""RePAIR simulation — ROS2 MoveIt2 grasp execution package."""

from setuptools import find_packages, setup

package_name = "repair_simulation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", [
            "launch/camera_pipeline.launch.py",
            "launch/grasp_executor.launch.py",
            "launch/full_pipeline.launch.py",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Rory Hlustik",
    maintainer_email="rory.hlustik@outlook.com",
    description="RePAIR simulation — MoveIt2 grasp execution node",
    license="MIT",
    entry_points={
        "console_scripts": [
            "grasp_executor = repair_simulation.grasp_executor:main",
            "camera_capture = repair_simulation.camera_capture:main",
            "perception_bridge = repair_simulation.perception_bridge:main",
        ],
    },
)
