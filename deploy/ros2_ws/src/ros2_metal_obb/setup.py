from setuptools import find_packages, setup

package_name = "ros2_metal_obb"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "metal_obb_node = ros2_metal_obb.metal_obb_node:main",
        ],
    },
)
