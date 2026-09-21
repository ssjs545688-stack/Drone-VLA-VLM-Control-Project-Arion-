from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'llm_drone_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share",package_name,"config"),glob("config/*.yaml")),
        (os.path.join("share",package_name,"launch"),glob("launch/*.launch.py")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hkit',
    maintainer_email='274044817+crab-ally@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "llm_service = llm_drone_control.llm_service:main",
            "flight_controller = llm_drone_control.flight_controller:main",
            "stt_node = llm_drone_control.stt_node:main",
            "smartphone_bridge = llm_drone_control.smartphone_bridge:main",
        ],
    },
)
