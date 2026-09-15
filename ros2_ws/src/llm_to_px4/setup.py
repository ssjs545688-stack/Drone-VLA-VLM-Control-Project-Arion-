from setuptools import find_packages, setup

package_name = 'llm_to_px4'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            "llm_service = llm_to_px4.llm_service:main",
            "llm_to_px4_converter = llm_to_px4.llm_to_px4_converter:main",
            "llm_px4_controller = llm_to_px4.llm_px4_controller:main",
        ],
    },
)
