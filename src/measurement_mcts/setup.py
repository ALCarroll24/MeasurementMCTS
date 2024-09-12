from setuptools import find_packages, setup

package_name = 'measurement_mcts'

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
    maintainer='austin',
    maintainer_email='alcarroll@tamu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test_node = measurement_mcts.nodes.test_node:main',
            'unity_pose_to_tf = measurement_mcts.nodes.unity_pose_to_tf:main',
            'twist_action_to_unity = measurement_mcts.nodes.twist_action_to_unity:main',
        ],
    },
)
