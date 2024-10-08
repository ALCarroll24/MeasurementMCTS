from setuptools import find_packages, setup

package_name = 'corner_detection'

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
            'pca_node = corner_detection.pca_node:main',
            'ransac_node = corner_detection.ransac_node:main',
            'ransac_3d_node = corner_detection.ransac_3d_node:main',
        ],
    },
)
