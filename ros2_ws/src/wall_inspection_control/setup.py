from setuptools import find_packages, setup

package_name = 'wall_inspection_control'

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
    maintainer='ponsan',
    maintainer_email='7042608+ponsan@users.noreply.github.com',
    description='Zigzag scanning control for the wall inspection robot',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'scan_node = wall_inspection_control.scan_node:main'
        ],
    },
)
