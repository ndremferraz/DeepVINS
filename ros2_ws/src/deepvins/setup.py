from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'deepvins'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'models'),
         glob(os.path.join('models', '*.pth')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andreferraz',
    maintainer_email='andreferraz@gwu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'deepvins_node = deepvins.deepvins_node:main'
        ],
    },
)
