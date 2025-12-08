from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tree_template'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Marcus Rosette',
    maintainer_email='rosettem@oregonstate.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tree_template = tree_template.generate_trellis_collision_obj:main',
            'trunk_to_template_position = tree_template.trunk_to_template_position:main',
            'trunk_row_datum = tree_template.row_datum:main',
            'row_prior_mapper = tree_template.row_prior_mapper:main',
        ],
    },
)
