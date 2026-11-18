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
        (os.path.join('share', package_name, 'config', 'nav'), glob('config/nav/*.yaml')),
        (os.path.join('share', package_name, 'config', 'nav_2025'), glob('config/nav_2025/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config', 'robot'), glob('config/robot/*.yaml')),
        (os.path.join('share', package_name, 'config', 'camera_extrinsics'), glob('config/camera_extrinsics/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'scikit-image', 'PyYAML', 'utm'],
    zip_safe=True,
    maintainer='Marcus Rosette',
    maintainer_email='rosettem@oregonstate.edu',
    description='Row-specialised FastSLAM trunk mapping and trellis template placement for orchard rows',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tree_template = tree_template.generate_trellis_collision_obj:main',
            'row_fast_slam = tree_template.row_fast_slam:main',
            'slam_odom_correction_tf = tree_template.slam_odom_correction_tf:main',
            'odom_reframer_calibrated = tree_template.odom_reframer_calibrated:main',
            'trunk_detection_relay = tree_template.trunk_detection_relay:main',
            'save_best_particle_map = tree_template.save_best_particle_map:main',
            'fast_slam_rgbd_extraction = tree_template.fast_slam_rgbd_extraction:main',
            'calibrate_depth_to_color = tree_template.calibrate_depth_to_color:main',
            'trellis_from_rgbd_extraction = tree_template.trellis_from_rgbd_extraction:main',
            'fix_cov_relay = tree_template.fix_cov_relay:main',
            'gps_course_heading = tree_template.gps_course_heading:main',
        ],
    },
)
