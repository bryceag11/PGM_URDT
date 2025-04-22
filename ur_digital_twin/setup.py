from setuptools import setup
import os
from glob import glob

package_name = 'ur_digital_twin'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        # Add web assets
        (os.path.join('share', package_name, 'web'), glob('ur_digital_twin/web/*')),
    ],
    install_requires=[
        'setuptools',
        'filterpy',
        'matplotlib',
        'scipy',
        'numpy',
        'pgmpy',
        'pomegranate',
        'tabulate',
        'webbrowser',
        'psutil',
    ],
    zip_safe=True,
    maintainer='Bryce Grant',
    maintainer_email='bag100@case.edu',
    description='Digital Twin for UR Robots using Probabilistic Graphical Models',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'digital_twin_node = ur_digital_twin.digital_twin_node:main',
            'web_dashboard = ur_digital_twin.web_dashboard:main',
        ],
    },
)