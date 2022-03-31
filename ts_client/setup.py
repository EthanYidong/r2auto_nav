from setuptools import setup

package_name = 'ts_client'
submodules = 'ts_client/pn532'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ethan',
    maintainer_email='ethan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'button = ts_client.button:main',
            'nfc = ts_client.nfc:main',
            'thermal = ts_client.thermal:main',
            'motors = ts_client.motors:main',
        ],
    },
)
