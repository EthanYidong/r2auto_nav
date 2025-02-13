# r2auto_nav

## ts_server

This directory contains the ros2 package that should be run on the ros2 server machine.

- `wallfollow.py` Contains the main bulk of the code for the robot. It manages state to navigate the bot through the maze, manage signals from various sensors on the bot, and orchestrate the completion of the task
- `conv.py` Contain helper code to assist in transformations of odometry data into base_link, required because the tf2 python library is not fully updated to work with ros2.

### Running

On the server side, two processes must be run:
1. `ros2 launch turtlebot3_cartographer turtlebot3_cartographer.launch.py`
2. `ros2 run ts_server wallfollow`

## ts_client

This directory contains the ros2 package that should be run on the ros2 client (i.e. the turtlebot).
- `launch.py` Defines all scripts that should be launched from this package
- `button.py` Publishes Empty messages to `button` whenever the button is pressed.
- `nfc.py` Publishes Empty messages to `nfc` while a NFC tag is detected
- `thermal.py` Publishes Float32MultiArray messages to `thermal` periodically once values are read from the thermal camera.
- `motors.py` Subscribes to the `/motor` topic, in order to allow the server to control the firing of balls.


### Running

On the client side, two processes must be run:
1. `ros2 launch turtlebot3_bringup robot.launch.py`
2. `ros2 launch ~/turtlebot_ws/src/r2auto_nav/ts_client/launch.py`

## test_scripts

This directory contains standalone scripts for testing and calibrating the systems.
- `firing.py` Run to test flywheel motors, servo motor, and firing sequence.
- `calibrate_thermals.py` Run to test thermal sensor and calibrate based on the temperature of the target.

## scripts

This directory contains setup scripts for installing the code onto a new machine.
- `bootstrap_client.py` Updates packages, installs required libraries, and builds the client side code.
- `bootstrap_server.py` Installs required libraries, and builds the server side code.
