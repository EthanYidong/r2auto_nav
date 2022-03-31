# r2auto_nav

## ts_server

This directory contains the ros2 package that should be run on the ros2 server machine.

- `wallfollow.py` Contains the main bulk of the code for the robot. It manages state to navigate the bot through the maze, manage signals from various sensors on the bot, and orchestrate the completion of the task
- `conv.py` Contain helper code to assist in transformations of odometry data into base_link, required because the tf2 python library is not fully updated to work with ros2.

## ts_client

This directory contains the ros2 package that should be run on the ros2 client (i.e. the turtlebot).

- `launch.py` Defines all scripts that should be launched from this package
- `button.py` Publishes Empty messages to `button` whenever the button is pressed.
- `nfc.py` Publishes Empty messages to `nfc` while a NFC tag is detected
- `thermal.py` Publishes Float32MultiArray messages to `thermal` periodically once values are read from the thermal camera.