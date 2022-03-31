#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade

# Set ROS_DOMAIN_ID on login
echo 'export ROS_DOMAIN_ID=35 #TURTLEBOT3' >> ~/.bashrc
source ~/.bashrc

# Update LDS driver
sudo apt-get install libudev-dev
cd ~/turtlebot3_ws/src
git clone -b ros2-devel https://github.com/ROBOTIS-GIT/ld08_driver.git
cd ~/turtlebot3_ws/src/turtlebot3 && git pull
rm -r turtlebot3_cartographer turtlebot3_navigation2
cd ~/turtlebot3_ws && colcon build --symlink-install

echo 'export LDS_MODEL=LDS-01' >> ~/.bashrc
source ~/.bashrc

# Install i2c tools
sudo apt-get install i2c-tools

# Manage i2c permissions
sudo groupadd i2c
echo "KERNEL==\"i2c-[0-9]*\", GROUP=\"i2c\"" | sudo tee -a /etc/udev/rules.d/10-local_i2c_group.rules
sudo usermod -a -G i2c ubuntu
sudo udevadm control --reload-rules
udevadm trigger

# Manage GPIO permissions
sudo groupadd gpio
sudo usermod -a -G gpio ubuntu

# Final bashrc setup
echo "alias rosbu='ros2 launch turtlebot3_bringup robot.launch.py'" >> ~/.bashrc
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
echo "export OPENCR_MODEL=burger" >> ~/.bashrc
source ~/.bashrc

# Install dependancies
sudo apt-get install python3-pip rpi.gpio
pip3 install adafruit-circuitpython-mlx90640 spidev

# Install ros2 packages
cd ~/turtlebot_ws/src
git clone https://github.com/EthanYidong/r2auto_nav

cd ~/turtlebot_ws
colcon build --packages-select ts_client
source ~/.bashrc

echo "Done setting up, please reboot."
