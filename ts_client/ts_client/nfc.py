import time

from .pn532 import *
import RPi.GPIO as GPIO

import rclpy
from rclpy.node import Node

from std_msgs.msg import Empty

PERIOD = 0.1

class NFC(Node):
    def __init__(self):
        super().__init__('nfc_pub')
        self.publisher_ = self.create_publisher(Empty, 'nfc', 10)
        self.timer = self.create_timer(PERIOD, self.timer_callback)

	# Initialize PN532 using I2C connection
        self.pn532 = PN532_I2C(debug=False, reset=20, req=16)
        ic, ver, rev, support = self.pn532.get_firmware_version()
        self.get_logger().info('Found PN532 with firmware version: {0}.{1}'.format(ver, rev))

        # Configure PN532 to communicate with MiFare cards
        self.pn532.SAM_configuration()

    def timer_callback(self):
        try:
	    # If any uid is returned, that means a card has been found. Report to the server by publishing a message.
            uid = self.pn532.read_passive_target(timeout=0.1)
            if uid is not None:
                self.get_logger().info("NFC Detected")
                self.publisher_.publish(Empty())
        except Exception as e:
            if e is not KeyboardInterrupt:
                return
                
def main(args=None):
	rclpy.init(args=args)
	nfc_pub = NFC()
	rclpy.spin(nfc_pub)
	nfc_pub.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
