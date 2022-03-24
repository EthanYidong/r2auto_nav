import sys

sys.path.insert(0, "/home/ubuntu/labcode/EG2310_NFC")

from pn532 import *
import RPi.GPIO as GPIO
import time
import rclpy
from rclpy.node import Node

from std_msgs.msg import Empty

PERIOD = 0.1

class Thermal(Node):
    def __init__(self):
        super().__init__('nfc_pub')
        self.publisher_ = self.create_publisher(Empty, 'nfc', 10)
        self.timer = self.create_timer(PERIOD, self.timer_callback)

        self.pn532 = PN532_I2C(debug=False, reset=20, req=16)
        ic, ver, rev, support = self.pn532.get_firmware_version()
        print('Found PN532 with firmware version: {0}.{1}'.format(ver, rev))

        # Configure PN532 to communicate with MiFare cards
        self.pn532.SAM_configuration()

    def timer_callback(self):
        try:
            uid = self.pn532.read_passive_target(timeout=0.1)
            if uid is not None:
                self.publisher_.publish(Empty())
        except Exception as e:
            if e is not KeyboardInterrupt:
                return
                
def main(args=None):
	rclpy.init(args=args)
	thermal_pub = Thermal()
	rclpy.spin(thermal_pub)
	thermal_pub.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
