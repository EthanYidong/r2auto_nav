import sys

import RPi.GPIO as GPIO
import time
import rclpy
from rclpy.node import Node

from std_msgs.msg import Empty

PERIOD = 0.1

class Button(Node):
    def __init__(self):
        super().__init__('button_pub')
        self.publisher_ = self.create_publisher(Empty, 'button', 10)
        self.timer = self.create_timer(PERIOD, self.timer_callback)

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.pressed = False

        print("Button publisher started")

    def timer_callback(self):
        try:
            if not self.pressed and not GPIO.input(15):
                print("Button pressed")
                self.publisher_.publish(Empty())
                self.pressed = True
            if GPIO.input(15):
                self.pressed = False
        except Exception as e:
            if e is not KeyboardInterrupt:
                return
                
def main(args=None):
	rclpy.init(args=args)
	button_pub = Button()
	rclpy.spin(button_pub)
	button_pub.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
