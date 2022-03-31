import time

import RPi.GPIO as GPIO

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Int32

class Motors(Node):
    def __init__(self):
        super().__init__('motor_run')

        self._button_subscription = self.create_subscription(
            Int32,
            'motor',
            self.motor_callback,
            qos_profile_sensor_data)

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(18, GPIO.OUT)
        GPIO.setup(22, GPIO.OUT)

        self.flywheels = GPIO.PWM(18, 1000)
        self.servo = GPIO.PWM(22, 50)
        self.servo.ChangeDutyCycle(7.5)
        self.get_logger().info("Motor runner started")

    def motor_callback(self, msg):
        cmd = msg.data
        if(cmd == 0):
            self.flywheels.start(0)
        elif(cmd == 1):
            for i in range(101):
                self.flywheels.start(i)
                time.sleep(0.02)
        elif(cmd == 2):
            self.servo.ChangeDutyCycle(2.5)
            time.sleep(0.5)
            self.servo.ChangeDutyCycle(7.5)
        pass
                
def main(args=None):
	rclpy.init(args=args)
	motor_pub = Motors()
	rclpy.spin(motor_pub)
	motor_pub.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
