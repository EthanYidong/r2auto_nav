import time
import threading
import pigpio
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
        self.pi = pigpio.pi()

        GPIO.setmode(GPIO.BOARD)
        # Use pigpio for flywheel motors to utilize hardware PWM
        self.pi.set_mode(18, pigpio.OUTPUT)
        
        # Use RPi.GPIO for servo motor
        GPIO.setup(22, GPIO.OUT)

        self.servo = GPIO.PWM(22, 50)
        self.servo.start(7.5)

        self.motor_thread = None

        self.get_logger().info("Motor runner started")

    def motor_callback(self, msg):
        cmd = msg.data
        # Use numbers as an enum rather than creating custom message through a package.
        # 0 = Stop flywheels
        # 1 = Start flywheels
        # 2 = Run servo once
        if(cmd == 0):
            # Set duty cycle to 0
            self.pi.hardware_PWM(18, 10000, 0)
        elif(cmd == 1):
            # Slowly increase duty cycle to prevent surge current
            for i in range(100):
                self.pi.hardware_PWM(18, 10000, 10000 * i)
                time.sleep(0.02)
        elif(cmd == 2):
            # Move servo backwards to allow ball to fall into tube, then push forward, and finally reset to blocked position.
            self.servo.ChangeDutyCycle(10.5)
            time.sleep(0.5)
            self.servo.ChangeDutyCycle(5.5)
            time.sleep(0.4)
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
