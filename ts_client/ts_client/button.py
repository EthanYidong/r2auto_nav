import RPi.GPIO as GPIO
import rclpy
from rclpy.node import Node

from std_msgs.msg import Empty

PERIOD = 0.1

class Button(Node):
    # Initialize button publisher
    def __init__(self):
        super().__init__('button_pub')
        self.publisher_ = self.create_publisher(Empty, 'button', 10)
        self.timer = self.create_timer(PERIOD, self.timer_callback)

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_UP)
	# Used to prevent repeat signals
        self.pressed = False

        self.get_logger().info("Button publisher started")

    def timer_callback(self):
        try:
	    # Since the button is being used with a pullup resistor, no input means the button is currently being pressed.
            if not self.pressed and not GPIO.input(15):
                self.get_logger().info("Button pressed")
                self.publisher_.publish(Empty())
		# Mark button as pressed to prevent repeat signals
                self.pressed = True
            # If the button has been let go, reset the pressed state
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
