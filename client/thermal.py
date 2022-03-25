import time
import board
import busio
import adafruit_mlx90640
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

PERIOD = 2

class Thermal(Node):
    def __init__(self):
        super().__init__('thermal_pub')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'thermal', 10)
        self.timer = self.create_timer(PERIOD, self.timer_callback)

        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        print("MLX addr detected on I2C")
        print([hex(i) for i in self.mlx.serial_number])
        
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
        self.frame = [0] * 768

    def timer_callback(self):
        try:
            self.mlx.getFrame(self.frame)
            msg = Float32MultiArray()
            msg.data = self.frame

            self.publisher_.publish(msg)
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
