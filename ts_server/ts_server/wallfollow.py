from collections import deque
from enum import Enum
import math
import cmath
import time
import traceback

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray, Int32, Empty

import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation

from .conv import *

# How often to update and re-execute state
UPDATE_PERIOD = 0.1

# Robot Information
TURNING_RADIUS = 0.12
PADDING = 0.05
MARGIN = TURNING_RADIUS + PADDING * 2

ROBOT_LEFT = 0.15
ROBOT_RIGHT = -0.15
ROBOT_FRONT = 0.12
ROBOT_BACK = -0.08
ROBOT_LEN = ROBOT_FRONT - ROBOT_BACK

THERMAL_X = 0.09
THERMAL_Y = -0.04

THERMAL_ANGLE_BOUNDS = 55.0
THERMAL_FOV = 110.0
THERMAL_WIDTH = 32
THERMAL_H_RANGE = range(8, 16)

TEMP_THRESHOLD = 35
TEMP_PERCENTILE = 75

# How many balls we load
BALLS_LOADED = 5

# How close to target before we taper speed
TURN_TAPER_THRESHOLD = 20.0
MOVE_TAPER_THRESHOLD = 0.5

# How close to target before we stop
TURNING_THRESHOLD = 5
MOVING_THRESHOLD = MARGIN

# How fast to go
TURNING_VEL = 1.82
SLOW_TURNING_VEL = 0.5
MIN_TURNING_VEL = 0.2
FORWARD_VEL = 0.21
MIN_FORWARD_VEL = 0.1

DEBUG = True

NBORS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

State = Enum('State', 'BEGIN SEEK TURN_LEFT TURN_RIGHT LOCKED_FORWARD FORWARD LOADING SEEK_TARGET MOVE_TARGET FIRING DONE')

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def angle_to(from_x, from_y, to_x, to_y):
    return math.atan2(to_y - from_y, to_x - from_x)

def dist_to(from_x, from_y, to_x, to_y):
    return math.sqrt((to_y - from_y) ** 2 + (to_x - from_x) ** 2)

def angle_diff(f, t):
    return (360 + t - f) % 360

def achieved_angle(diff):
    return diff < TURNING_THRESHOLD or diff > 360 - TURNING_THRESHOLD

def turn_towards(yaw, target):
    twist = Twist()
    diff = angle_diff(yaw, target)
    
    if achieved_angle(diff):
        return twist, True
    if(diff < 180):
        twist.angular.z = taper_turn(diff)
    else:
        twist.angular.z = -taper_turn(360 - diff)
    return twist, False

def taper_turn(delta):
    if delta > TURN_TAPER_THRESHOLD:
        return TURNING_VEL
    else:
        return MIN_TURNING_VEL

def taper_move(delta):
    if delta > MOVE_TAPER_THRESHOLD:
        return FORWARD_VEL
    else:
        return MIN_FORWARD_VEL

def generate_surroundings(laser_range):
    laser_surroundings = []
    for deg, dist in enumerate(laser_range):
        ang = np.deg2rad(deg)
        if dist != np.inf:
            rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            p = np.array([dist, 0])
            p_rot = np.dot(rot, p)
            laser_surroundings.append(p_rot)
    return laser_surroundings

def mlx_index(angle):
    if angle >= -THERMAL_ANGLE_BOUNDS and angle <= THERMAL_ANGLE_BOUNDS:
        index = int(round((angle + THERMAL_FOV / 2) / THERMAL_FOV * THERMAL_WIDTH))
        if index >= 0 and index < THERMAL_WIDTH:
            return index
    return None

class AutoNav(Node):
    def __init__(self):
        super().__init__('auto_nav')
        self.publisher_ = self.create_publisher(Twist,'cmd_vel',10)
        self.motor_publisher_ = self.create_publisher(Int32,'motor',10)
        

        self._odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        
        self._occ_subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.occ_callback,
            qos_profile_sensor_data)
        
        self._scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile_sensor_data)

        self._therm_subscription = self.create_subscription(
            Float32MultiArray,
            'thermal',
            self.thermal_callback,
            qos_profile_sensor_data)
        
        self._nfc_subscription = self.create_subscription(
            Empty,
            'nfc',
            self.nfc_callback,
            qos_profile_sensor_data)
        
        self._button_subscription = self.create_subscription(
            Empty,
            'button',
            self.button_callback,
            qos_profile_sensor_data)

        self._timer = self.create_timer(UPDATE_PERIOD, self.timer_callback)

        self.tfBuffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=60))
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.laser_range = None
        self.laser_surroundings = None
        self.thermal_surroundings = None
        self.scan_data = None
        self.occdata = None
        self.map_info = None
        self.odom = None
        self.base_link = None
        self.current_twist = None
        self.thermal = None
        self.seen_nfc = False
        # TODO: change to 0 after debugging
        self.loaded = 1

        self.state = State.BEGIN
        self.state_data = {}

    def current_location(self):
        if self.odom is None:
            return None, None, None
        odom_pose = PoseStamped(
            header=self.odom.header,
            pose=self.odom.pose.pose,
        )

        last_error = None
        for _tries in range(10):
            try:
                base_link = self.tfBuffer.transform(odom_pose, 'map')
            except Exception as e:
                last_error = e
                if odom_pose.header.stamp.nanosec >= 100000000:
                    odom_pose.header.stamp.nanosec -= 100000000
                else:
                    odom_pose.header.stamp.sec -= 1
                    odom_pose.header.stamp.nanosec += 900000000
                if e is KeyboardInterrupt:
                    raise e
            else:
                break
        else:
            print("failed to get location:", last_error)
            return None, None, None
        _roll, _pitch, yaw = euler_from_quaternion(base_link.pose.orientation.x, base_link.pose.orientation.y, base_link.pose.orientation.z, base_link.pose.orientation.w)
        return base_link.pose.position.x, base_link.pose.position.y, np.rad2deg(yaw)

    def blocked(self):
        for x, y in self.laser_surroundings:
            if y >= ROBOT_RIGHT and y <= ROBOT_LEFT and x >= ROBOT_FRONT and x <= ROBOT_FRONT + PADDING * 2:
                return True
        return False

    def empty_right(self):
        for x, y in self.laser_surroundings:
            if x <= ROBOT_FRONT + PADDING and x >= ROBOT_BACK - PADDING and y <= ROBOT_RIGHT and y >= ROBOT_RIGHT - PADDING * 2:
                return False
        return True
    
    def to_map_coords(self, x, y):
        map_position = self.map_info.origin.position
        map_orientation = self.map_info.origin.orientation

        map_rotation = Rotation.from_quat([map_orientation.x, map_orientation.y, map_orientation.z, map_orientation.w])
        offset_x = x - map_position.x
        offset_y = y - map_position.y

        rotated_offset = map_rotation.inv().apply(np.array([offset_x, offset_y, 0]))
        map_x = int(rotated_offset[1] // self.map_info.resolution)
        map_y = int(rotated_offset[0] // self.map_info.resolution)

        return map_x, map_y

    def valid_point(self, x, y):
        return x >= 0 and x < np.size(self.occdata, 0) and y >= 0 and y < np.size(self.occdata, 1)

    def turning_margin(self):
        return TURNING_RADIUS + self.map_info.resolution

    def odom_callback(self, msg):
        self.odom = msg

    def occ_callback(self, msg):
        msgdata = np.array(msg.data)
        oc2 = msgdata
        self.occdata = oc2.reshape(msg.info.height,msg.info.width)
        self.map_info = msg.info

    def scan_callback(self, msg):
        self.laser_range = np.array(msg.ranges)
        self.laser_range[self.laser_range==0] = np.inf
        self.scan_data = msg
        self.laser_surroundings = generate_surroundings(self.laser_range)

        self.update_state_scan()

    def thermal_callback(self, msg):
        msgdata = np.array(msg.data)
        self.thermal = msgdata.reshape([24,32])
        self.thermal_surroundings = []
        if self.laser_surroundings is not None:
            for x, y in self.laser_surroundings:
                dx = x - THERMAL_X
                dy = y - THERMAL_Y

                # Swap x and y because we are measuring angle with respect to the y axis
                ang = np.rad2deg(math.atan2(dy, dx))
                index = mlx_index(ang)
                if index is None:
                    continue
                temps = []
                for h in THERMAL_H_RANGE:
                    temps.append(self.thermal[h][index])
                temp_per = np.percentile(temps, TEMP_PERCENTILE)
                if temp_per >= TEMP_THRESHOLD:
                    self.thermal_surroundings.append((x, y, temp_per))
        self.update_state_thermal()

    def nfc_callback(self, _msg):
        self.update_state_nfc()

    def button_callback(self, _msg):
        self.update_state_button()

    def timer_callback(self):
        self.execute_state()

    def change_state(self, new_state, **kwargs):
        x, y, yaw = self.current_location()
        if x is None:
            self.stopbot()
            return

        if new_state == State.SEEK:
            closest_idx = np.nanargmin(self.laser_range)
            self.state_data = { "target_angle": (yaw + closest_idx) % 360 }
        elif new_state == State.TURN_RIGHT:
            self.state_data = { "target_angle": (yaw + 270) % 360 }
        elif new_state == State.LOCKED_FORWARD:
            self.state_data = { "dist": kwargs.get("dist"), "start": (x, y) }
        elif new_state == State.LOADING:
            self.state_data = {
                "previous_state": self.state,
                "previous_state_data": self.state_data,
            }
        elif new_state == State.SEEK_TARGET:
            self.state_data = { "target_angle": (yaw + kwargs.get("angle") ) }
        else:
            self.state_data = {}
        self.state = new_state
        print("state", self.state, self.state_data)

    def update_state_nfc(self):
        if not self.seen_nfc:
            self.seen_nfc = True
            self.change_state(State.LOADING)
            self.stopbot()

    def update_state_button(self):
        if self.state == State.LOADING:
            print("Button pressed, resuming.")
            self.state = self.state_data["previous_state"]
            self.state_data = self.state_data["previous_state_data"]
            self.loaded = BALLS_LOADED

    def update_state_scan(self):
        x, y, yaw = self.current_location()
        if x is None:
            self.stopbot()
            return
        if self.state == State.BEGIN:
            self.change_state(State.SEEK)
        elif self.state == State.SEEK:
            _, achieved = turn_towards(yaw, self.state_data["target_angle"])
            if achieved:
                self.change_state(State.LOCKED_FORWARD)
        elif self.state == State.TURN_LEFT:
            if not self.blocked():
                self.change_state(State.FORWARD)
        elif self.state == State.TURN_RIGHT:
            _, achieved = turn_towards(yaw, self.state_data["target_angle"])
            if achieved:
                self.change_state(State.LOCKED_FORWARD, dist=ROBOT_LEN)
        elif self.state == State.FORWARD:
            if self.empty_right():
                self.change_state(State.TURN_RIGHT)
            elif self.blocked():
                self.change_state(State.TURN_LEFT)
        elif self.state == State.LOCKED_FORWARD:
            if self.blocked():
                self.change_state(State.TURN_LEFT)
            elif self.state_data["dist"] is not None and dist_to(*self.state_data["start"], x, y) >= self.state_data["dist"]:
                self.change_state(State.FORWARD)
        elif self.state == State.SEEK_TARGET:
            _, achieved = turn_towards(yaw, self.state_data["target_angle"])
            if achieved:
                self.change_state(State.MOVE_TARGET)
        elif self.state == State.MOVE_TARGET:
            if self.blocked():
                self.change_state(State.FIRING)

    def update_state_thermal(self):
        if not self.thermal_surroundings:
            return

        # Calculate average location of hot points
        sx = 0.0
        sy = 0.0
        for xt, yt, temp in self.thermal_surroundings:
            sx += xt
            sy += yt
            
        ax = sx / len(self.thermal_surroundings)
        ay = sy / len(self.thermal_surroundings)

        print("Found heat at:", ax, ay)

        if self.loaded != 0 and self.state != State.SEEK_TARGET:
            self.change_state(State.SEEK_TARGET, angle = np.rad2deg(math.atan2(ay, ax)))

            
    def execute_state(self):
        x, y, yaw = self.current_location()
        if x is None:
            self.stopbot()
            return
        twist = Twist()
        twist.angular.z = 0.0
        twist.linear.x = 0.0
        if self.state == State.SEEK:
            twist, _ = turn_towards(yaw, self.state_data["target_angle"])
        elif self.state == State.TURN_LEFT:
            twist.angular.z = SLOW_TURNING_VEL
        elif self.state == State.TURN_RIGHT:
            twist, _ = turn_towards(yaw, self.state_data["target_angle"])
        elif self.state == State.FORWARD:
            twist.linear.x = taper_move(self.laser_range[0])
        elif self.state == State.LOCKED_FORWARD:
            twist.linear.x = taper_move(self.laser_range[0])
        elif self.state == State.SEEK_TARGET:
            twist, _ = turn_towards(yaw, self.state_data["target_angle"])
        elif self.state == State.MOVE_TARGET:
            twist.linear.x = taper_move(self.laser_range[0])
        elif self.state == State.FIRING:
            self.stopbot()
            time.sleep(1)
            self.motor_publisher_.publish(Int32(data=1))
            time.sleep(10)
            while self.loaded > 0:
                self.motor_publisher_.publish(Int32(data=2))
                time.sleep(1)
                self.loaded -= 1
            self.motor_publisher_.publish(Int32(data=0))
            self.change_state(State.DONE)

        if self.current_twist != twist:
            self.publisher_.publish(twist)
            self.current_twist = twist

    def mover(self):
        try:
            while rclpy.ok():              
                rclpy.spin(self)
        except Exception as e:
            traceback.print_exc()
        finally:
            self.stopbot()

    def stopbot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    auto_nav = AutoNav()
    auto_nav.mover()

    auto_nav.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
