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

import random

import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from .conv import *

import sys

sys.setrecursionlimit(1000000)

# How often to update and re-execute state
UPDATE_PERIOD = 0.02

# Robot Information
ROBOT_FRONT = 0.14
ROBOT_LEFT = 0.12
ROBOT_RIGHT = -0.12
ROBOT_BACK = -0.09
ROBOT_LEN = ROBOT_FRONT - ROBOT_BACK

# Surroundings definition

# How far ahead to check for walls
LOOKAHEAD = 0.10
ADJUSTLOOKAHEAD = 0.20
ADJUSTLOOKRIGHT = 0.05
LOOKRIGHT = 0.20

# Wall distance to the right check
TOO_CLOSE = 0.02
TOO_FAR = 0.10

# Location of thermal camera relative to the center of the lidar sensor.
THERMAL_X = 0.09
THERMAL_Y = -0.04

# FOV of the thermal camera, should be kept constant
THERMAL_FOV = 110.0
# Width of the thermal image, should be kept constant
THERMAL_WIDTH = 32

# Which angles to consider in the thermal camera's FOV
THERMAL_ANGLE_BOUNDS = 55.0
# Range of heights in the thermal image to consider
THERMAL_H_RANGE = range(8, 16)
# Threshold temperature: must adjust!
TEMP_THRESHOLD = 40
# Which percentile of temperatures to consider
TEMP_PERCENTILE = 50

# How many balls we load
BALLS_LOADED = 3

# How close to target before we taper speed
TURN_TAPER_THRESHOLD = 10.0
MOVE_TAPER_THRESHOLD = 0.5

# How close to target angle before we stop
TURNING_THRESHOLD = 2
PRECISE_THRESHOLD = 0.2

# How close to the known sighting of NFC tag do we have to be to stop
RETRACE_MARGIN = 0.05

# How fast to go
TURNING_VEL = 1.4
ADJUST_TURNING_VEL = 0.3
PRECISE_TURNING_VEL = 0.5
FORWARD_VEL = 0.22

# Threshold
MAP_THRESHOLD = 50

# Randomly turn in some direction after map has been completed. Enable if there are disconnected walls.
RANDOMIZE = False

# Keep on for debug purposes, else disables printing and outputting of maps.
DEBUG = True

# Neighbors to check for DFS floodfill for map completion
NBORS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

State = Enum('State', 'BEGIN SEEK FORWARD TURN_LEFT TURN_RIGHT LOCKED_FORWARD LOADING SEEK_TARGET MOVE_TARGET FIRING DONE')

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

def achieved_angle(diff, precise = False):
    if precise:
        return diff < PRECISE_THRESHOLD or diff > 360.0 - PRECISE_THRESHOLD
    return diff < TURNING_THRESHOLD or diff > 360.0 - TURNING_THRESHOLD

def turn_towards(yaw, target, precise = False):
    twist = Twist()
    diff = angle_diff(yaw, target)
    
    if achieved_angle(diff, precise):
        return twist, True
    if(diff < 180):
        twist.angular.z = taper_turn(diff, precise)
    else:
        twist.angular.z = -taper_turn(360 - diff, precise)
    return twist, False

def scale_right_turn(delta):
    twist = Twist()
    twist.linear.x = FORWARD_VEL
    twist.angular.z = -(min(delta, 0.03) / 0.03) * TURNING_VEL
    return twist

def scale_left_turn(delta):
    twist = Twist()
    twist.linear.x = FORWARD_VEL
    twist.angular.z = (min(delta, 0.03) / 0.03) * TURNING_VEL
    if delta >= TOO_CLOSE:
        twist.linear.x = 0.0
    return twist

def taper_turn(delta, precise = False):
    if delta > TURN_TAPER_THRESHOLD:
        if precise:
            return PRECISE_TURNING_VEL
        return TURNING_VEL
    else:
        return ADJUST_TURNING_VEL

def taper_move(delta):
    if delta > MOVE_TAPER_THRESHOLD:
        return FORWARD_VEL
    else:
        return ADJUST_TURNING_VEL

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
        self.scan_data = None

        self.thermal_surroundings = None

        self.occdata = None
        self.raycast = None
        self.area = 0

        self.map_info = None

        self.odom = None
        self.base_link = None
        self.x = None
        self.y = None
        self.yaw = None

        self.current_twist = None

        self.thermal = None
        self.seen_nfc = False
        self.toured = False
        self.toured_infront = False
        self.nfc_loc = None
        self.maps_seen = 0
        # TODO: change to 0 after debugging
        self.loaded = 0

        self.state = State.BEGIN
        self.state_data = {}

    def current_location(self):
        start = time.time_ns()
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
        
        ms = (time.time_ns() - start) / 1000000
        if ms > 100:
            print(f"found time in {ms} ms")
        return base_link.pose.position.x, base_link.pose.position.y, np.rad2deg(yaw)

    def front(self):
        backmost = math.inf
        for x, y in self.laser_surroundings:
            if x >= ROBOT_FRONT and y <= ROBOT_LEFT and y >= ROBOT_RIGHT:
                backmost = min(x, backmost)
        return backmost - ROBOT_FRONT

    def front_right_half(self):
        backmost = math.inf
        for x, y in self.laser_surroundings:
            if x >= ROBOT_FRONT and y <= 0 and y >= ROBOT_RIGHT - ADJUSTLOOKRIGHT:
                backmost = min(x, backmost)
        return backmost - ROBOT_FRONT
    
    def right(self):
        leftmost = -math.inf
        for x, y in self.laser_surroundings:
            if x <= ROBOT_FRONT and x >= ROBOT_BACK and y <= ROBOT_RIGHT:
                leftmost = max(y, leftmost)
        return ROBOT_RIGHT - leftmost
    
    def to_map_coords(self, x, y, yaw):
        map_position = self.map_info.origin.position
        map_orientation = self.map_info.origin.orientation

        map_rotation = Rotation.from_quat([map_orientation.x, map_orientation.y, map_orientation.z, map_orientation.w])
        offset_x = x - map_position.x
        offset_y = y - map_position.y

        rotated_offset = map_rotation.inv().apply(np.array([offset_x, offset_y, 0]))
        map_x = int(rotated_offset[1] // self.map_info.resolution)
        map_y = int(rotated_offset[0] // self.map_info.resolution)
        _, _, offset_yaw = euler_from_quaternion(map_orientation.x, map_orientation.y, map_orientation.z, map_orientation.w)


        return map_x, map_y, yaw - offset_yaw

    def valid_point(self, x, y):
        return x >= 0 and x < np.size(self.occdata, 0) and y >= 0 and y < np.size(self.occdata, 1)

    def floodfill_edges(self, x, y):
        if x == 0 or x == np.size(self.occdata, 0) - 1 or y == 0 or y == np.size(self.occdata, 1):
            return False
        for dx, dy in NBORS:
            cx = dx + x
            cy = dy + y
            if self.valid_point(cx, cy) and self.floodfill_vis[cx][cy] == 0:
                self.floodfill_vis[cx][cy] = 1
                if self.occdata[cx][cy] < MAP_THRESHOLD:
                    if not self.floodfill_edges(cx, cy):
                        return False
        return True

    def generate_raycast(self, map_x, map_y, map_yaw):
        raycast = np.full_like(self.occdata, 0, dtype=np.int32)

        slope = 1 / math.tan(np.rad2deg(map_yaw))

        dx = 0
        while True:
            cx = map_x + dx
            bound_1 = round(slope * dx + map_y)
            bound_2 = round(slope * (dx + 1) + map_y)
            to_mark = range(min(bound_1, bound_2), max(bound_1, bound_2) + 1)
            if slope < 0:
                to_mark = reversed(to_mark)
            for cy in to_mark:
                if not self.valid_point(cx, cy) or self.occdata[cx][cy] >= MAP_THRESHOLD:
                    break
                raycast[cx][cy] = 1
            else:
                dx += 1
                continue
            break
        
        dx = -1
        while True:
            cx = map_x + dx
            bound_1 = round(slope * dx + map_y)
            bound_2 = round(slope * (dx + 1) + map_y)
            to_mark = range(min(bound_1, bound_2), max(bound_1, bound_2) + 1)
            if slope > 0:
                to_mark = reversed(to_mark)
            for cy in to_mark:
                if not self.valid_point(cx, cy) or self.occdata[cx][cy] >= MAP_THRESHOLD:
                    break
                raycast[cx][cy] = 1
            else:
                dx -= 1
                continue
            break
        
        return raycast

    def floodfill_infront(self, map_x, map_y):
        if map_x == 0 or map_x == np.size(self.occdata, 0) - 1 or map_y == 0 or map_y == np.size(self.occdata, 1):
            return False
        for dx, dy in NBORS:
            cx = dx + map_x
            cy = dy + map_y
            if self.valid_point(cx, cy) and self.floodfill_vis[cx][cy] == 0 and self.raycast[cx][cy] == 0:
                self.floodfill_vis[cx][cy] = 1
                if self.occdata[cx][cy] < MAP_THRESHOLD:
                    if not self.floodfill_infront(cx, cy):
                        return False
        return True

    def odom_callback(self, msg):
        self.odom = msg
        self.x, self.y, self.yaw = self.current_location()
        self.update_state_odom()

    def occ_callback(self, msg):
        msgdata = np.array(msg.data)
        oc2 = msgdata
        self.occdata = oc2.reshape(msg.info.height,msg.info.width)
        self.map_info = msg.info
        self.maps_seen += 1

        if self.maps_seen > 2 and not self.toured:
            if self.x is None:
                return
            map_x, map_y, map_yaw = self.to_map_coords(self.x, self.y, self.yaw)
            self.floodfill_vis = np.full_like(self.occdata, 0, dtype=np.int32)
            self.toured = self.floodfill_edges(map_x, map_y)

            if self.toured:
                print("Tour complete!")

            self.update_state_occ()

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
        if self.x is None:
            self.stopbot()
            return

        if new_state == State.SEEK:
            self.state_data = { "target_angle": (self.yaw + kwargs["target_angle"]) % 360, "precise": kwargs.get("precise", False) }
        elif new_state == State.TURN_RIGHT:
            self.state_data = { "start_angle": self.yaw }
        elif new_state == State.LOCKED_FORWARD:
            self.state_data = { "dist": kwargs.get("dist"), "start": (self.x, self.y) }
        elif new_state == State.LOADING:
            self.state_data = {
                "previous_state": self.state,
                "previous_state_data": self.state_data,
            }
        elif new_state == State.SEEK_TARGET:
            self.state_data = { "target_angle": (self.yaw + kwargs.get("angle") ) }
        else:
            self.state_data = {}
        self.state = new_state
        print("state", self.state, self.state_data)
        self.execute_state()

    def update_state_occ(self):
        if not self.toured and (self.state == State.FORWARD or self.state == State.TURN_LEFT or self.state == State.TURN_RIGHT):
            max_yaw = None
            for i in range(90):
                try_yaw = (self.yaw + i) % 360

                map_x, map_y, map_yaw = self.to_map_coords(self.x, self.y, try_yaw)
                self.floodfill_vis = np.full_like(self.occdata, 0, dtype=np.int32)
                
                yaw_rad = np.deg2rad(try_yaw)
                rot = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad)], [math.sin(yaw_rad), math.cos(yaw_rad)]])
                
                p = np.array([ROBOT_BACK, 0])
                trans = np.dot(rot, p)

                back_x = self.x + trans[0]
                back_y = self.y + trans[1]
                
                map_back_x, map_back_y, _ = self.to_map_coords(back_x, back_y, try_yaw)

                self.raycast = self.generate_raycast(map_back_x, map_back_y, try_yaw)
                
                start_point = [map_back_x, map_back_y]

                
                p = np.array([0.02, 0])
                trans = np.dot(rot, p)

                cx = self.x
                cy = self.y

                while start_point == (map_x, map_y) or self.raycast[start_point[0]][start_point[1]] == 1:
                    cx += trans[0]
                    cy += trans[1]
                    start_point[0], start_point[1], _ = self.to_map_coords(cx, cy, try_yaw)

                self.toured_infront = self.floodfill_infront(*start_point)
                
                if self.toured_infront:
                    for x in range(np.size(self.raycast, 0)):
                        for y in range(np.size(self.raycast, 1)):
                            if self.raycast[x][y] == 1:
                                self.floodfill_vis[x][y] += 2
                            
                    self.floodfill_vis[map_x][map_y] = 4
                    self.floodfill_vis[start_point[0]][start_point[1]] = 5
                    np.savetxt("toured_infront.txt", self.floodfill_vis)
                
                if self.toured_infront:
                    max_yaw = i
            if max_yaw:
                self.change_state(State.SEEK, target_angle = max_yaw + 85.0 , precise = True)
                print("Already explored, skipping")

    def update_state_nfc(self):
        if self.toured:
            if not self.seen_nfc:
                self.seen_nfc = True
                self.change_state(State.LOADING)
                self.stopbot()
        elif self.nfc_loc is None:
            print("Found NFC, but waiting for tour completion first.")
            if self.x is None:
                return
            self.nfc_loc = (self.x, self.y)

    def update_state_button(self):
        if self.state == State.LOADING:
            print("Button pressed, resuming.")
            self.state = self.state_data["previous_state"]
            self.state_data = self.state_data["previous_state_data"]
            self.loaded = BALLS_LOADED

    def update_state_scan(self):
        if self.state == State.BEGIN:
            if self.laser_range is not None:
                self.change_state(State.SEEK, target_angle = np.nanargmin(self.laser_range))
        elif self.state == State.FORWARD:
            if self.right() > LOOKRIGHT:
                self.change_state(State.TURN_RIGHT)
            elif self.front() < LOOKAHEAD:
                self.change_state(State.TURN_LEFT)
        elif self.state == State.TURN_LEFT:
            if not self.front() < LOOKAHEAD:
                self.change_state(State.FORWARD)
        #elif self.state == State.TURN_RIGHT:
        #    if not self.right() > LOOKRIGHT:
        #        self.change_state(State.FORWARD)
        elif self.state == State.LOCKED_FORWARD:
            if self.front() < LOOKAHEAD:
                self.change_state(State.TURN_LEFT)
        elif self.state == State.MOVE_TARGET:
            pass
            # todo: rework
            #if self.front_left() <= PADDING * 6:
            #    self.change_state(State.FIRING)
        if RANDOMIZE and self.toured and self.seen_nfc:
            if random.randint(0, 50)  == 0:
                self.change_state(State.SEEK, target_angle=90.0)

    def update_state_odom(self):
        if self.x is None:
            self.stopbot()
            return
        if self.state == State.SEEK:
            _, achieved = turn_towards(self.yaw, self.state_data["target_angle"], self.state_data["precise"])
            if achieved:
                self.change_state(State.LOCKED_FORWARD)
        elif self.state == State.SEEK_TARGET:
            _, achieved = turn_towards(self.yaw, self.state_data["target_angle"], True)
            if achieved:
                self.change_state(State.MOVE_TARGET)
        elif self.state == State.TURN_RIGHT:
            if min((self.yaw - self.state_data["start_angle"] + 360) % 360, 360 - (self.yaw - self.state_data["start_angle"] + 360) % 360) >= 90:
                self.change_state(State.LOCKED_FORWARD, dist=ROBOT_LEN)
        elif self.state == State.LOCKED_FORWARD:
            if self.state_data["dist"] is not None and dist_to(*self.state_data["start"], self.x, self.y) >= self.state_data["dist"]:
                self.change_state(State.FORWARD)

        if self.toured and self.nfc_loc is not None and not self.seen_nfc and dist_to(self.x, self.y, self.nfc_loc[0], self.nfc_loc[1]) < RETRACE_MARGIN:
            self.seen_nfc = True
            self.change_state(State.LOADING)
            self.stopbot()
            

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

        if len(self.thermal_surroundings) >= 5 and self.loaded != 0 and self.state != State.SEEK_TARGET:
            if self.toured:
                self.change_state(State.SEEK_TARGET, angle = np.rad2deg(math.atan2(ay, ax)))
            else:
                print("Found target, but waiting for tour completion first")

            
    def execute_state(self):
        if self.x is None:
            self.stopbot()
            return
        twist = Twist()
        twist.angular.z = 0.0
        twist.linear.x = 0.0

        if self.state == State.SEEK:
            twist, _ = turn_towards(self.yaw, self.state_data["target_angle"], self.state_data["precise"])
        elif self.state == State.FORWARD:
            twist.linear.x = FORWARD_VEL
            twist.angular.z = -ADJUST_TURNING_VEL
            if self.front_right_half() < ADJUSTLOOKAHEAD:
                twist.angular.z = ADJUST_TURNING_VEL
        elif self.state == State.TURN_LEFT:
            twist.angular.z = TURNING_VEL
        elif self.state == State.TURN_RIGHT:
            twist.angular.z = -TURNING_VEL
        elif self.state == State.LOCKED_FORWARD:
            twist.linear.x = taper_move(self.laser_range[0])
        elif self.state == State.SEEK_TARGET:
            twist, _ = turn_towards(self.yaw, self.state_data["target_angle"], True)
        elif self.state == State.MOVE_TARGET:
            twist.linear.x = taper_move(self.laser_range[0])
        elif self.state == State.FIRING:
            self.stopbot()
            time.sleep(1)
            self.motor_publisher_.publish(Int32(data=1))
            time.sleep(3)
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
