from re import A
from tkinter import E
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
import math
import cmath
import time
import tf2_ros
from scipy.spatial.transform import Rotation
from collections import deque

TURNING_RADIUS = 0.12
MOVEMENT_TOL = 0.01
ANGLE_TOL = 5

OCCMAP_THRESHOLD = 50.0
UNSURE_THRESHOLD = 5.0

RAYCAST_ANGLES = 360

DEBUG = True

NBORS = [(-1, 0), (1, 0), (0, 1), (0, -1)]

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


class AutoNav(Node):
    def __init__(self):
        super().__init__('auto_nav')
        self.publisher_ = self.create_publisher(Twist,'cmd_vel',10)

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

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.x = 0
        self.y = 0

        self.margin_circle = []
        self.path = []
        self.cur_goal = None

    def current_location(self):
        try:
            base_link = self.tfBuffer.lookup_transform('map', 'base_link', rclpy.time.Time(), rclpy.duration.Duration(seconds=1))
        except Exception as e:
            print("failed to get location")
            if e is KeyboardInterrupt:
                raise e
            return None, None, None

        _roll, _pitch, yaw = euler_from_quaternion(base_link.transform.rotation.x, base_link.transform.rotation.y, base_link.transform.rotation.z, base_link.transform.rotation.w)
        return base_link.transform.translation.x, base_link.transform.translation.y, yaw

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

    def scannable_unknowns(self, start_x, start_y):
        bfs = np.full_like(self.occdata, -1, dtype=np.int32)
        scannable = []
        q = deque()
        q.append((start_x, start_y))
        bfs[start_x][start_y] = 0

        while q:
            px, py = q.popleft()
            for dx, dy in NBORS:
                cx = dx + px
                cy = dy + py
                if self.valid_point(cx, cy) and bfs[cx][cy] == -1:
                    bfs[cx][cy] = bfs[px][py] + 1
                    if self.occdata[cx][cy] < OCCMAP_THRESHOLD:
                        if self.occdata[cx][cy] != -1:
                            q.append((cx, cy))
                        else:
                            scannable.append((cx, cy))
        return scannable
    
    def border(self, start_x, start_y):
        bfs = np.full_like(self.occdata, -1, dtype=np.int32)
        border = []
        q = deque()
        q.append((start_x, start_y))
        bfs[start_x][start_y] = 0

        while q:
            px, py = q.popleft()
            for dx, dy in NBORS:
                cx = dx + px
                cy = dy + py
                if self.valid_point(cx, cy) and bfs[cx][cy] == -1:
                    bfs[cx][cy] = bfs[px][py] + 1
                    if self.occdata[cx][cy] < UNSURE_THRESHOLD and self.occdata[cx][cy] != -1:
                        q.append((cx, cy))
                    else:
                        border.append((cx, cy))
        return border, bfs

    def raycast(self, x, y):
        raycast = np.full_like(self.occdata, -1, dtype=np.int32)
        for ang in np.arange(0.0, 360.0, 360.0 / RAYCAST_ANGLES):
            dir = -1 if (ang > 90.0 and ang <= 270.0) else 1
            dx = 1
            if ang == 90.0 or ang == 270:
                while True:
                    cx = x
                    cy = y + dx * dir
                    if (not self.valid_point(cx, cy)) or self.occdata[cx][cy] > OCCMAP_THRESHOLD:
                        break
                    raycast[cx][cy] = 1
                    dx += 1
                continue

            slope = math.tan(np.deg2rad(ang))
            while True:
                cx = x + dx * dir
                y_bound_min = int(math.floor(slope * (float(dx) - 0.5)))
                y_bound_max = int(math.ceil(slope * (float(dx) + 0.5)))
                for dy in range(min(y_bound_min, y_bound_max), max(y_bound_min, y_bound_max) + 1):
                    cy = y + dy * dir
                    if (not self.valid_point(cx, cy)) or self.occdata[cx][cy] > OCCMAP_THRESHOLD:
                        break
                    raycast[cx][cy] = 1
                else:
                    dx += 1
                    continue
                break
        return raycast

    def valid_point(self, x, y):
        return x >= 0 and x < np.size(self.occdata, 0) and y >= 0 and y < np.size(self.occdata, 1)

    def turning_margin(self):
        return TURNING_RADIUS + self.map_info.resolution

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation_quat = msg.pose.pose.orientation
        self.x, self.y = position.x, position.y
        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w)

    def occ_callback(self, msg):
        msgdata = np.array(msg.data)
        oc2 = msgdata
        self.occdata = oc2.reshape(msg.info.height,msg.info.width)
        self.map_info = msg.info

        raw_x, raw_y, _ = self.current_location()
        if raw_x is None:
            return
        map_x, map_y = self.to_map_coords(raw_x, raw_y)
        print(map_x, map_y)
        scannable = self.scannable_unknowns(map_x, map_y)
        border, dist = self.border(map_x, map_y)

        scan_x, scan_y = scannable[0]
        print(scan_x, scan_y)
        rays = self.raycast(scan_x, scan_y)
        
        if DEBUG:
            scan_visualize = np.full_like(self.occdata, 0, dtype=np.int32)
            for s_x, s_y in scannable:
                scan_visualize[s_x][s_y] = 1
            border_visualize = np.full_like(self.occdata, 0, dtype=np.int32)
            for s_x, s_y in border:
                border_visualize[s_x][s_y] = 1
            np.savetxt("occ.txt", self.occdata)
            np.savetxt("scan.txt", scan_visualize)
            np.savetxt("border.txt", border_visualize)
            np.savetxt("dist.txt", dist)
            np.savetxt("rays.txt", rays)
            print("debug saved")

    def scan_callback(self, msg):
        self.laser_range = np.array(msg.ranges)
        self.laser_range[self.laser_range==0] = np.nan

    def stopbot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)

    def mover(self):
        try:
            while rclpy.ok():
                rclpy.spin_once(self)
        except Exception as e:
            print(e)
        finally:
            self.stopbot()


def main(args=None):
    rclpy.init(args=args)

    auto_nav = AutoNav()
    auto_nav.mover()

    auto_nav.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
