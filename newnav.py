# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from re import A
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

TURNING_RADIUS = 0.12

OCCMAP_THRESHOLD = 50.0

RAYCAST_ANGLES = 360


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
        self.map_info = None
        self.occdata = np.array([])
        self.laser_range = np.array([])
        self.traversable_map = np.array([])
        self.raycast = np.array([])
        self.margin_circle = []

    def turning_margin(self):
        return TURNING_RADIUS + self.map_info.resolution

    def odom_callback(self, msg):
        orientation_quat = msg.pose.pose.orientation
        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w)

    def occ_callback(self, msg):
        msgdata = np.array(msg.data)
        oc2 = msgdata + 1
        self.occdata = np.uint8(oc2.reshape(msg.info.height,msg.info.width))
        self.map_info = msg.info
        self.regenerate_map()

        map_position = self.map_info.origin.position
        map_orientation = self.map_info.origin.orientation

        map_rotation = Rotation.from_quat([map_orientation.x, map_orientation.y, map_orientation.z, map_orientation.w])

        base_link = self.tfBuffer.lookup_transform('map', 'base_link', rclpy.time.Time())

        offset_x = base_link.transform.translation.x - map_position.x
        offset_y = base_link.transform.translation.y - map_position.y

        rotated_offset = map_rotation.inv().apply(np.array([offset_x, offset_y, 0]))
        map_x = int(rotated_offset[1] // self.map_info.resolution)
        map_y = int(rotated_offset[0] // self.map_info.resolution)
        self.traversable_map[map_x][map_y] = 2
        np.savetxt("occ.txt", self.occdata)
        np.savetxt("trav.txt", self.traversable_map)

        print("start ray cast")
        self.raycast_circle(map_x, map_y, 0)
        np.savetxt("ray.txt", self.raycast)
        print("ray cast")

    def scan_callback(self, msg):
        self.laser_range = np.array(msg.ranges)
        self.laser_range[self.laser_range==0] = np.nan

    def stopbot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)

    def regenerate_map(self):
        self.regenerate_margin_circle()
        self.traversable_map = np.full_like(self.occdata, 1, dtype=np.int32)
        for x in range(np.size(self.occdata, 0)):
            for y in range(np.size(self.occdata, 1)):
                if self.occdata[x][y] >= OCCMAP_THRESHOLD:
                    self.traversable_map[x][y] = -1
                    self.fill_map(x, y, 0)

    def raycast_circle(self, x, y, avoid):
        self.raycast = np.full_like(self.occdata, -1, dtype=np.int32)
        for ang in np.arange(0.0, 360.0, 360.0 / RAYCAST_ANGLES):
            if ang == 90.0:
                ang = 90.001

            if ang == 270.0:
                ang = 270.001

            slope = math.tan(np.deg2rad(ang))
            dir = -1 if (ang > 90.0 and ang < 270.0) else 1
            dx = 1
            while True:
                cx = x + dx * dir
                y_bound_min = int(math.floor(slope * (float(dx) - 0.5)))
                y_bound_max = int(math.ceil(slope * (float(dx) + 0.5)))
                for dy in range(min(y_bound_min, y_bound_max), max(y_bound_min, y_bound_max) + 1):
                    cy = y + dy * dir
                    if (not self.valid_point(cx, cy)) or self.traversable_map[cx][cy] <= avoid:
                        break
                    self.raycast[cx][cy] = 1
                else:
                    dx += 1
                    continue
                break

    def regenerate_margin_circle(self):
        self.margin_circle.clear()
        margin = self.turning_margin()
        x_range = math.ceil(margin / self.map_info.resolution)
        for dx in range (0, x_range + 1):
            y_min = math.ceil(math.sqrt((x_range ** 2) - (dx ** 2)))
            y_max = max(y_min, math.ceil(math.sqrt((x_range ** 2) - ((dx - 1) ** 2))) - 1)
            for dy in range(y_min, y_max + 1):
                self.margin_circle.append((dx, -dy))
                self.margin_circle.append((dx, dy))
                self.margin_circle.append((-dx, -dy))
                self.margin_circle.append((-dx, dy))

    def valid_point(self, x, y):
        return x >= 0 and x < np.size(self.occdata, 0) and y >= 0 and y < np.size(self.occdata, 1)

    def fill_map(self, x, y, val):
        for dx, dy in self.margin_circle:
            cx = x + dx
            cy = y + dy
            if self.valid_point(cx, cy):
                self.traversable_map[cx][cy] = min(self.traversable_map[cx][cy], val)

    def mover(self):
        try:
            while rclpy.ok():
                if self.map_info is not None:
                    pass
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
