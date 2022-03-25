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
        self.map_info = None
        self.occdata = np.array([])
        self.laser_range = np.array([])
        self.traversable_map = np.array([])
        self.raycast = np.array([])
        self.bfs = np.array([])
        self.reachable_bfs = np.array([])
        self.reachable_par = np.array([])

        self.margin_circle = []
        self.path = []
        self.cur_goal = None

    def turning_margin(self):
        return TURNING_RADIUS + self.map_info.resolution

    def odom_callback(self, msg):
        orientation_quat = msg.pose.pose.orientation
        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w)

    def occ_callback(self, msg):
        if not self.path:
            msgdata = np.array(msg.data)
            oc2 = msgdata
            self.occdata = np.int8(oc2.reshape(msg.info.height,msg.info.width))
            self.map_info = msg.info
            print("regenerating map")
            self.regenerate_map()
            np.savetxt("trav.txt", self.traversable_map)

            map_position = self.map_info.origin.position
            map_orientation = self.map_info.origin.orientation

            map_rotation = Rotation.from_quat([map_orientation.x, map_orientation.y, map_orientation.z, map_orientation.w])

            try:
                base_link = self.tfBuffer.lookup_transform('map', 'base_link', rclpy.time.Time(), rclpy.duration.Duration(seconds=1))
            except Exception as e:
                if e is KeyboardInterrupt:
                    raise e
                return

            # Reversed for some reason
            offset_y = base_link.transform.translation.x - map_position.x
            offset_x = base_link.transform.translation.y - map_position.y

            rotated_offset = map_rotation.inv().apply(np.array([offset_x, offset_y, 0]))
            map_x = int(rotated_offset[0] // self.map_info.resolution)
            map_y = int(rotated_offset[1] // self.map_info.resolution)

            print(map_x, map_y)

            print(offset_x, offset_y, rotated_offset, map_x, map_y)

            # TODO: handle no gaps found (either map complete or we screwed up)
            print("finding closest gap")
            gap_x, gap_y = self.run_bfs(map_x, map_y)
            print("calculating pathing")
            self.run_reachable_bfs(map_x, map_y)
            print("finding locations to view gap from")
            vis_x, vis_y = self.raycast_circle(gap_x, gap_y, -2)
            print(vis_x, vis_y)
            print(self.reachable_bfs[vis_x][vis_y])

            goal_x, goal_y = vis_x, vis_y
            path = np.full_like(self.occdata, 0, np.int32)
            while not (goal_x == map_x and goal_y == map_y):
                self.path.append((goal_x, goal_y))
                goal_x, goal_y = self.reachable_par[goal_x][goal_y]
                path[goal_x][goal_y] = 1

            np.savetxt("occ.txt", self.occdata)
            np.savetxt("ray.txt", self.raycast)

            np.savetxt("bfs.txt", self.bfs)
            np.savetxt("rbfs.txt", self.reachable_bfs)
            np.savetxt("path.txt", path)
            print("Regenerated pathing!")

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
                    self.traversable_map[x][y] = -2
                    self.fill_map(x, y, -1)
                elif self.occdata[x][y] == -1 or self.occdata[x][y] >= UNSURE_THRESHOLD:
                    self.fill_map(x, y, 0)

    def raycast_circle(self, x, y, avoid):
        closest = None
        ret = None
        self.raycast = np.full_like(self.occdata, -1, dtype=np.int32)
        for ang in np.arange(0.0, 360.0, 360.0 / RAYCAST_ANGLES):
            dir = -1 if (ang > 90.0 and ang <= 270.0) else 1
            dx = 1

            if ang == 90.0 or ang == 270:
                while True:
                    cx = x
                    cy = y + dx * dir
                    if (not self.valid_point(cx, cy)) or self.traversable_map[cx][cy] <= avoid:
                        break
                    self.raycast[cx][cy] = 1
                    if self.reachable_bfs[cx][cy] != -1 and (closest is None or closest > self.reachable_bfs[cx][cy]):
                        closest = self.reachable_bfs[cx][cy]
                        ret = (cx, cy)
                    dx += 1
                continue

            slope = math.tan(np.deg2rad(ang))
            while True:
                cx = x + dx * dir
                y_bound_min = int(math.floor(slope * (float(dx) - 0.5)))
                y_bound_max = int(math.ceil(slope * (float(dx) + 0.5)))
                for dy in range(min(y_bound_min, y_bound_max), max(y_bound_min, y_bound_max) + 1):
                    cy = y + dy * dir
                    if (not self.valid_point(cx, cy)) or self.traversable_map[cx][cy] <= avoid:
                        break
                    self.raycast[cx][cy] = 1
                    if self.reachable_bfs[cx][cy] != -1 and (closest is None or closest > self.reachable_bfs[cx][cy]):
                        closest = self.reachable_bfs[cx][cy]
                        ret = (cx, cy)
                else:
                    dx += 1
                    continue
                break
        return ret
    
    def run_bfs(self, x, y):
        self.bfs = np.full_like(self.occdata, -1, dtype=np.int32)
        q = deque()
        q.append((x, y))
        self.bfs[x][y] = 0

        ret = None
        while q:
            px, py = q.popleft()
            for dx, dy in NBORS:
                cx = dx + px
                cy = dy + py
                if self.valid_point(cx, cy):
                    if self.traversable_map[cx][cy] > -2 and self.bfs[cx][cy] == -1:
                        self.bfs[cx][cy] = self.bfs[px][py] + 1
                        q.append((cx, cy))
                    if self.occdata[cx][cy] == -1:
                        if ret is None:
                            ret = (cx, cy)
        return ret

    def run_reachable_bfs(self, x, y):
        self.reachable_bfs = np.full_like(self.occdata, -1, dtype=np.int32)
        self.reachable_par = np.full_like(self.occdata, -1, dtype=(np.int32, 2))

        q = deque()
        q.append((x, y))
        self.reachable_bfs[x][y] = 0

        while q:
            px, py = q.popleft()
            for dx, dy in NBORS:
                cx = dx + px
                cy = dy + py
                if self.valid_point(cx, cy):
                    if self.traversable_map[cx][cy] > -1 and self.reachable_bfs[cx][cy] == -1:
                        self.reachable_bfs[cx][cy] = self.reachable_bfs[px][py] + 1
                        self.reachable_par[cx][cy] = [px, py]
                        q.append((cx, cy))


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

    def move_to_path(self):
        base_link = None
        odom_link = None
        try:
            base_link = self.tfBuffer.lookup_transform('map', 'base_link', rclpy.time.Time(), rclpy.duration.Duration(seconds=1))
            odom_link = self.tfBuffer.lookup_transform('map', 'odom_link', rclpy.time.Time(), rclpy.duration.Duration(seconds=1))

        except Exception as e:
            if e is KeyboardInterrupt:
                raise e
            return

        map_position = self.map_info.origin.position
        map_orientation = self.map_info.origin.orientation
        map_res = self.map_info.resolution

        map_rotation = Rotation.from_quat([map_orientation.x, map_orientation.y, map_orientation.z, map_orientation.w])

        # Reversed for some reason
        rot_x = self.path[-1][1] * map_res
        rot_y = self.path[-1][0] * map_res

        dest_offset_x, dest_offset_y, _ = map_rotation.apply(np.array([rot_x, rot_y, 0]))
        dest_x = dest_offset_x + map_position.x
        dest_y = dest_offset_y + map_position.y

        dist = dist_to(base_link.transform.translation.x, base_link.transform.translation.y, dest_x, dest_y)
        desired_angle = angle_to(base_link.transform.translation.x, base_link.transform.translation.y, dest_x, dest_y)
        _roll, _pitch, yaw = euler_from_quaternion(odom_link.transform.rotation.x, odom_link.transform.rotation.y, odom_link.transform.rotation.z, odom_link.transform.rotation.w)

        yaw += self.yaw
        
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0


        print(dist)
        if dist <= MOVEMENT_TOL:
           self.path.pop()
        else:
            print(yaw, desired_angle)
            if abs(yaw - desired_angle) <= np.deg2rad(ANGLE_TOL):
                twist.linear.x = 0.22
            else:
                twist.angular.z = 1.0
        self.publisher_.publish(twist)

        time.sleep(0.25)


    def mover(self):
        try:
            while rclpy.ok():
                if self.path:
                    self.move_to_path()
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
