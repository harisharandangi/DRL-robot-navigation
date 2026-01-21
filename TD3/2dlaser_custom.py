import math
import os
import subprocess
import time
from os import path
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# --- SETTINGS ---
GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1

# LDS-02 Settings
ENVIRONMENT_DIM = 20
MAX_TRAINING_RANGE = 10.0
REAL_SENSOR_MAX = 3.5

class GazeboEnv:
    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.velodyne_data = np.ones(self.environment_dim) * MAX_TRAINING_RANGE
        self.last_odom = None
        self.data_ready = False

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")

        rospy.init_node("gym", anonymous=True)
        
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = "/home/harry/workspaces/td_ws/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch/" + launchfile

        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # --- CORRECTED TOPIC NAMES HERE ---
        self.vel_pub = rospy.Publisher("/turtlebot3_burger/cmd_vel", Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        
        self.velodyne = rospy.Subscriber("/scan", LaserScan, self.velodyne_callback, queue_size=1)
        self.odom = rospy.Subscriber("/turtlebot3_burger/odom", Odometry, self.odom_callback, queue_size=1)

    def velodyne_callback(self, msg):
        raw_ranges = np.array(msg.ranges)
        raw_ranges[np.isinf(raw_ranges)] = REAL_SENSOR_MAX
        raw_ranges[np.isnan(raw_ranges)] = REAL_SENSOR_MAX
        new_data = []
        chunk_size = len(raw_ranges) // self.environment_dim
        for i in range(self.environment_dim):
            segment = raw_ranges[i * chunk_size : (i + 1) * chunk_size]
            min_val = np.min(segment)
            if min_val >= REAL_SENSOR_MAX - 0.1:
                final_val = MAX_TRAINING_RANGE
            else:
                final_val = min_val
            new_data.append(final_val)
        self.velodyne_data = np.array(new_data)
        self.data_ready = True

    def odom_callback(self, od_data):
        self.last_odom = od_data

    # --- NEW FUNCTION: USER SETS GOAL ---
    def set_custom_goal(self, x, y):
        self.goal_x = x
        self.goal_y = y
        print(f"New Goal Set: X={x}, Y={y}")
        self.publish_markers([0,0]) # Update visual marker in Rviz

    def step(self, action):
        target = False

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0] * 0.5 
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        if self.last_odom is None:
            return [0] * (self.environment_dim + 4), 0, False, False

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = 0 # Reward doesn't matter for testing
        return state, reward, done, target

    def reset(self):
        # Stop the robot initially
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # --- WAIT FOR SENSOR DATA ---
        print("Waiting for Laser & Odom data...")
        while not self.data_ready or self.last_odom is None:
            time.sleep(0.1)
        print("Data received! Starting episode.")
        # -----------------------------

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        try:
            beta = math.acos(dot / (mag1 * mag2))
        except:
            beta = 0

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def publish_markers(self, action):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

    @staticmethod
    def observe_collision(laser_data):
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser
