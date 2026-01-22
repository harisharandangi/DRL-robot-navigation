import math
import os
import random
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

# --- CONSTANTS ---
GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1

# LDS-01/02 Settings (2D Lidar)
ENVIRONMENT_DIM = 20
MAX_TRAINING_RANGE = 10.0
REAL_SENSOR_MAX = 3.5

class GazeboEnv:
    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        # --- STAGE 1 SETTINGS: VIRTUAL CAGE ---
        # We start with a 2.0m "Invisible Box" in the empty world.
        # This prevents the robot from driving to infinity and forces it to turn.
        self.boundary_limit = 2.0 
        self.success_count = 0  # To track curriculum progress

        self.upper = self.boundary_limit
        self.lower = -self.boundary_limit
        
        self.velodyne_data = np.ones(self.environment_dim) * MAX_TRAINING_RANGE
        self.last_odom = None
        
        # Reward Shaping Variables
        self.last_distance = 0.0 
        self.idle_counter = 0

        # ROS Initialization
        self.set_self_state = ModelState()
        self.set_self_state.model_name = "turtlebot3_burger"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

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

        self.vel_pub = rospy.Publisher("/turtlebot3_burger/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        
        self.velodyne = rospy.Subscriber("/scan", LaserScan, self.velodyne_callback, queue_size=1)
        self.odom = rospy.Subscriber("/turtlebot3_burger/odom", Odometry, self.odom_callback, queue_size=1)

    def check_pos(self, x, y):
        goal_ok = True
        # Dynamic check using current boundary_limit
        if x > self.boundary_limit or x < -self.boundary_limit or \
           y > self.boundary_limit or y < -self.boundary_limit:
            goal_ok = False
        return goal_ok

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

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def step(self, action):
        target = False

        # 1. Execute Action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0] * 0.5 
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        # 2. Physics Step
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        if self.last_odom is None:
            # Return dummy state size 26 (20 laser + 6 robot state)
            return [0] * (self.environment_dim + 6), 0, False, False

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # 3. Collision / Boundary Check
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        
        # Kill if outside the "Invisible Cage"
        if self.odom_x > self.boundary_limit or self.odom_x < -self.boundary_limit or \
           self.odom_y > self.boundary_limit or self.odom_y < -self.boundary_limit:
            done = True
            collision = True 
            print(f"OUT OF BOUNDS ({self.boundary_limit}m)! Resetting...")

        # 4. State Construction
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

        # Idle Check (Prevent Spinning)
        if action[0] < 0.05:
            self.idle_counter += 1
        else:
            self.idle_counter = 0
            
        if self.idle_counter > 20: 
            print("LAZY ROBOT! Killing episode...")
            done = True
            collision = True 

        # Angle Calculation
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

        # 5. Goal Success & Curriculum Learning
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True
            
            # AUTOMATIC CURRICULUM: Expand world after 3 wins
            self.success_count += 1
            if self.success_count >= 3:
                self.success_count = 0
                self.boundary_limit += 0.5
                print(f"STAGE 1 LEVEL UP! New Boundary: {self.boundary_limit} meters")

        # 6. AUGMENTED STATE: Add current velocity
        current_vel_x = self.last_odom.twist.twist.linear.x
        current_vel_w = self.last_odom.twist.twist.angular.z
        
        # State = [Laser(20)] + [Dist, Angle, ActionLinear, ActionAngular, VelLinear, VelAngular]
        robot_state = [distance, theta, action[0], action[1], current_vel_x, current_vel_w]
        state = np.append(laser_state, robot_state)
        
        # 7. Dense Reward
        reward = self.get_reward(target, collision, action, min_laser, distance, self.last_distance)
        self.last_distance = distance 
        
        return state, reward, done, target

    def reset(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            # Spawn random within CURRENT boundary
            x = np.random.uniform(-self.boundary_limit + 0.5, self.boundary_limit - 0.5)
            y = np.random.uniform(-self.boundary_limit + 0.5, self.boundary_limit - 0.5)
            position_ok = self.check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        self.change_goal()
        self.random_box()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        self.last_distance = distance 

        # Recalculate Angle for Reset State
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

        # Initial State (Zeros for actions and velocity)
        robot_state = [distance, theta, 0.0, 0.0, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):
        # Goal must also be inside dynamic boundary
        self.upper = self.boundary_limit
        self.lower = -self.boundary_limit

        goal_ok = False
        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = self.check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        pass

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

    @staticmethod
    def get_reward(target, collision, action, min_laser, distance, last_distance):
        if target:
            return 200.0
        elif collision:
            return -100.0
        else:
            # Hot/Cold Dense Reward
            distance_rate = last_distance - distance
            reward = distance_rate * 400.0 
            reward -= 0.05
            return reward
