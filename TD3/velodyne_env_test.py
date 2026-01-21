import math
import os
import random
import time
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

# Configuration
GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1
MAX_TRAINING_RANGE = 10.0
REAL_SENSOR_MAX = 3.5

class GazeboEnv:
    def __init__(self, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.velodyne_data = np.ones(self.environment_dim) * MAX_TRAINING_RANGE
        self.last_odom = None

        # Initialize ROS Node (if not already done)
        try:
            rospy.init_node("gym_test", anonymous=True)
        except rospy.exceptions.ROSException:
            pass # Node already initialized

        # Publishers & Subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        
        self.velodyne = rospy.Subscriber("/scan", LaserScan, self.velodyne_callback, queue_size=1)
        self.odom = rospy.Subscriber("/r1/odom", Odometry, self.odom_callback, queue_size=1)

        # Service Proxies (for pausing simulation to keep step times consistent)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)

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

    # --- NEW FUNCTION: Set Goal without moving robot ---
    def set_goal(self, x, y):
        self.goal_x = float(x)
        self.goal_y = float(y)
        self.publish_markers([0, 0])
        print(f"Goal Updated to: ({self.goal_x}, {self.goal_y})")

    def step(self, action):
        target = False

        # Publish Action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        # Unpause physics to let robot move
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        # Pause physics to compute state
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # Process Sensor Data
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        
        # Get Position
        if self.last_odom is not None:
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
        else:
            # Fallback if odom not ready
            return [0] * (self.environment_dim + 4), 0, False, False

        # Calculate Distance/Angle to Goal
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        
        if mag1 * mag2 == 0:
            beta = 0
        else:
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

        # Construct State
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]
        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        
        reward = self.get_reward(target, collision, action, min_laser)
        return state, reward, done, target

    # --- MODIFIED RESET: Does NOT teleport robot ---
    def reset(self):
        # We assume the robot is already in a valid position from the previous run.
        # We just need to return the current state relative to the NEW goal.
        
        # Ensure we have data
        while self.last_odom is None:
            time.sleep(0.1)
            
        # Trigger a 0-action step to refresh calculations without moving
        state, _, _, _ = self.step([0, 0])
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

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
