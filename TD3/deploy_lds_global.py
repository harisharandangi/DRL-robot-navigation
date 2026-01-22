import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rospy
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from squaternion import Quaternion
from std_srvs.srv import Empty

# --- SETTINGS ---
# Must match your training settings exactly!
ENVIRONMENT_DIM = 20
ROBOT_DIM = 6         # Augmented State (Laser + Dist + Angle + Actions + Velocity)
MAX_ACTION = 1
REAL_SENSOR_MAX = 3.5
MAX_TRAINING_RANGE = 10.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deployment Settings
LOCAL_PLANNER_DIST = 1.5  # How far ahead to place the local waypoint
GOAL_TOLERANCE = 0.35     # Distance to consider "Arrived"

# --- NEURAL NETWORK ARCHITECTURE (Must match Training) ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

class TD3(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)

    def load(self, filename, directory):
        # Load the Actor weights only (Critic is not needed for deployment)
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename), map_location=DEVICE)
        )

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(DEVICE)
        return self.actor(state).cpu().data.numpy().flatten()


# --- THE DEPLOYMENT MANAGER ---
class GlobalPlannerNode:
    def __init__(self):
        rospy.init_node("gdae_deployment", anonymous=True)

        # 1. Load the Brain
        self.state_dim = ENVIRONMENT_DIM + ROBOT_DIM
        self.action_dim = 2
        self.network = TD3(self.state_dim, self.action_dim)
        
        # TRY TO LOAD STAGE 2/3 MODEL, FALLBACK TO STAGE 1
        model_dir = "./pytorch_models"
        try:
            self.network.load("TD3_lds_cluttered", model_dir)
            print(">>> LOADED: Stage 2/3 (Cluttered) Model")
        except:
            print("!!! WARNING: Cluttered model not found. Loading Stage 1 (Empty) Model.")
            try:
                self.network.load("TD3_lds_empty", model_dir)
            except:
                print("!!! ERROR: No models found. Train your robot first!")
                exit()

        # 2. Setup ROS
        self.vel_pub = rospy.Publisher("/turtlebot3_burger/cmd_vel", Twist, queue_size=1)
        self.marker_pub = rospy.Publisher("/visual_planner", MarkerArray, queue_size=1)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/turtlebot3_burger/odom", Odometry, self.odom_callback, queue_size=1)

        # 3. Variables
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.angle = 0.0
        self.current_vel_x = 0.0
        self.current_vel_w = 0.0
        self.scan_data = np.ones(ENVIRONMENT_DIM) * MAX_TRAINING_RANGE
        self.raw_scan = None # Keep raw scan for "Gap Finding" logic
        
        # Action smoothing
        self.last_action = [0.0, 0.0]

        print(">>> Global Planner Ready. Waiting for Odom/Scan...")
        time.sleep(1) # Wait for callbacks

    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        
        # Quaternion to Euler
        q = Quaternion(
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        )
        e = q.to_euler(degrees=False)
        self.angle = round(e[2], 4)

        # Capture Velocity (Proprioception)
        self.current_vel_x = msg.twist.twist.linear.x
        self.current_vel_w = msg.twist.twist.angular.z

    def scan_callback(self, msg):
        self.raw_scan = np.array(msg.ranges) # Store full 360 scan for gap finding
        
        # Downsample to 20 rays for Neural Network
        raw = np.array(msg.ranges)
        raw[np.isinf(raw)] = REAL_SENSOR_MAX
        raw[np.isnan(raw)] = REAL_SENSOR_MAX
        
        new_data = []
        chunk = len(raw) // ENVIRONMENT_DIM
        for i in range(ENVIRONMENT_DIM):
            segment = raw[i * chunk : (i + 1) * chunk]
            new_data.append(min(np.min(segment), MAX_TRAINING_RANGE))
        
        self.scan_data = np.array(new_data)

    # --- THE GLOBAL PLANNER LOGIC ---
    # This implements the "Heuristic Manager" from the paper
    def get_local_waypoint(self, global_x, global_y):
        # 1. Vector to Global Goal
        dx = global_x - self.odom_x
        dy = global_y - self.odom_y
        dist_to_global = math.hypot(dx, dy)
        global_angle = math.atan2(dy, dx)

        # 2. If close, just go there
        if dist_to_global < LOCAL_PLANNER_DIST:
            return global_x, global_y

        # 3. GAP FINDING / OBSTACLE CHECK
        # We want to place a waypoint 1.5m ahead. Is that path blocked?
        # Simple check: Look at the laser ray corresponding to the global angle
        
        # Convert global angle to local robot angle
        local_angle = global_angle - self.angle
        # Normalize to -pi to pi
        while local_angle > np.pi: local_angle -= 2*np.pi
        while local_angle < -np.pi: local_angle += 2*np.pi

        # If the path ahead is clear, place waypoint 1.5m ahead
        local_waypoint_x = self.odom_x + LOCAL_PLANNER_DIST * math.cos(global_angle)
        local_waypoint_y = self.odom_y + LOCAL_PLANNER_DIST * math.sin(global_angle)

        # (Optional: You could add logic here to rotate the waypoint if laser < 1.0m)
        # For now, we trust the DRL agent to dodge, and the Global Planner just 
        # feeds the "General Direction".
        
        return local_waypoint_x, local_waypoint_y

    def navigate_to(self, final_x, final_y):
        print(f"--- NAVIGATING TO GLOBAL GOAL: ({final_x}, {final_y}) ---")
        
        arrived = False
        
        while not arrived and not rospy.is_shutdown():
            # 1. Update Global Status
            dist_to_goal = math.hypot(final_x - self.odom_x, final_y - self.odom_y)
            
            if dist_to_goal < GOAL_TOLERANCE:
                print(">>> GLOBAL GOAL REACHED! <<<")
                self.stop_robot()
                arrived = True
                break

            # 2. Generate Local Waypoint (The "Manager")
            wx, wy = self.get_local_waypoint(final_x, final_y)
            self.visualize_goals(final_x, final_y, wx, wy)

            # 3. Construct State for DRL (The "Pilot")
            # Calculate distance/angle to LOCAL waypoint, not global
            rel_dx = wx - self.odom_x
            rel_dy = wy - self.odom_y
            local_dist = math.hypot(rel_dx, rel_dy)
            
            # Angle calculation
            dot = rel_dx * 1 + rel_dy * 0
            mag1 = local_dist
            mag2 = 1.0
            beta = math.acos(dot / (mag1 * mag2))
            if rel_dy < 0: beta = -beta
            
            theta = beta - self.angle
            # Normalize Angle
            if theta > np.pi: theta -= 2 * np.pi
            if theta < -np.pi: theta += 2 * np.pi

            # CONSTRUCT STATE: [Laser(20), Dist, Angle, LastAct1, LastAct2, VelX, VelW]
            robot_state = [local_dist, theta, self.last_action[0], self.last_action[1], 
                           self.current_vel_x, self.current_vel_w]
            
            state = np.append(self.scan_data, robot_state)

            # 4. Get Action from Brain
            action = self.network.get_action(np.array(state))
            
            # 5. Execute Action
            vel = Twist()
            vel.linear.x = (action[0] + 1) / 2  # Map -1..1 to 0..1
            vel.angular.z = action[1]
            self.vel_pub.publish(vel)
            
            self.last_action = action
            time.sleep(0.1) # 10Hz Control Loop

    def stop_robot(self):
        vel = Twist()
        vel.linear.x = 0.0
        vel.angular.z = 0.0
        self.vel_pub.publish(vel)

    def visualize_goals(self, gx, gy, lx, ly):
        ma = MarkerArray()
        
        # Green Marker: Global Goal
        m1 = Marker()
        m1.header.frame_id = "odom"
        m1.id = 1
        m1.type = Marker.SPHERE
        m1.action = Marker.ADD
        m1.pose.position.x = gx
        m1.pose.position.y = gy
        m1.pose.orientation.w = 1.0
        m1.scale.x = 0.3; m1.scale.y = 0.3; m1.scale.z = 0.3
        m1.color.a = 1.0; m1.color.g = 1.0
        ma.markers.append(m1)

        # Blue Marker: Local Waypoint
        m2 = Marker()
        m2.header.frame_id = "odom"
        m2.id = 2
        m2.type = Marker.SPHERE
        m2.action = Marker.ADD
        m2.pose.position.x = lx
        m2.pose.position.y = ly
        m2.pose.orientation.w = 1.0
        m2.scale.x = 0.15; m2.scale.y = 0.15; m2.scale.z = 0.15
        m2.color.a = 1.0; m2.color.b = 1.0
        ma.markers.append(m2)

        self.marker_pub.publish(ma)

# --- MAIN LOOP ---
if __name__ == "__main__":
    try:
        planner = GlobalPlannerNode()
        
        while not rospy.is_shutdown():
            print("\n--- READY FOR NEW MISSION ---")
            try:
                x_input = float(input("Enter Global Goal X: "))
                y_input = float(input("Enter Global Goal Y: "))
                planner.navigate_to(x_input, y_input)
            except ValueError:
                print("Invalid input. Please enter numbers.")
            
    except rospy.ROSInterruptException:
        pass
