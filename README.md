Deep Reinforcement Learning (TD3) for Robot Navigation
This project trains a Turtlebot3 robot to navigate a custom environment using the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm. The robot learns to reach a target goal while avoiding obstacles using LaserScan (LIDAR) data.

Prerequisites
ROS Noetic

Python 3.8+

PyTorch

Gazebo & Turtlebot3 Simulations

Configuration
Before running, ensure your .bashrc is configured correctly:

Bash

export TURTLEBOT3_MODEL=burger
source /opt/ros/noetic/setup.bash
source ~/workspaces/td_ws/DRL-robot-navigation/catkin_ws/devel/setup.bash
🚀 How to Train
To start training the robot from scratch or resume training:

Open the training script: train_velodyne_td3.py

Select your mode:

Start Fresh: Set load_model = False (Line ~173).

Resume Training: Set load_model = True.

Select your Map:

In train_velodyne_td3.py, set the launch file:

Python

env = GazeboEnv("mylaunch.launch", environment_dim) # Custom Map
# OR
env = GazeboEnv("empty_training.launch", environment_dim) # Empty World
Run the command:

Bash

python3 train_velodyne_td3.py
