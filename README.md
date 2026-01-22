Prerequisites

OS: Ubuntu 20.04 (Focal)

ROS Distro: Noetic

Simulation: Gazebo

Python: 3.8+ (PyTorch, NumPy)

Setup

Create a workspace mkdir -p ~/catkin_ws/src cd ~/catkin_ws/src

Clone the repository git clone https://github.com/harisharandangi/DRL-robot-navigation.git

Install Python dependencies pip3 install torch torchvision numpy

Build workspace cd ~/catkin_ws catkin_make source devel/setup.bash

Training Pipeline

This project uses a sequential "Zero-to-Hero" training strategy to accelerate learning.

Stage 1: Kinematics Pre-Training (Empty World) Goal: Train the agent to master velocity control and goal-seeking behavior without physical obstacles. Method: Uses a "Virtual Cage" (software boundary) in an infinite empty world to enforce steering constraints.

Commands:

Terminal 1: Launch Empty Environment
roslaunch multi_robot_scenario empty_training.launch

Terminal 2: Start Stage 1 Training
python3 train_lds_empty_world.py

Output: Weights saved to pytorch_models/TD3_lds_empty_actor.pth

Stage 2 & 3: Obstacle Avoidance (Cluttered World) Goal: Train the agent to avoid collisions while maintaining speed. Method: Loads Stage 1 weights. Starts with a tight 1.0m boundary. Automatically expands the boundary (Curriculum Learning) as the agent logs successful episodes.

Commands:

Terminal 1: Launch Cluttered Environment
roslaunch multi_robot_scenario mylaunch.launch

Terminal 2: Start Stage 2/3 Training
python3 train_lds_cluttered.py

Note: This script automatically looks for Stage 1 weights if Stage 2 weights do not exist.

Deployment (Global Planner)

The deployment script implements a hierarchical architecture. A rule-based Global Planner calculates intermediate waypoints, while the trained TD3 Pilot handles local navigation to those waypoints.

Commands:

1. Launch Simulation
roslaunch multi_robot_scenario mylaunch.launch

2. Run the Deployment Node
python3 deploy_lds_global.py

Usage: Once the node is running, enter target coordinates in the terminal when prompted:

Enter Global Goal X: 4.0 Enter Global Goal Y: 0.0
