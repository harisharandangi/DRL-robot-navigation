import os
import time
import numpy as np
import torch
import torch.nn as nn
from importlib import import_module

# --- CONFIG ---
laser_env = import_module("2dlaser_custom") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
environment_dim = 20
robot_dim = 4
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1
file_name = "TD3_velodyne" 

# --- NETWORK ARCHITECTURE ---
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

import torch.nn.functional as F

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.max_action = max_action

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )

# --- MAIN ---
env = laser_env.GazeboEnv("mylaunch.launch", environment_dim)
time.sleep(5)

network = TD3(state_dim, action_dim, max_action)
try:
    network.load(file_name, "./pytorch_models")
    print("Model loaded successfully!")
except:
    print("Error loading model.")
    exit()

while True:
    # 1. Ask User for Goal
    try:
        print("\n--- NEW MISSION ---")
        target_x = float(input("Enter Goal X: "))
        target_y = float(input("Enter Goal Y: "))
    except ValueError:
        print("Invalid input. Try numbers.")
        continue

    # 2. Update Environment
    env.set_custom_goal(target_x, target_y)
    
    # 3. Get Initial State
    state = env.reset() # This reset doesn't move the robot, just calcs distance
    done = False
    
    print(f"Navigating to ({target_x}, {target_y})...")

    # 4. Navigation Loop
    while not done:
        action = network.get_action(np.array(state))
        a_in = [(action[0] + 1) / 2, action[1]]
        
        state, reward, done, target = env.step(a_in)

        if target:
            print(">>> GOAL REACHED! <<<")
            break
        
        # Note: If it crashes (collision), done becomes True and loop breaks.
        # But since we aren't resetting the world, the robot just stays there.
        if done and not target:
            print(">>> CRASHED! stopping... <<<")
