import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# IMPORT THE NEW TEST ENV
from velodyne_env_test import GazeboEnv

# ---------------------------------------------------------
# Network Architecture (Must match training)
# ---------------------------------------------------------
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
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    # 1. Config
    seed = 0
    file_name = "TD3_velodyne" 
    environment_dim = 20
    robot_dim = 4
    state_dim = environment_dim + robot_dim
    action_dim = 2

    # 2. Initialize Environment (Connects to existing Gazebo)
    print("Connecting to Gazebo...")
    # NOTE: We no longer pass a launch file path because we assume Gazebo is already running
    env = GazeboEnv(environment_dim)
    
    # 3. Load Model
    print("Loading Model...")
    network = TD3(state_dim, action_dim)
    try:
        network.load(file_name, "./pytorch_models")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure 'TD3_velodyne_actor.pth' is in ./pytorch_models")
        sys.exit(1)

    # 4. Wait for Odom Data (Robot must exist)
    print("Waiting for robot Odom data...")
    while env.last_odom is None:
        time.sleep(0.5)
    print("Robot detected!")

    # 5. Loop
    while True:
        try:
            print("\n--------------------------------")
            gx_input = input("Enter Goal X (or 'q' to quit): ")
            if gx_input.lower() == 'q': break
            gy_input = input("Enter Goal Y: ")

            # Set Goal
            env.set_goal(gx_input, gy_input)

            # Get Initial State relative to this new goal
            state = env.reset()
            done = False
            episode_timesteps = 0

            print(f"Navigating to ({gx_input}, {gy_input})...")

            while not done:
                # Get Action
                action = network.get_action(np.array(state))
                
                # Scale Action: v in [0,1], w in [-1,1]
                a_in = [(action[0] + 1) / 2, action[1]]

                # Step Environment
                next_state, reward, done, target = env.step(a_in)

                # Update State
                state = next_state
                episode_timesteps += 1

                # Feedback every 20 steps
                if episode_timesteps % 20 == 0:
                    dist = np.linalg.norm([env.odom_x - env.goal_x, env.odom_y - env.goal_y])
                    print(f"Step: {episode_timesteps} | Dist to Goal: {dist:.2f}m | Vel: {a_in[0]:.2f}")

                if target:
                    print("SUCCESS: Goal Reached!")
                    break

                if episode_timesteps > 1000:
                    print("FAIL: Timeout (1000 steps)")
                    break

        except ValueError:
            print("Invalid Input. Use numbers.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
