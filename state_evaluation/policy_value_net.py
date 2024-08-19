import math
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# MCTS code imports
sys.path.append("..")  # Adds higher directory to python modules path.
from main import ToyMeasurementControl
from utils import rotate_about_point

class PolicyValueNetwork(nn.Module):
    """
    Neural net combining policy and value network (state -> (policy, value))
    params:
        state_dims: Number of dimensions of the state space
        action_length: Number of actions in the action space
    """
    def __init__(self, state_dims, action_length):
        super(PolicyValueNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dims, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_length+1) # +1 for the value output

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Named tuple for transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    """
    Replay memory for storing transitions
    Creates a deque with a maximum length of capacity. Transitions are stored as named tuples.
    
    methods:
        push(*args): Save a transition
        sample(batch_size): Sample a batch of transitions
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class MCTSRLWrapper():
    def __init__(self, q_network: PolicyValueNetwork, target_q_network: PolicyValueNetwork,
                 replay_memory: ReplayMemory, gamma: float=0.99, batch_size: int=64, 
                 lr: float=0.001, tau: float=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        self.q_network = q_network.to(self.device) # Q Network which is changed only after full batch is processed
        self.target_q_network = target_q_network.to(self.device) # Q network which is updated during batch training
        self.replay_memory = replay_memory # Replay memory for storing transitions
        self.gamma = gamma # Discount factor
        self.batch_size = batch_size # Number of transitions to sample for training
        self.tau = tau # Soft update parameter for target network (target = tau * q_network + (1 - tau) * target_q_network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, amsgrad=True) # Adam optimizer for training the Q network
        
    def loss(self, batch):
        """
        Calculate the loss for a batch of transitions
        """
        # Pull out the components of the batch
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch
        
        # Get max Q target values of next state from target network (max_a' Q_target(s', a'))
        max_next_q_value = self.target_q_network(next_state_batch).max(dim=1).values
        
        # Calculate the target (r + γ * max_a' Q_target(s', a'))
        y_targets = reward_batch + self.gamma * max_next_q_value * (1 - done_batch) # If done, the target is just the reward
        
        # Get the q_values and reshape to match y_targets
        q_values = self.q_network(state_batch).gather(1, action_batch)
        
        # Calculate the loss
        loss = F.mse_loss(y_targets, q_values)
        
        return loss
    
    def optimize_model(self):
        # Check if there are enough transitions in the replay memory to optimize
        if len(self.replay_memory) < self.batch_size:
            return
        
        # Sample a batch of transitions
        transitions = self.replay_memory.sample(self.batch_size)
        
        # Set the network to training mode
        self.q_network.train()
        
        # Zero the gradients
        self.optimizer.zero_grad()
        
        # Calculate the loss
        loss = self.loss(transitions)
        
        # Backpropagate the loss
        loss.backward()
        
        # Perform a step of optimization
        self.optimizer.step()
        
        # Soft update of the target network
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            

# Create state image from car position, OOI corners, car width, car_length and obstacles
def get_image_based_state(env: ToyMeasurementControl, state: tuple, width_pixels=200, width_meters=50):
    # Get car collision length and width
    car_width, car_length = env.car.width, env.car.length
    
    # Get obstacle means and radii
    obstacle_means, obstacle_radii = env.eval_kd_tree.get_obstacle_points(), env.eval_kd_tree.get_obstacle_radii()

    # Pull out the state components
    car_state, corner_means, corner_covariance, horizon = state
    corner_means = corner_means.reshape(-1, 2) # Reshape to 2D array where each row is a corner point
    
    # Get normalized point covariances
    pt_traces = env.get_normalized_cov_pt_traces(state)
    
    # Since image is body frame representation of car, obstacles and OOIs. The neural net only needs [vx, delta, delta_dot] as input
    # These are the components of the state which will determine how actions effect the car state, the rest of the state is used to generate the image
    nn_car_state = car_state[[2, 4, 5]]
    
    # Make the image
    image = np.zeros((width_pixels, width_pixels), dtype=np.float32)
    
    # Calculate the scaling factor from meters to pixels
    scale = width_pixels / width_meters
    
    # Rotate the obstacle and corner points to the car's yaw angle
    car_pos, car_yaw = car_state[:2], car_state[3]
    rotated_corners = rotate_about_point(corner_means, np.pi/2-car_yaw, car_pos) # Negative to rotate into a coordinate system where the car is facing up
    rotated_obstacles = rotate_about_point(obstacle_means, np.pi/2-car_yaw, car_pos)
    
    # Subtract the car's position from the rotated points to get the points relative to the car
    rotated_corners -= car_state[:2]
    rotated_obstacles -= car_state[:2]
    
    # Find which points are within the image bounds
    in_bounds_corners = (-width_meters/2 <= rotated_corners[:, 0]) & (rotated_corners[:, 0] <= width_meters/2) & \
                        (-width_meters/2 <= rotated_corners[:, 1]) & (rotated_corners[:, 1] <= width_meters/2)

    in_bounds_obstacles = (-width_meters/2 <= rotated_obstacles[:, 0]) & (rotated_obstacles[:, 0] <= width_meters/2) & \
                          (-width_meters/2 <= rotated_obstacles[:, 1]) & (rotated_obstacles[:, 1] <= width_meters/2)

    # Convert the car frame in bounds points to pixel coordinates
    in_bounds_corner_pixels = (rotated_corners[in_bounds_corners] * scale + width_pixels / 2).astype(int)
    in_bounds_obstacle_pixels = (rotated_obstacles[in_bounds_obstacles] * scale + width_pixels / 2).astype(int)
    in_bounds_obstacle_radii_pixels = (obstacle_radii[in_bounds_obstacles] * scale).astype(int)
    
    # First place obstacles so that rewards and car overlay them
    for i, point in enumerate(in_bounds_obstacle_pixels):
        x_pixel, y_pixel = point
        radius_pixel = in_bounds_obstacle_radii_pixels[i]
        x, y = np.ogrid[-x_pixel:width_pixels-x_pixel, -y_pixel:width_pixels-y_pixel]
        mask = x*x + y*y <= radius_pixel*radius_pixel
        image[mask] = -1.0
        
    # Place the car (draw a rectangle at the center given length and width), car is always facing up (positive x axis)
    car_width_pixels = int(car_width * scale)
    car_length_pixels = int(car_length * scale)
    car_max_x_index = car_width_pixels + width_pixels // 2
    car_max_y_index = car_length_pixels + width_pixels // 2
    image[-car_max_x_index:car_max_x_index, -car_max_y_index:car_max_y_index] = -0.5
    
    # Place the corners
    image[in_bounds_corner_pixels[:, 0], in_bounds_corner_pixels[:, 1]] = pt_traces[in_bounds_corners]
        
    # Return the neural net state and the image
    return nn_car_state, image