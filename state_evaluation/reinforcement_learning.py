import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import sys
import timeit
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# MCTS code imports
sys.path.append("..")  # Adds higher directory to python modules path.
from main import MeasurementControlEnvironment
from utils import rotate_about_point

class PolicyValueNetwork(nn.Module):
    """
    Neural net combining policy and value network (state -> (policy, value))
    params:
        state_dims: Number of dimensions in the state space
        action_space_len: Number of dimensions in the action space
    """
    def __init__(self, state_dims, action_space_len):
        super(PolicyValueNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dims, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_space_len+1) # +1 for the value output

    # Called with either one element to determine next action, or a transitions
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Named tuple for transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    """
    Replay memory for storing transitions
    Creates a deque with a maximum length of capacity. Transitions are stored as named tuples.
    parameters:
        capacity: Maximum number of transitions to store
    methods:
        push(*args): Save a transition
        sample(batch_size): Sample a batch of transitions
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class MCTSRLWrapper:
    def __init__(self, env: MeasurementControlEnvironment, model: str, num_actions: int, width_pixels: int=200, 
                 width_meters: float=50, gamma: float=0.99, batch_size: int=64, lr: float=0.001, tau: float=0.005,
                 max_transitions: int=10000):
        """
        Wrapper for MCTS with reinforcement learning on the Measurement Control environment
        
        params:
            env: Measurement Control environment
            model: Name of the model to load or 'new' to create a new model
            num_actions: Number of actions in the action space
            width_pixels: Width of the image in pixels
            width_meters: Width of the image in meters
            gamma: Discount factor
            batch_size: Number of transitions to sample for training
            lr: Learning rate for the optimizer
            tau: Soft update parameter for target network
            max_transitions: Maximum number of transitions to store in replay memory
            
        methods:
            add_transition(state, action, next_state, reward, done): Add a transition to the replay memory
            inference(state): Get the action probabilities and value of a state
            save_model(name): Save the model to a file
            optimize_model(): Optimize the Q network using a batch of transitions from the replay memory, soft update the target Q network
            
        """
        # Check device and use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        # Find observation (nn state) dimensions and output size
        observation_dims = 3 + width_pixels**2 # NN Car state + num image pixels
        
        # Create the target network which is updated slowly for stability and used to create targets 
        self.target_q_network = PolicyValueNetwork(observation_dims, num_actions).to(self.device)
            
        # If we are making a new model
        if model == 'new':
            self.q_network = PolicyValueNetwork(observation_dims, num_actions).to(self.device)
            
        # Otherwise check if the model exists by listing the models in the models directory
        else:
            try:
                self.q_network = torch.load(f'models/{model}').to(self.device)
            except:
                raise Exception("Model does not exist")
            
        # Copy the weights of the q network to the target q network
        self.target_q_network.load_state_dict(self.q_network.state_dict()) 
        
        # Width of the image in pixels and meters for the state image
        self.width_pixels = width_pixels
        self.width_meters = width_meters
        
        # Hyperparameters
        self.env = env # Measurement Control environment
        self.gamma = gamma # Discount factor
        self.batch_size = batch_size # Number of transitions to sample for training
        self.tau = tau # Soft update parameter for target network (target = tau * q_network + (1 - tau) * target_q_network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, amsgrad=True) # Adam optimizer for training the Q network
        self.replay_memory = ReplayMemory(max_transitions) # Replay memory for storing transitions
        
        print("Model loaded")
        
    def add_transition(self, state, action, next_state, reward, done):
        """
        Add a transition to the replay memory using full environment state
        params: 
            state: Full state tuple
            action: Action taken
            next_state: Next state tuple
            reward: Reward received
            done: Whether the episode is done
        """
        # Convert states to neural net states
        nn_state = get_nn_state(self.env, state, self.device)
        nn_next_state = get_nn_state(self.env, next_state, self.device)
        
        # Convert other components to tensors
        action = torch.tensor([action], dtype=torch.int64, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.tensor([done], dtype=torch.uint8, device=self.device)
        
        # Add the transition to the replay memory
        self.replay_memory.push(nn_state, action, nn_next_state, reward, done)
        
    def inference(self, state: tuple) -> Tuple[np.ndarray, float]:
        """
        Get the action probabilities and value of a state
        params:
            state: Full state tuple
        returns:
            action_probs: Probabilities of each action being the best
            value: Value of the state
        """
        # Convert state to neural net state
        nn_state = get_nn_state(self.env, state, self.device)
        
        # Get the action probabilities and value from the q_network
        with torch.no_grad():
            self.q_network.eval()
            inference = self.q_network(nn_state)
        
        # Put the inference on the CPU and convert to numpy
        inference_np = inference.cpu().numpy()
        print(f'Inference shape: {inference_np.shape}')
        
        # Split the inference into action probabilities and value
        action_probs, value = inference_np[:-1], inference_np[-1]
        print(f'Action probs shape: {action_probs.shape}')
        print(f'Value shape: {value.shape}')
        
        return action_probs, value
        
    def save_model(self, name: str):
        """
        Save the model to a file
        params:
            name: Name of the file to save the model to
        """
        torch.save(self.q_network.state_dict(), f'models/{name}')
        
    def loss(self, transitions: List[Transition]) -> torch.Tensor:
        """
        Calculate the loss for a batch of transitions
        params: 
            transitions: List of transitions
        returns:
            loss: Loss for the batch (MSE between target and predicted Q values)
        """
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        transitions = Transition(*zip(*transitions))
        
        # Pull out the components of the batch and concatinate them (concatenate scalar lists and stack tensor lists)
        state_transitions = torch.stack(transitions.state)
        action_transitions = torch.cat(transitions.action)
        next_state_transitions = torch.stack(transitions.next_state)
        reward_transitions = torch.cat(transitions.reward)
        done_transitions = torch.cat(transitions.done)
        
        # Get max Q target values of next state from target network (max_a' Q_target(s', a'))
        max_next_q_value = self.target_q_network(next_state_transitions).max(dim=1).values
        
        # Calculate the target (r + γ * max_a' Q_target(s', a')), what we want the Q network to predict
        y_targets = reward_transitions + self.gamma * max_next_q_value * ~done_transitions # If done, the target is just the reward
        
        # Get the current q_values, pick the values that are from the actions taken and then squeeze the tensor (remove the extra dimension)
        q_values = self.q_network(state_transitions).gather(1, action_transitions).squeeze()

        # Calculate the loss (difference between the target and the current q_value prediction)
        loss = F.mse_loss(y_targets, q_values)
        
        return loss
    
    def optimize_model(self):
        """
        Optimize the Q network using a batch of transitions from the replay memory, soft update the target Q network
        """
        # Check if there are enough transitions in the replay memory to optimize
        if len(self.replay_memory) < self.batch_size:
            return
        
        # Sample a batch of transitions
        transitions = self.replay_memory.sample(self.batch_size)
        
        # Set the network to training mode
        self.q_network.train()
        
        # Zero the gradients (reset the optimizer)
        self.optimizer.zero_grad()
        
        # Calculate the loss
        loss = self.loss(transitions)
        
        # Backpropagate the loss (calculate gradients for each parameter)
        loss.backward()
        
        # Perform a step of optimization (update the parameters of the q_network)
        self.optimizer.step()

        # Soft update of the target network (slow changes to make training more stable)
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            

# Create state image from car position, OOI corners, car width, car_length and obstacles
def get_image_based_state(env: MeasurementControlEnvironment, state: tuple, width_pixels=200, width_meters=50) -> Tuple[np.ndarray, np.ndarray]:
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

def get_nn_state(env: MeasurementControlEnvironment, state: tuple, device: torch.device) -> torch.Tensor:
    """
    Convert full state into image and then combine car state and flattened image into a single tensor
    params:
        env: ToyMeasurement Control environment
        state: Full state tuple
        device: Device to put the tensor on
    """
    nn_car_state, image = get_image_based_state(env, state)
    
    nn_state = torch.cat((torch.tensor(nn_car_state, dtype=torch.float32, device=device),
                          torch.tensor(image.flatten(), dtype=torch.float32, device=device)))
    
    return nn_state

def plot_state_image(image, title):
    plt.imshow(image.T, cmap='gray', origin='lower')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.show()