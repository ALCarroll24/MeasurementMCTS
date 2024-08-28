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
from torch.amp import GradScaler, autocast

# MCTS code imports
sys.path.append("..")  # Adds higher directory to python modules path.
from main import MeasurementControlEnvironment
from utils import rotate_about_point, get_pixels_and_values

# Only fully connected layers (untested but not good for image based state)
# class PolicyValueNetwork(nn.Module):
#     """
#     Neural net combining policy and value network (state -> (policy, value))
#     params:
#         state_dims: Number of dimensions in the state space
#         action_space_len: Number of dimensions in the action space
#     """
#     def __init__(self, state_dims, action_space_len):
#         super(PolicyValueNetwork, self).__init__()
#         self.layer1 = nn.Linear(state_dims, 128)
#         self.layer2 = nn.Linear(128, 128)
#         self.layer3 = nn.Linear(128, action_space_len+1) # +1 for the value output

#     # Called with either one element to determine next action, or a transitions
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         return self.layer3(x)

class PolicyValueNetwork(nn.Module):
    """
    Neural net combining policy and value network (state -> (policy, value))
    params:
        vector_state_size: Number of dimensions in the vector state space
        image_state_size: Dimensions of the image state space (H, W)
        action_space_len: Number of dimensions in the action space
    """
    def __init__(self, vector_state_size, image_state_size, action_space_len):
        super(PolicyValueNetwork, self).__init__()
        
        # Convolutional layers for image state processing
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        
        # Compute the size of the flattened feature map after convolutions
        conv_output_size = 4 * (image_state_size[0] // 4) * (image_state_size[1] // 4)
        
        # Fully connected layers for vector state processing
        self.fc_vector = nn.Linear(vector_state_size, 16)
        
        # Fully connected layers for combined feature processing
        self.fc_combined = nn.Linear(16 + conv_output_size, 32)
        
        # Separate heads for policy and value output
        self.policy_head = nn.Linear(32, action_space_len)
        self.value_head = nn.Linear(32, 1)
    
    def forward(self, vector_state, image_state):
        # Process the image state through convolutional layers
        x_img = F.relu(self.conv1(image_state))# -> [batch, out_channels, H, W]
        x_img = F.max_pool2d(x_img, 2)         # -> [batch, out_channels, ~H/2, ~W/2]
        x_img = F.relu(self.conv2(x_img))      # -> [batch, out_channels, ~H/2, ~W/2]
        x_img = F.max_pool2d(x_img, 2)         # -> [batch, out_channels, ~H/4, ~W/4]
        
        # Flatten the feature map
        x_img = x_img.view(1, -1)              # -> [batch, flattened_size]
        
        # Process the vector state through a fully connected layer
        x_vec = F.relu(self.fc_vector(vector_state)) # -> [batch, num_out_neurons]
        
        # Combine the features from both states
        x = torch.cat((x_vec, x_img), dim=1)    # -> [batch, both flattened sizes]
        
        # Further processing through fully connected layers
        x = F.relu(self.fc_combined(x)) # -> [batch, num_out_neurons]
        
        # Separate the outputs into policy and value
        policy = self.policy_head(x) # -> [batch, num_actions]
        value = self.value_head(x) # -> [batch, 1]

        # Normalize the policy to add to 1
        policy = F.softmax(policy, dim=1)
        
        return policy, value



# Named tuple for transitions with separate vector and image states
Transition = namedtuple('Transition',
                        ('vector_state', 'image_state', 'action', 'next_vector_state', 'next_image_state', 'reward', 'done'))

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
    def __init__(self, env: MeasurementControlEnvironment, model: str, num_actions: int, width_pixels: int=30, 
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
        
        # Find the image representation of the state's dimensions using the width of the image
        image_state_dims = [width_pixels, width_pixels] # Image dimensions
        
        # The states not represented in the image are the car's velocity, steering angle and steering angle rate
        vector_state_dims = 3 # [vx, delta, delta_dot]
        
        # Create the target network which is updated slowly for stability and used to create targets 
        self.target_q_network = PolicyValueNetwork(vector_state_dims, image_state_dims, num_actions).to(self.device)
            
        # If we are making a new model
        if model == 'new':
            self.q_network = PolicyValueNetwork(vector_state_dims, image_state_dims, num_actions).to(self.device)
            
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

        # Optimizer and replay memory
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, amsgrad=True) # Adam optimizer for training the Q network
        self.replay_memory = ReplayMemory(max_transitions) # Replay memory for storing transitions
        
        # Create a Gradient Scalar object which allows for mixed precision training (smaller or bigger numbers as needed :)
        self.scalar = GradScaler()
        
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
        car_state, image_state = self.get_nn_state(state)
        next_car_state, next_image_state = self.get_nn_state(next_state)
        print(f'Add transition, image state shape: {image_state.shape}')
        # Find the index of this action in the action space using numpy
        action_index = np.where(np.all(action == self.env.action_space, axis=1))[0][0]
        
        # Convert other components to tensors
        action_index = torch.tensor([action_index], dtype=torch.int64, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.tensor([done], dtype=torch.uint8, device=self.device)
        
        # Add the transition to the replay memory
        self.replay_memory.push(car_state, image_state, action_index, next_car_state, next_image_state, reward, done)
        
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
        nn_car_state, nn_image_state = self.get_nn_state(state)
        
        # For inferencing we disable gradient calculations since they aren't needed for speed
        with torch.no_grad():
            self.q_network.eval()
            
            # When inferencing use autocast for mixed precision training
            with autocast(device_type=self.device.type):
                # Get the action probabilities and value from the q_network
                action_probs, value = self.q_network(nn_car_state, nn_image_state)
        
        # Put both the action probabilities and value on the cpu, convert to numpy and flatten
        action_probs, value = action_probs.cpu().numpy(), value.cpu().numpy()
        action_probs, value = action_probs.flatten(), value.flatten()
        
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
        print(f'Image state size before stack: {transitions.image_state[0].shape}')
        vector_state_transitions = torch.stack(transitions.vector_state)
        image_state_transitions = torch.stack(transitions.image_state)
        action_transitions = torch.cat(transitions.action)
        next_vector_state_transitions = torch.stack(transitions.next_vector_state)
        next_image_state_transitions = torch.stack(transitions.next_image_state)
        reward_transitions = torch.cat(transitions.reward)
        done_transitions = torch.cat(transitions.done)
        print(f'Entire image state shape: {image_state_transitions.shape}')
        
        # Get target network policy and value estimates (max_a' Q_target(s', a'))
        # next_policy, next_value = self.target_q_network(next_vector_state_transitions, next_image_state_transitions)
        
        # Find the best policy at this state by taking the max of the next policy
        # max_next_q_value = next_policy.max(dim=1).values
        
        # Calculate the target (r + γ * max_a' Q_target(s', a')), what we want the Q network to predict
        # y_targets = reward_transitions + self.gamma * max_next_q_value * ~done_transitions # If done, the target is just the reward
        
        # Get the current q_values, pick the values that are from the actions taken and then squeeze the tensor (remove the extra dimension)
        q_values = self.q_network(vector_state_transitions, image_state_transitions)[0].gather(1, action_transitions).squeeze()

        # Calculate the loss (difference between the target and the current q_value prediction)
        # loss = F.mse_loss(y_targets, q_values)
        
        
        
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
        
        # Perform the loss calculation with autocast (Allows for mixed precision training)
        with autocast(device_type=self.device.type):
            # Calculate the loss
            loss = self.loss(transitions)
        
        # Backpropagate the loss (scalar is used to scale the gradients in the calculation for mixed precision training)
        self.scalar.scale(loss).backward()
        
        # Perform a step of optimization (update the parameters of the q_network)
        self.scalar.step(self.optimizer)
        
        # Update the scaler for the next iteration
        self.scalar.update()

        # Soft update of the target network (slow changes to make training more stable)
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def get_nn_state(self, state: tuple, explore_grid: np.ndarray=None, grid_origin: np.ndarray=None)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert full state into image and then combine car state and flattened image into a single tensor
        Args:
            env: ToyMeasurement Control environment
            state: Full state tuple
            explore_grid: Exploration grid to overlay on the image
            grid_origin: Origin of the exploration grid
            
        Returns:
            nn_car_state: Tensor of the car state
            nn_image_state: Tensor of the image state
        """
        # car state is 3,0 vector, image state is class width_pixels x width_pixels
        car_state, image = get_image_based_state(self.env, state, width_pixels=self.width_pixels,
                                                 width_meters=self.width_meters, explore_grid=explore_grid,
                                                 grid_origin=grid_origin)
        nn_car_state = torch.tensor(car_state, dtype=torch.float32, device=self.device).unsqueeze(0) # Add a channel dimension (1,3)
        nn_image_state = torch.tensor(image, dtype=torch.float32, device=self.device).unsqueeze(0) # Add a channel dimension [channel, H, W]
        nn_image_state = nn_image_state.unsqueeze(0) # Add a batch dimension - [batch, channel, H, W]
        
        return nn_car_state, nn_image_state
    
    def plot_state(self, state: tuple, explore_grid: np.ndarray=None, grid_origin: np.ndarray=None):
        """Plot the state image the neural network sees with full environment state

        Args:
            state (tuple): Full state tuple
            grid (np.ndarray, optional): Exploration grid to add to the image. Defaults to None.
            grid_origin (np.ndarray, optional): Origin of the exploration grid. Defaults to None.
        """
        car_state, image = get_image_based_state(self.env, state, width_pixels=self.width_pixels,
                                                 width_meters=self.width_meters, explore_grid=explore_grid,
                                                 grid_origin=grid_origin)
        
        plot_state_image(image, "State Image")

# Create state image from car position, OOI corners, car width, car_length and obstacles
def get_image_based_state(env: MeasurementControlEnvironment, state: tuple, width_pixels=30, 
                          width_meters=50, explore_grid: np.ndarray=None, grid_origin: np.ndarray=None, meters_per_pixel: float=None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the image representation of the state the neural network will see

    Args:
        env (MeasurementControlEnvironment): The game environment
        state (tuple): Full state tuple
        width_pixels (int, optional): Width of image in pixels. Defaults to 30.
        width_meters (int, optional): Width of the image in meters. Defaults to 50.
        explore_grid (np.ndarray, optional): Exploration grid to overlay on image. Defaults to None.
        grid_origin (np.ndarray, optional): Origin of the grid in meters. Defaults to None.
        meters_per_pixel (float, optional): Meters per pixel for the explore grid. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: NN vector car state and NN image state
    """
    
    # Get obstacle means and radii
    obstacle_means, obstacle_radii = env.eval_kd_tree.get_obstacle_points(), env.eval_kd_tree.get_obstacle_radii()

    # Pull out the state components
    car_state, corner_means, corner_covariance, explore_grid, horizon = state
    corner_means = corner_means.reshape(-1, 2) # Reshape to 2D array where each row is a corner point
    
    # Get car collision length, width and car position and yaw
    car_width, car_length = env.car.width, env.car.length
    car_pos, car_yaw = car_state[:2], car_state[3]
    
    # Get normalized point covariances
    pt_traces = env.get_normalized_cov_pt_traces(state)
    
    # Since image is body frame representation of car, obstacles and OOIs. The neural net only needs [vx, delta, delta_dot] as input
    # These are the components of the state which will determine how actions effect the car state, the rest of the state is used to generate the image
    non_image_car_state = car_state[[2, 4, 5]]
    
    # Make the image
    image = np.zeros((width_pixels, width_pixels), dtype=np.float32)
    
    # Calculate the scaling factor from meters to pixels
    scale = width_pixels / width_meters
    
    # If we have the parameters for the explore grid, overlay it on the image (done first to make sure it is in the background)
    if (explore_grid is not None) and (grid_origin is not None) and (meters_per_pixel is not None):        
        # Convert grid into W*Wx2 array where each row is a pixel and the columns are [x_idx, y_idx] combinations
        pixel_points, values = get_pixels_and_values(explore_grid)
        
        # Now convert the pixel points to world coords and then to car frame
        world_coords = grid_origin + meters_per_pixel * pixel_points
        car_trans_points = world_coords - car_state[:2] # rows are now [x_m, y_m] in car frame (but only translated)
        car_frame_points = rotate_about_point(car_trans_points, np.pi/2-car_yaw, np.array([0,0])) # rows are now [x_m, y_m] in car frame
        
        # Find which points are within the image bounds
        in_bounds = (-width_meters/2 <= car_frame_points[:, 0]) & (car_frame_points[:, 0] <= width_meters/2) & \
                    (-width_meters/2 <= car_frame_points[:, 1]) & (car_frame_points[:, 1] <= width_meters/2)
        
        # Place the in bounds points on the image
        in_bounds_pixels = (car_frame_points[in_bounds] * scale + width_pixels / 2).astype(int)
        image[in_bounds_pixels[:, 0], in_bounds_pixels[:, 1]] = values[in_bounds]
    
    # Rotate the obstacle and corner points to the car's yaw angle
    rotated_corners = rotate_about_point(corner_means, np.pi/2-car_yaw, car_pos) # Negative to rotate into a coordinate system where the car is facing up
    rotated_obstacles = rotate_about_point(obstacle_means, np.pi/2-car_yaw, car_pos)
    
    # Subtract the car's position from the rotated points to get the points relative to the car
    rotated_corners -= car_state[:2]
    rotated_obstacles -= car_state[:2]
    
    # Find which points are within the image bounds
    in_bounds_corners = (-width_meters/2 <= rotated_corners[:, 0]) & (rotated_corners[:, 0] <= width_meters/2) & \
                        (-width_meters/2 <= rotated_corners[:, 1]) & (rotated_corners[:, 1] <= width_meters/2)

    # TODO: This does not account for the radius of the obstacles, it only goes onto the image if the center is in bounds
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
    return non_image_car_state, image

def plot_state_image(image, title):
    fig, ax = plt.subplots()
    ax.imshow(image.T, cmap='RdYlGn', origin='lower')
    fig.colorbar(ax.imshow(image.T, cmap='RdYlGn', origin='lower', vmin=-1, vmax=1), ax=ax, label='Value')
    ax.set_title(title)
