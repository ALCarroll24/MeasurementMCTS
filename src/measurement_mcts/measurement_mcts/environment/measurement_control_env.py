import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Tuple
from copy import deepcopy
import timeit
import pickle
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
from measurement_mcts.utils.ui import MatPlotLibUI
from measurement_mcts.environment.car import Car
from measurement_mcts.environment.object_manager import ObjectManager
from measurement_mcts.environment.static_kf_2d import StaticKalmanFilter, measurement_model
from measurement_mcts.utils.utils import min_max_normalize, find_farthest_point, rotate
from measurement_mcts.mcts.mcts import Environment
from measurement_mcts.state_evaluation.hertg import HERTG

class MeasurementControlEnvironment(Environment):
    def __init__(self, init_reset=True, interactive=False):
        # General important parameters
        self.simulation_dt = 0.6 # time step size for forward simulation search
        self.obstacle_punishment = -0.003 # reward for colliding with an obstacle
        self.init_covariance_diag = 8. # Initial diagonal value for all diagonals of (2x2) point covariance matrix
        self.fully_observered_corner_reward = 0.02 # reward for fully observing a corner
        self.horizon_length = 6 # length of the horizon for the environment
        self.car_collision_radius = 5 # Collision radius of the car
        self.obstacle_discount_factor = 0.8 # Discount factor for obstacles in the environment
        
        # Sensor parameters used in Object Manager for observation simulation and minimums for the measurement model
        sensor_min_range = 5. # minimum range for sensor model
        sensor_min_bearing = 5. # minimum bearing for sensor model
        sensor_max_range = 60. # meters
        sensor_max_bearing = np.radians(60) # degrees
        obs_range_dev = 0.2 # standard deviation for the range scaling of measurement model
        obs_bearing_dev = 0.1 # standard deviation for the bearing scaling of measurement model
        
        # Action space parameters
        self.long_acc_options = np.array([-1., -0.5, 0., 0.5, 1.]) # options for longitudinal acceleration (scaled from [-1, 1] to vehicle [-max_acc, max_acc])
        self.steering_acc_options = np.array([-1., -0.25, 0., 0.25, 1.]) # options for steering acceleration (scaled from [-1, 1] to vehicle [-max_steering_alpha, max_steering_alpha])
        action_space = np.array(np.meshgrid(self.long_acc_options, self.steering_acc_options)).T.reshape(-1, 2) # Generate all combinations using the Cartesian product of the two action spaces
        zero_action = np.array([0.0, 0.0]) # Move the zero action to the front of the list (this is the default action)
        zero_action_index = np.where(np.all(action_space == zero_action, axis=1))[0][0]
        self.action_space = np.concatenate((action_space[zero_action_index:], action_space[:zero_action_index]))
                
        # Create a UI object to pass to different classes for easy plotting
        self.ui = MatPlotLibUI(interactive=interactive)

        # Create a car model with the initial state bounds
        init_pos_bounds = np.array([10., 90.])
        init_yaw_bounds = np.array([-np.pi, np.pi])
        self.car = Car(max_range=sensor_max_range, max_bearing=sensor_max_bearing,
                       init_pos_bounds=init_pos_bounds, init_yaw_bounds=init_yaw_bounds, ui=self.ui)
        
        # Create the object manager which manages collision, and getting observations accounting for occlusions
        # Reset parameters for generating random objects
        self.num_obstacles = 5   # Random obstacles to generate on reset
        self.num_occlusions = 5  # Random occlusions to generate on reset
        self.num_oois = 3        # Random OOI's to generate on reset
        object_bounds = np.array([15, 85]) # Bounds for random object generation
        object_size_bounds = np.array([2, 7]) # Bounds for random object size generation
        ooi_size_bounds = np.array([3, 12]) # Bounds for random OOI size generation
        bounding_box_buffer = 5 # (m) Buffer in sensor area checks
        object_min_spacing = 10 # (m) Spacing to prevent object overlap
        
        # Parameters for estimation and noise observations
        self.init_covariance_trace = self.num_oois * 4 * 2 * self.init_covariance_diag # Total trace available, makes trace based rewards normalized to [0, 1]
        self.final_corner_cov_trace = 0.4 * 2 # (m) Covariance trace threshold for each corner to consider fully observed
        self.final_cov_trace = self.final_corner_cov_trace * 4 * self.num_oois # Covariance trace threshold for all corners to consider fully observed
        init_center_stddev = 1. # Standard deviation for the center guess for estimator initialization
        init_width_guess = 5. # Initial guess for the width of the object
        self.object_manager = ObjectManager(self.num_obstacles, self.num_occlusions, self.num_oois, self.car_collision_radius, 
                                            sensor_max_range, sensor_max_bearing, object_bounds=object_bounds,
                                            size_bounds=object_size_bounds, ooi_size_bounds=ooi_size_bounds,
                                            bounding_box_buffer=bounding_box_buffer, object_min_spacing=object_min_spacing,
                                            init_covariance_diag=self.init_covariance_diag, ui=self.ui,
                                            init_center_stddev=init_center_stddev, init_width_guess=init_width_guess,
                                            range_stddev=obs_range_dev, bearing_stddev=obs_bearing_dev,
                                            final_corner_covariance=self.final_corner_cov_trace)
        
        # Create a Static 2d Kalman Filter object
        range_dev = 0.3 # standard deviation for the range scaling of measurement model
        bearing_dev = 0.15 # standard deviation for the bearing scaling of measurement model
        self.skf = StaticKalmanFilter(range_dev=range_dev, min_range=sensor_min_range,
                                      bearing_dev=bearing_dev, min_bearing=sensor_min_bearing, ui=self.ui)
        
        # Flag for whether goal has been reached
        self.done = False
        
        # Save state within class for easy access
        self.state = None
        
        # Do initial reset to set the initial state of each subcomponent at random within bounds
        if init_reset:
            self.reset()
        
        print("Toy Measurement Control Initialized")

    @property
    def N(self):
        """ Number of actions in the action space """
        return len(self.action_space)

    @N.setter
    def N(self, value):
        self.N = value
        
    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Reset the environment to a random state
        
        returns (Tuple[Car, objects_dataframe, explore_grid]) the initial state of the environment
        """
        # Reset the car to a random state (only random position and yaw), velocities and steering angle are set to 0
        car_state = self.car.reset()
        self.car.set_state(car_state)
        
        # Reset the object manager to generate a new set of objects (maintained within class)
        self.object_manager.reset(car_state)
        
        # Get the noisy initial state of the ooi means and covariances
        ooi_means, ooi_covs = self.object_manager.get_noisy_initial_state()
        
        # Place state into tuple format with horizon set to 0
        state = (car_state, ooi_means, ooi_covs, 0)
        
        # Save and return the initial state
        self.state = state
        return state

    # Not needed for now (because mean and cov are seperate from object manager)
    def get_state(self, horizon=0) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, int]:
        '''
        Returns full state -> Tuple[Car state, Object Manager DF, Exploration Grid, horizon]
        '''
        copied_state_list = list(deepcopy(self.state))
        copied_state_list[3] = horizon
        return tuple(copied_state_list)
    
    def set_state(self, state) -> None:
        """
        Set the state of the environment to a specific state.
        
        :param state: (np.ndarray) the state tuple (Car state, OOI means, OOI covariances, horizon)
        """
        self.state = deepcopy(state)
        
    def save_state(self, path, name) -> None:
        """
        Save the state of the environment to a file.
        
        :param state: (np.ndarray) the state tuple (Car state, Object Manager DF, Exploration Grid, horizon)
        :param name: (str) the name of the file to save the state to
        """
        # Save the state to a file
        with open(f'{path}/{name}.pkl', 'wb') as file:
            pickle.dump((self.get_state(), self.object_manager.get_true_state()), file)
        
    def load_state(self, path, name) -> None:
        """
        Load the state of the environment from a file.
        
        :param name: (str) the name of the file to load the state from
        """
        # Load the state from a file
        with open(f'{path}/{name}.pkl', 'rb') as file:
            state, object_true_state = pickle.load(file)
        
        # Set the state of the environment
        self.set_state(state)
        
        # Set the true state of the object manager
        self.object_manager.set_true_state(object_true_state)
    
    def estimate_remaining_points(self, points, car_state):
        """
        Take 2 or 3 points from the observation and estimate the remaining points of the object.
        Used to complete the observation on real/Unity data where only a few points are observed.
        
        :param points: (np.ndarray) the observed points of the object
        :param car_state: (np.ndarray) the state of the car (x, y, yaw)
        :return: (np.ndarray) the original points of the object with estimated points added
        :return: (np.ndarray) the indices of the new estimated points
        """
        # Extract the car position from the car state
        car_position = car_state[:2]
        
        # If there are 3 points, we can estimate the fourth point by adding the two vectors from the intersection point to edges
        if points.shape[0] == 3:
            # Point 1 is the intersection point, point 0 and point 2 are the points at the ends of the plane
            final_point = (points[0]-points[1]) + (points[2] - points[1]) + points[1]
            
            # Return the original points and the index of the new point
            return np.vstack([points, final_point]), np.array([0, 0, 0, 1])
        
        # For two points, we can assume that the object is a square and add the two points that are missing
        elif points.shape[0] == 2:
            # First take the vector between the two points
            between_points = points[1] - points[0]
            
            # Rotate the vector by +-90 degrees
            bp_plus, bp_minus = rotate(between_points, np.pi/2), rotate(between_points, -np.pi/2)
            
            # Find the vector from the vehicle to the first point, giving us the general direction of the object
            to_first_point = points[0] - car_position
            
            # Calculate the dot product between the object direction and the two vectors after normalizing
            dot_plus = to_first_point/np.linalg.norm(to_first_point) @ bp_plus/np.linalg.norm(bp_plus)
            dot_minus = to_first_point/np.linalg.norm(to_first_point) @ bp_minus/np.linalg.norm(bp_minus)
            
            # Take the vector with the larger dot product which is the vector with the a similar direction as the object
            square_vector = bp_plus if dot_plus > dot_minus else bp_minus
            
            # Add the square vector to the two points to get the final points
            final_points = np.vstack([points, points[0] + square_vector, points[1] + square_vector])
            
            # Return the original points and the indices of the new points
            return final_points, np.array([0, 0, 1, 1])
        
        else:
            raise ValueError("Points must be of length 2 or 3")

    # For use with ROS and real/Unity data, needs to be reworked for new object manager no database format    
    # def corner_data_association(self, obs_list: list, object_df: pd.DataFrame, 
    #                             car_state: np.ndarray, node=None) -> Tuple[dict, pd.DataFrame, np.ndarray]:
    #     # Create output observation dictionary which holds ooi_id's as keys and the associated corner indeces as values
    #     obs_dict = {}
        
    #     # First estimate the remaining points of the objects based on the observed points
    #     obs_polys = np.zeros((len(obs_list), 4, 2))
    #     estimated_indices = np.zeros((len(obs_list), 4))
    #     for i, poly in enumerate(obs_list):
    #         obs_polys[i], estimated_indices[i] = self.estimate_remaining_points(poly, car_state)
            
    #     # If there are no objects maintained, add all sets of corners as new objects
    #     if object_df.empty:
    #         for i, poly in enumerate(obs_polys):
    #             # Add the new polygon to the object manager
    #             object_df = self.object_manager.add_ooi(poly, object_df)
                
    #         # Remove all objects from the observation polygons since they have all been applied
    #         obs_polys = np.zeros((0, 4, 2))
            
    #         return obs_dict, object_df, obs_polys, estimated_indices

    #     # Pull out corner points from the object dataframe where the object type is 'ooi'
    #     ooi_df = object_df[object_df['object_type'] == 'ooi']
    #     rects = np.stack(ooi_df['points'].values) # Get the corner points of the OOI's
    #     ooi_ids = np.stack(ooi_df['ooi_id'].values) # Get the OOI id's
        
    #     # First calculate centroids of the polygons
    #     obs_centroids = np.mean(obs_polys, axis=1)
    #     centroids = np.mean(rects, axis=1)
        
    #     # Calculate distance between all pairs of centroids
    #     distances = cdist(obs_centroids, centroids)
        
    #     # Solve the assignment problem to find the best match using scipy
    #     row_ind, col_ind = linear_sum_assignment(distances)
    #     node.get_logger().info(f'Object Assignment:')
    #     node.get_logger().info(f'Row Index: {row_ind}')
    #     node.get_logger().info(f'Col Index: {col_ind}')
        
    #     # Organize the polygons based on the assignment 
    #     assigned_rects = rects[col_ind]
        
    #     # Now with objects associated, perform point to point data association
    #     for i, (obs_rect, maint_rect) in enumerate(zip(obs_polys, assigned_rects)):
    #         distances = cdist(obs_rect, maint_rect)
    #         row_idx, col_idx = linear_sum_assignment(distances)
    #         node.get_logger().info(f'Object {i} Point Assignment:')
    #         node.get_logger().info(f'Row Index: {row_idx}')
    #         node.get_logger().info(f'Col Index: {col_idx}')
    #         obs_dict[ooi_ids[i]] = col_idx
            
    #     # Also organize the observation polygons based on the assignment
    #     obs_polys = obs_polys[col_ind]
            
    #     # TODO: Add non-associated objects to the object manager as new objects
    #     node.get_logger().info(f'obs_dict: {obs_dict}')

    #     return obs_dict, object_df, obs_polys, estimated_indices
    
    def apply_observation(self, observation_indices: dict, observation: dict, car_state: np.ndarray, 
                          ooi_means: np.ndarray, ooi_covs: np.ndarray) -> Tuple[pd.DataFrame, float]:
        # Take a copy of the means and covariances to update before modifying
        new_ooi_means = deepcopy(ooi_means)
        new_ooi_covs = deepcopy(ooi_covs)
        
        # Apply the KF update to the observed corners
        trace_delta_sum = 0. # Sum of the difference in trace made in this update
        fully_observed_corners = 0 # Number of corners that become fully observed
        for ooi_idx, corner_indices in observation_indices.items():
            
            # Go through the indeces of the OOI points that were observed
            for j, c_idx in enumerate(corner_indices):
                # KF update with the observed corner using the previous mean for now
                prev_trace = np.trace(ooi_covs[ooi_idx][c_idx]) # Get the trace of the covariance matrix pre-update
                
                new_mean, new_cov = self.skf.update(ooi_means[ooi_idx][c_idx], ooi_covs[ooi_idx][c_idx], observation[ooi_idx][j], car_state)
                    
                # Check if the trace of the covariance is already below the threshold
                if prev_trace < self.final_corner_cov_trace:
                    pass # No reward for already fully observed corners
                
                # Check if the trace of the covariance is below the threshold after the update
                elif np.trace(new_cov) < self.final_corner_cov_trace:
                    trace_delta_sum += prev_trace - self.final_corner_cov_trace # Add the difference in trace to the sum
                    fully_observed_corners += 1 # Increment the number of fully observed corners
                    
                # Otherwise normal trace update
                else:
                    trace_delta_sum += prev_trace - np.trace(new_cov) # Add the difference in trace to the sum
                
                # Place the new mean and covariance into the copied means and covs
                new_ooi_means[ooi_idx][c_idx] = new_mean
                new_ooi_covs[ooi_idx][c_idx] = new_cov
            
        return new_ooi_means, new_ooi_covs, trace_delta_sum, fully_observed_corners
    
    def step(self, state, action, dt=None, check_failure=False, obs_at_mean=False, print_rewards=False, return_observation=False,
             return_min_obs_dist=False, negative_to_zero=False) -> Tuple[tuple, float, bool]:
        """
        Step the environment by one time step. The action is applied to the car, and the state is observed by the OOI.
        The observation is then passed to the KF for update.
        
        :param state: (np.ndarray) the state (Car state(x,y,yaw), corner means, corner covariances)
        :param action: (np.ndarray) the control input to the car (velocity, steering angle)
        :param dt: (float) the time step size for the simulation
        :param check_failure: (bool) whether to check for failure (hard collision)
        :param obs_at_mean: (bool) whether to get the observation at the mean or at the noisy state
        :param print_rewards: (bool) whether to print the rewards for each step
        :param return_observation: (bool) whether to return the observation
        :param return_min_obs_dist: (bool) whether to return the minimum obstacle distance
        :param negative_to_zero: (bool) whether to set negative rewards to 0
        :return: (tuple, float, bool) the new state, the reward of the state-action pair, and whether the episode is done
        """
        # If dt is not specified, use the default period
        if dt is None:
            dt = self.simulation_dt
        
        # Pull out the state elements
        car_state, ooi_means, ooi_covs, horizon = state
        
        # Increment the horizon
        horizon += 1
        
        # Apply the action to the car and get the next state
        new_car_state = self.car.update(dt, action, starting_state=car_state)
        
        # Now see if the car has collided with any objects in the object manager
        in_collision_obs, in_collision_ocl, in_collision_oois, collision_distances, min_obs_dist = self.object_manager.check_collision(new_car_state)
        
        # Get an observation from the object manager at this new car state
        if obs_at_mean is False:
            # This is for a real update after deciding action with MCTS
            observation_indices, noisy_observation = self.object_manager.get_noisy_observation(new_car_state)
        else:
            # This is for predicted states in MCTS
            observation_indices, noisy_observation = self.object_manager.get_observation_at_mean(new_car_state, ooi_means)
        
        # Apply the observation and get sum of the trace differences and the new object dataframe
        new_ooi_means, new_ooi_covs, trace_delta_sum, fully_observed_corners = \
        self.apply_observation(observation_indices, noisy_observation, new_car_state, ooi_means, ooi_covs)

        # Calculate rewards
        # num_in_collision = len(in_collision_obs) + len(in_collision_ocl) + len(in_collision_oois) # Number of objects in collision
        # obstacle_reward = num_in_collision * self.obstacle_punishment  # Reward for colliding with obstacles
        obstacle_reward = self.obstacle_punishment * np.sum(collision_distances**2) # Reward for colliding with obstacles
        obstacle_reward = obstacle_reward * self.obstacle_discount_factor ** horizon # Discount the reward based on the horizon
        trace_delta_reward = min_max_normalize(trace_delta_sum, 0, self.init_covariance_trace) # Reward for reducing covariance trace
        fully_observed_reward = fully_observed_corners * self.fully_observered_corner_reward # Reward for fully observing a corner
        reward = obstacle_reward + trace_delta_reward + fully_observed_reward # Total reward is sum of all rewards
        
        if negative_to_zero is True and reward < 0:
            reward = 0. # Set negative rewards to 0
        
        # Print rewards if enabled
        if print_rewards:
            print(f'Obstacle Reward: {obstacle_reward}')
            print(f'Trace Delta Reward: {trace_delta_reward}')
            print(f'fully observed Reward: {fully_observed_reward}')
            print(f'Total Reward: {reward}')
        
        # Check if the episode is done
        all_traces = np.trace(new_ooi_covs, axis1=2, axis2=3).flatten() # trace for each corner in flattened array
        done = np.all(all_traces <= self.final_corner_cov_trace) # Check if all traces are below the final corner trace
        
        # Also done if doing simulated update and horizon is equal to the maximum horizon length
        if obs_at_mean is True:
            done = done or horizon >= self.horizon_length
        
        # Check for failure
        failure = min_obs_dist <= 0.
        
        # Combine the updated car state, mean, covariance and horizon into a new state
        new_state = (new_car_state, new_ooi_means, new_ooi_covs, horizon)
        
        # Return options based on the flags
        if return_observation and return_min_obs_dist:
            return new_state, reward, done, noisy_observation, min_obs_dist
        
        if return_observation:
            if check_failure:
                return new_state, reward, done, noisy_observation, failure
            return new_state, reward, done, noisy_observation
        
        if return_min_obs_dist:
            return new_state, reward, done, min_obs_dist
        
        if check_failure:
            return new_state, reward, done, failure
        
        return new_state, reward, done
    
    def draw_state(self, state, title=None, plot=True, root_node=None, 
                   rew=None, q_val=None, qu_val=None, scaling=1, bias=0,
                   max=4, get_fig_ax: bool=False,
                   observation=None, hertg=None, hertg_scale=10) -> None:
        """
        Draw the state on the UI.
        
        :param state: (np.ndarray) the state of the car and OOI (position, corner means, corner covariances)
        :param plot: (bool) whether to plot the state
        :param root_node: (Node) the root node of the MCTS tree (used for drawing the simulated states when passed)
        :param rew: (bool) whether to size based on reward
        :param q_val: (bool) whether to size based on Q value
        :param qu_val: (bool) whether to size based on upper confidence bound
        :param scaling: (float) the scaling factor for the radius of the points
        :param bias: (float) the bias to add to the radius of the points
        :param max: (float) the maximum radius of the points
        :param get_fig_ax: (bool) whether to return the figure and axis
        :param observation: (np.ndarray) the observation to display
        :param hertg: (HERTG) the HERTG object to display
        """
        
        # Pull elements out of the state
        car_state, ooi_means, ooi_covs, horizon = state
        
        # Simulate collision and observation to get objects in collision and observation display
        in_collision_obs, in_collision_ocl, in_collision_oois, collision_distances, min_obs_dist = self.object_manager.check_collision(car_state)
        observation_indices = self.object_manager.get_observation_indices(car_state)
        
        # Draw the car state
        self.car.draw_car_state(car_state)
        
        # Draw the objects in the dataframe
        self.object_manager.draw_objects(car_state, in_collision_obs, in_collision_ocl, in_collision_oois, 
                                         observation_indices=observation_indices, observation=observation,
                                         ooi_means=ooi_means, ooi_covs=ooi_covs)
        
        # Draw the HERTG
        if hertg is not None:
            hertg.draw_target_point(state)
        
        # Draw the simulated states
        if root_node is not None:
            self.draw_simulated_states(root_node, rew=rew, q_val=q_val, qu_val=qu_val, scaling=scaling,
                                       bias=bias, max=max, hertg=hertg, hertg_scale=hertg_scale)
        
        if plot:
            return self.ui.plot(get_fig_ax=get_fig_ax, title=title)
    
    # Old method using a circle patch for each predicted state
    # def draw_simulated_states(self, node, rew=False, q_val=False, qu_val=False, scaling=1, bias=0, max=4) -> None:
    #     """
    #     Recursively go through tree of simulated states and draw points of each position sized by the reward
    #     :param node: (Node) the node to draw the simulated states from
    #     :param color: (str) the color of the points to draw 
    #     :param rew: (bool) whether to size based on reward
    #     :param q_val: (bool) whether to size based on Q value
    #     :param qu_val: (bool) whether to size based on upper confidence bound
    #     :param scaling: (float) the scaling factor for the radius of the points
    #     :param bias: (float) the bias to add to the radius of the points
    #     """
    #     if not (rew or q_val or qu_val):
    #         raise ValueError("Must select at least one of rew, q_val, or qu_val to draw simulated states")
        
    #     if rew:
    #         # Rewards are already normalized between 0 and 1, add 0.05 to make all rewards visible
    #         radius = node.reward
            
    #     elif q_val:
    #         # Q values are normalized between 0 and 1, add 0.05 to make all rewards visible
    #         radius = node.Q
            
    #     elif qu_val:
    #         radius = node.Q + node.U
            
    #     color = 'g' if radius >= 0 else 'r'
    #     radius = np.abs(radius) * scaling + bias
    #     radius = np.clip(radius, 0, max)
            
    #     # Place a point at the state of this node
    #     self.ui.draw_point(node.state[0][:2], color=color, radius=radius, alpha=0.2)
        
    #     # Draw the children recursively by calling this function
    #     for child in node.children.values():
    #         self.draw_simulated_states(child, rew=rew, q_val=q_val, qu_val=qu_val, scaling=scaling, bias=bias, max=max)
    
    def draw_simulated_states(self, node, rew=False, q_val=False, qu_val=False, scaling=1, bias=0, max=4, hertg=None, hertg_scale=10) -> None:
        """
        Recursively go through tree of simulated states and draw points of each position sized by the reward
        :param node: (Node) the node to draw the simulated states from
        :param color: (str) the color of the points to draw 
        :param rew: (bool) whether to size based on reward
        :param q_val: (bool) whether to size based on Q value
        :param qu_val: (bool) whether to size based on upper confidence bound
        :param scaling: (float) the scaling factor for the radius of the points
        :param bias: (float) the bias to add to the radius of the points
        :param max: (float) the maximum radius of the points
        """
        
        ucb_states_pos = np.empty((0, 2))
        ucb_rewards_pos = np.empty(0)
        ucb_states_neg = np.empty((0, 2))
        ucb_rewards_neg = np.empty(0)
        rollout_states_pos = np.empty((0, 2))
        rollout_rewards_pos = np.empty(0)
        rollout_states_neg = np.empty((0, 2))
        rollout_rewards_neg = np.empty(0)
        hertg_states_pos = np.empty((0, 2))
        hertg_rewards_pos = np.empty(0)
        
        def accumulate_data(node, rew=False, q_val=False, qu_val=False, scaling=1, bias=0, max=4, hertg=None, hertg_scale=hertg_scale):
            nonlocal ucb_states_pos, ucb_rewards_pos, ucb_states_neg, ucb_rewards_neg
            nonlocal rollout_states_pos, rollout_rewards_pos, rollout_states_neg, rollout_rewards_neg
            nonlocal hertg_states_pos, hertg_rewards_pos
            
            if not (rew or q_val or qu_val):
                raise ValueError("Must select at least one of rew, q_val, or qu_val to draw simulated states")
            
            if rew:
                # Rewards are already normalized between 0 and 1, add 0.05 to make all rewards visible
                reward = node.reward
                
            elif q_val:
                # Q values are normalized between 0 and 1, add 0.05 to make all rewards visible
                reward = node.Q
                
            elif qu_val:
                reward = node.Q + node.U
                
            radius = np.abs(reward) * scaling + bias
            radius = np.clip(radius, 0, max)

            if node.is_expanded:
                if reward >= 0:
                    ucb_states_pos = np.vstack([ucb_states_pos, node.state[0][:2]])
                    ucb_rewards_pos = np.append(ucb_rewards_pos, radius)
                else:
                    ucb_states_neg = np.vstack([ucb_states_neg, node.state[0][:2]])
                    ucb_rewards_neg = np.append(ucb_rewards_neg, radius)
            else:
                if reward >= 0:
                    rollout_states_pos = np.vstack([rollout_states_pos, node.state[0][:2]])
                    rollout_rewards_pos = np.append(rollout_rewards_pos, radius)
                else:
                    rollout_states_neg = np.vstack([rollout_states_neg, node.state[0][:2]])
                    rollout_rewards_neg = np.append(rollout_rewards_neg, radius)
            
            if not node.children and hertg is not None:
                reward = hertg.get_reward(node.state)
                
                if reward < 0:
                    raise ValueError("HERTG reward must be positive")
                
                # radius = reward * scaling + bias
                radius = reward * hertg_scale * scaling + bias
                
                hertg_states_pos = np.vstack([hertg_states_pos, node.state[0][:2]])
                hertg_rewards_pos = np.append(hertg_rewards_pos, radius)

            # Draw the children recursively by calling this function
            for child in node.children.values():
                accumulate_data(child, rew=rew, q_val=q_val, qu_val=qu_val, scaling=scaling, bias=bias, max=max, hertg=hertg, hertg_scale=hertg_scale)
        
        # Place data into arrays recursively and then place into ui for later plotting
        accumulate_data(node, rew=rew, q_val=q_val, qu_val=qu_val, scaling=scaling, bias=bias, max=max, hertg=hertg)
        self.ui.update_future_state_data(ucb_states_pos.T, ucb_rewards_pos, ucb_states_neg.T, ucb_rewards_neg,
                                         rollout_states_pos.T, rollout_rewards_pos, rollout_states_neg.T, rollout_rewards_neg,
                                         hertg_states_pos.T, hertg_rewards_pos)
        

    def draw_state_set(self, state_set, title_perm=None, rewards=None, roots=None, scaling=4, bias=0.1, max=1., rew=True):
        """
        Use matplotlib animate to create a video with the normal state display over time
        params: state_set - list of states to display
        """
        def animate(i):
            # Clear all existing patches from the axis
            for patch in ax.patches:
                patch.remove()
            
            # Draw the state create artists in UI class
            if (roots is not None) and i!=0:
                self.draw_state(state_set[i], plot=False, root_node=roots[i-1], rew=rew, scaling=scaling, bias=bias, max=max)
            else:
                self.draw_state(state_set[i], plot=False)
            
            # Add artists to the axis
            for artist in self.ui.get_artists():
                ax.add_patch(artist)
                
            # Add title
            if title_perm is not None and rewards is None:
                ax.set_title(title_perm)
            if title_perm is not None and rewards is not None:
                ax.set_title(f'{title_perm}\nstate {i}, {round(state_set[i][0][2], 2)} m/s, reward: {round(rewards[i-1], 2)}, cumulative reward: {round(np.sum(rewards[:i]), 2)}')
            
            return ax.patches
        
        # Get the figure and axis from the UI
        fig, ax = self.ui.plot(get_fig_ax=True)
        plt.close()
        
        ani = FuncAnimation(fig, animate, frames=len(state_set), interval=200, blit=False)
            
        # Display the animation in the notebook
        display(HTML(ani.to_jshtml()))
    
    def draw_action_set(self, root, action_set):
        """
        Use matplotlib animate to create a video with the normal state display over time with actions
        params: root - the root node of the MCTS tree with starting state
                action_set - list of actions to take in the environment
        """
        # Function called by matplotlib animate to get a frame of the video
        def animate(i):
            # Use the state and axis from the parent function
            nonlocal current_node
            nonlocal state
            nonlocal ax
            nonlocal last_index
                        
            # Clear all existing patches from the axis
            for patch in ax.patches:
                patch.remove()
            
            # Get the index of this action using the action space
            action_idx = np.where(np.all(self.action_space == action_set[i], axis=1))[0][0]
            
            # Draw the state create artists in UI class
            self.draw_state(state, plot=False)
            
            # Add artists to the axis
            for artist in self.ui.get_artists():
                ax.add_patch(artist)
                
            # Add background image if it exists
            if self.ui.background_image is not None:
                ax.imshow(self.ui.background_image[0], extent=self.ui.background_image[1], alpha=self.ui.background_image[2])
            
            # Update state for next iteration if it hasn't already been called
            if last_index != i:
                state = current_node.state
                current_node = current_node.children[action_idx]
            last_index = i
            
            return ax.patches
        
        # Get the figure and axis from the UI
        fig, ax = self.ui.plot(get_fig_ax=True)
        plt.close()
        
        # Track the last index to avoid desyncing from the action set when matplotlib calls the same frame multiple times
        last_index = -1
        
        # Start traversal at the root node
        current_node = root
        state = root.state
        ani = FuncAnimation(fig, animate, frames=len(action_set)-1, interval=200, blit=False)
            
        # Display the animation in the notebook
        display(HTML(ani.to_jshtml()))
    