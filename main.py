import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from ui import MatPlotLibUI
from car import Car
from object_manager import ObjectManager
from static_kf_2d import StaticKalmanFilter, measurement_model
from utils import min_max_normalize
from mcts.mcts import Environment
from exploration_grid import ExplorationGrid
from copy import deepcopy
import timeit
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

class MeasurementControlEnvironment(Environment):
    def __init__(self, init_reset=True):
        # General important parameters
        self.final_cov_trace = 0.03 # Covariance trace threshold for stopping the episode (normalized from (0, initial trace)-> (0, 1))
        self.simulation_dt = 0.3 # time step size for forward simulation search
        self.obstacle_punishment = -10. # reward for colliding with an obstacle
        self.init_covariance_diag = 8. # Initial diagonal value for all diagonals of (2x2) point covariance matrix
        self.explored_cell_reward = 0.0001 # reward for exploring a cell
        
        # Sensor parameters used in Object Manager for observation simulation and minimums for the measurement model
        sensor_min_range = 5. # minimum range for sensor model
        sensor_min_bearing = 5. # minimum bearing for sensor model
        sensor_max_range = 60. # meters
        sensor_max_bearing = np.radians(60) # degrees
        
        # Action space parameters
        long_acc_options = np.array([-1., -0.5, 0., 0.5, 1.]) # options for longitudinal acceleration (scaled from [-1, 1] to vehicle [-max_acc, max_acc])
        steering_acc_options = np.array([-1., -0.25, 0., 0.25, 1.]) # options for steering acceleration (scaled from [-1, 1] to vehicle [-max_steering_alpha, max_steering_alpha])
        action_space = np.array(np.meshgrid(long_acc_options, steering_acc_options)).T.reshape(-1, 2) # Generate all combinations using the Cartesian product of the two action spaces
        zero_action = np.array([0.0, 0.0]) # Move the zero action to the front of the list (this is the default action)
        zero_action_index = np.where(np.all(action_space == zero_action, axis=1))[0][0]
        self.action_space = np.concatenate((action_space[zero_action_index:], action_space[:zero_action_index]))
                
        # Create a UI object to pass to different classes for easy plotting
        self.ui = MatPlotLibUI()
                
        # Create a car model with the initial state bounds
        init_pos_bounds = np.array([10., 90.])
        init_yaw_bounds = np.array([-np.pi, np.pi])
        self.car = Car(max_range=sensor_max_range, max_bearing=sensor_max_bearing,
                       init_pos_bounds=init_pos_bounds, init_yaw_bounds=init_yaw_bounds, ui=self.ui)
        
        # Create the object manager which manages collision, and getting observations accounting for occlusions
        # Parameters are mainly for generating random objects
        num_obstacles = 5   # Random obstacles to generate on reset
        num_occlusions = 5  # Random occlusions to generate on reset
        num_oois = 4        # Random OOI's to generate on reset
        self.init_covariance_trace = 4 * self.init_covariance_diag # Initial trace for one OOI (this defines a reward of 1 for reducing the trace of an OOI to 0)
        car_collision_radius = 3.0 # Collision radius of the car
        object_bounds = np.array([15, 85]) # Bounds for random object generation
        object_size_bounds = np.array([1, 10]) # Bounds for random object size generation
        self.object_manager = ObjectManager(num_obstacles, num_occlusions, num_oois, car_collision_radius, 
                                            sensor_max_range, sensor_max_bearing, object_bounds=object_bounds,
                                            size_bounds=object_size_bounds, init_covariance_diag=self.init_covariance_diag,
                                            ui=self.ui)
        
        # Create a Static 2d Kalman Filter object
        range_dev = 1. # standard deviation for the range scaling of measurement model
        bearing_dev = 0.5 # standard deviation for the bearing scaling of measurement model
        self.skf = StaticKalmanFilter(range_dev=range_dev, min_range=sensor_min_range,
                                      bearing_dev=bearing_dev, min_bearing=sensor_min_bearing, ui=self.ui)
        
        # Exploration grid which gives rewards for exploring the environment
        meters_per_pixel = 1 # meters per pixel of the grid
        explore_grid_bounds = np.array([[5, 95], [5, 95]]) # bounds of the grid
        self.explore_grid = ExplorationGrid(explore_grid_bounds, meters_per_pixel, sensor_max_range, sensor_max_bearing, ui=self.ui)
        
        # Flag for whether goal has been reached
        self.done = False
        
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

    # Not using or worrying about yet
    # def update(self, action):
    #     # Update the car's state based on the control inputs
    #     self.car.update(self.simulation_dt, action)
        
    #     observable_corners, indeces = self.ooi.get_noisy_observation(self.car.get_state(), draw=self.draw)

    #     # If there are no observable corners, skip the update
    #     if len(indeces) > 0:
    #         self.skf.update(observable_corners, indeces, self.car.get_state())
        
    #     # Check if we are done (normalized covariance trace is below threshold)
    #     done = self.check_done(self.get_state())
    #     if done:
    #         print("Goal Reached")
        
    #     # Print normalized covariance trace compared to final trace
    #     trace_normalized = min_max_normalize(np.trace(self.skf.P_k), 0, self.covariance_trace_init)
    #     print(f'Normalized Covariance Trace: {np.round(trace_normalized, 4)} Final: {self.final_cov_trace}')
        
    def reset(self, first_update=True, print_rewards=False) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, int]:
        """
        Reset the environment to a random state
        
        returns (Tuple[Car, objects_dataframe, explore_grid]) the initial state of the environment
        """
        # Reset the car to a random state (only random position and yaw), velocities and steering angle are set to 0
        car_state = self.car.reset()
        
        # Reset the object manager to generate a new set of objects
        object_df = self.object_manager.reset(car_state)
        
        # Reset the exploration grid
        explore_grid = self.explore_grid.reset()
        
        # Place state into tuple format with horizon set to 0
        state = (car_state, object_df, explore_grid, 0)
        
        # Set the state of the subclasses to this new random initial state
        self.set_state(state)
        
        # If we are doing the first update, then update grid and KF with the initial step
        # This is done to remove reward gotten for the initial state (which is not a real action)
        if first_update:
            # Use the first action (no acceleration) in the action space to get the first state
            state, reward, done = self.step(state, self.action_space[0], print_rewards=print_rewards)
            
            # Unpack state and update subclass states
            self.set_state(state)
        
        # Return the initial state
        return state
        
    def get_state(self, horizon=0) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, int]:
        '''
        Returns full state -> Tuple[Car state, Object Manager DF, Exploration Grid, horizon]
        '''
        return self.car.get_state(), self.object_manager.get_df(), self.explore_grid.get_grid(), horizon
    
    def set_state(self, state) -> None:
        """
        Set the state of the environment to a specific state.
        
        :param state: (np.ndarray) the state tuple (Car state, Object Manager DF, Exploration Grid, horizon)
        """
        # Set the car state
        self.car.set_state(state[0])
        
        # Set the object manager state
        self.object_manager.set_df(state[1])
        
        # Set the exploration grid state
        self.explore_grid.set_grid(state[2])
    
    def step(self, state, action, dt=None, print_rewards=False) -> Tuple[tuple, float, bool]:
        """
        Step the environment by one time step. The action is applied to the car, and the state is observed by the OOI.
        The observation is then passed to the KF for update.
        
        :param state: (np.ndarray) the state (Car state(x,y,yaw), corner means, corner covariances)
        :param action: (np.ndarray) the control input to the car (velocity, steering angle)
        :return: (tuple, float, bool) the new state, the reward of the state-action pair, and whether the episode is done
        """
        # If dt is not specified, use the default period
        if dt is None:
            dt = self.simulation_dt
        
        # Pull out the state elements
        car_state, object_df, explore_grid, horizon = state
        
        # Increment the horizon
        horizon += 1
        
        # Apply the action to the car and get the next state
        new_car_state = self.car.update(dt, action, starting_state=car_state)
        
        # Now see if the car has collided with any objects in the object manager
        objects_in_collision_df = self.object_manager.check_collision(new_car_state)
        
        # Get an observation from the object manager at this new car state
        observation_dict, new_object_df = self.object_manager.get_observation(new_car_state, df=object_df)
        
        # Get the ooi observed and the indeces of the corners observed
        trace_delta_sum = 0. # Sum of the difference in trace made in this update
        for ooi_id, observed_indices in observation_dict.items():
            # Get the row corresponding to this ooi and the means and covariances of the OOI corners
            ooi_index = new_object_df.loc[new_object_df['ooi_id'] == ooi_id].index[0] # Index of the OOI in the object dataframe
            means = deepcopy(new_object_df.loc[ooi_index, 'points']) # 4x2 numpy array of corner means
            covs = deepcopy(new_object_df.loc[ooi_index, 'covariances']) # List of 4 2x2 numpy covariance matrices
            
            # Go through the indeces of the OOI points that were observed
            for i in observed_indices:
                # KF update with the observed corner using the previous mean for now
                prev_trace = np.trace(covs[i]) # Get the trace of the covariance matrix pre-update
                new_mean, new_cov = self.skf.update(means[i,:], covs[i], means[i,:], new_car_state)
                trace_delta_sum += prev_trace - np.trace(new_cov) # Add the difference in trace to the sum
                
                # Place the new mean and covariance into the copied means and covs
                means[i,:] = new_mean # The numpy arrays are mutable so this will update the original object_df
                covs[i] = new_cov
                
            # Now place the updated means and covs back into the object dataframe
            new_object_df.at[ooi_index, 'points'] = means
            new_object_df.at[ooi_index, 'covariances'] = covs
        
        # Update the exploration grid based on the new car state
        new_grid, num_explored = self.explore_grid.update(explore_grid, new_car_state)
        
        # Calculate rewards
        obstacle_reward = objects_in_collision_df.shape[0] * self.obstacle_punishment  # Reward for colliding with obstacles
        trace_delta_reward = min_max_normalize(trace_delta_sum, 0, self.init_covariance_trace) # Reward for reducing covariance trace
        explore_reward = self.explored_cell_reward * num_explored # Reward for exploring unexplored cells
        reward = obstacle_reward + trace_delta_reward + explore_reward # Total reward is sum of all rewards
        
        # Print rewards if enabled
        if print_rewards:
            print(f'Obstacle Reward: {obstacle_reward}')
            print(f'Trace Delta Reward: {trace_delta_reward}')
            print(f'Explore Reward: {explore_reward}')
            print(f'Total Reward: {reward}')
        
        # Check if the episode is done
        new_ooi_df = new_object_df[new_object_df['object_type'] == 'ooi']
        total_trace = new_ooi_df['covariances'].apply(lambda matrices: np.sum([np.trace(matrix) for matrix in matrices])).sum()
        done = total_trace < self.final_cov_trace
        
        # Combine the updated car state, mean, covariance and horizon into a new state
        new_state = (new_car_state, new_object_df, new_grid, horizon)
        
        # Return the reward and the new state
        return new_state, reward, done
        
    def check_done(self, state) -> bool:
        """
        Check if the episode is done based on the state.
        
        :param state: (np.ndarray) the state of the car and OOI (position(0:2), corner means(2:10), corner covariances(10:74))
        :return: (bool) whether the episode is done
        """
        # Normalize the trace between 0 and 1 (in this case this just divides by initial variance times the dimensions)
        trace_normalized = min_max_normalize(np.trace(state[2]), 0, self.covariance_trace_init)
                                             
        # Check if the trace of the covariance matrix is below the final threshold
        return trace_normalized < self.final_cov_trace
    
    # Get normlized covariance trace for each point in the corners
    def get_normalized_cov_pt_traces(self, state) -> np.ndarray:
        # Get the diagonals of the covariance matrix for the corners to get trace
        corner_cov_diags = np.diag(state[2])
        
        # Get the trace of every 2 diagonals (to get a per point trace)
        reshaped_cov_diags = corner_cov_diags.reshape(4, 2) # this puts x in first column and y in second
        point_traces = np.mean(reshaped_cov_diags, axis=1) # Sum the x and y covariances for each point to get trace
        
        # Normalize the point traces to [0, covariance_trace_init/4] (this is the max trace for a single point)
        norm_point_traces = min_max_normalize(point_traces, 0, self.covariance_trace_init/4)
        
        return norm_point_traces
        
    # Quick state evaluation based on kdtree for quick distance lookup to corners and obstacles
    def evaluate(self, state, draw=False) -> float:
        # Get the normalized covariance trace for each corner
        norm_point_traces = self.get_normalized_cov_pt_traces(state)
        
        # Evaluate for each action in action space
        prior_reward = np.zeros([self.N], dtype=np.float32)
        for n, action in enumerate(self.action_space):
            # Pass the car state to the KDTree evaluation to get the reward
            prior_reward[n] = self.eval_kd_tree.evaluate(action, state[0], norm_point_traces, state[4], self.discount_factor, draw=draw)
            
        avg_reward = np.mean(prior_reward)
        
        return prior_reward, avg_reward
    
    # # Same repeating action evaluation as in evaluation.py but using full environment step
    # def full_evaluate(self, action, state, depth, draw=False) -> float:
    #     # Use the full environment step to evaluate the reward eval_steps times
    #     cumulative_reward = 0.
    #     for i in range(self.eval_steps):
    #         state, reward, done = self.step(state, action, dt=self.eval_dt)
    #         discounted_reward = reward * (self.discount_factor**(depth+i))
    #         cumulative_reward += discounted_reward
            
    #         # Draw the state if draw is True with size based on reward
    #         if draw:
    #             # self.ui.draw_circle(state[0][:2], 0.1, color='r')
    #             self.ui.draw_text(f'fl: {reward:.2f}', state[0][:2] + np.array([-0.2, 0.6]), color='black', fontsize=12)
    #         #     print(f'i={i} Fl Reward={reward}')
    #         # print()
        
    #     return cumulative_reward
    
    # # Run evaluation with display from starting state for debugging
    # def display_evaluation(self, state, pause_time=0.1, draw=False) -> None:
    #     # Run both evaluations for comparison on each action
    #     test_actions = np.array([[1.0, -1.0], [1.0, -0.5], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    #     # test_actions = np.array([[0., 0.]])
    #     # test_actions = np.array([[-1.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])
        
    #     for a in test_actions:
    #     # for a in self.action_space:
    #         print()
    #         print(f'Action: {a}')
    #         # Run the KDTree evaluation
    #         print('KD TREE EVALUATION')
    #         kd_cumulative_reward = self.evaluate(a, state, 0, draw=draw)
            
    #         # Run the full environment evaluation
    #         print('FULL EVALUATION')
    #         full_cumulative_reward = self.full_evaluate(a, state, 0, draw=draw)
            
    #     if draw:  
    #         # Draw the initial state
    #         self.car.draw_state(state[0])
    #         self.skf.draw_state(state[1], state[2])
    #         self.ooi.draw()
    #         self.eval_kd_tree.draw_obstacles()
            
    #         # Create plot for the UI
    #         self.ui.single_plot()
    
    def draw_state(self, state, title=None, plot_explore_grid=True, plot=True, root_node=None, 
                   rew=None, q_val=None, qu_val=None, scaling=1, bias=0,
                   max=4, get_fig_ax: bool=False):
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
        """
        # Pull elements out of the state
        car_state, object_df, explore_grid, horizon = state
        
        # Draw the car state
        self.car.draw_car_state(car_state)
        
        # Draw the objects in the dataframe
        self.object_manager.draw_objects(car_state, df=object_df)
        
        if plot_explore_grid:
            # Draw the exploration grid
            self.explore_grid.draw_grid(explore_grid)
        
        # Draw the simulated states
        if root_node is not None:
            self.draw_simulated_states(root_node, rew=rew, q_val=q_val, qu_val=qu_val, scaling=scaling, bias=bias, max=max)
        
        if plot:
            return self.ui.plot(get_fig_ax=get_fig_ax)
    
    def draw_simulated_states(self, node, rew=False, q_val=False, qu_val=False, scaling=1, bias=0, max=4) -> None:
        """
        Recursively go through tree of simulated states and draw points of each position sized by the reward
        :param node: (Node) the node to draw the simulated states from
        :param color: (str) the color of the points to draw 
        :param rew: (bool) whether to size based on reward
        :param q_val: (bool) whether to size based on Q value
        :param qu_val: (bool) whether to size based on upper confidence bound
        :param scaling: (float) the scaling factor for the radius of the points
        :param bias: (float) the bias to add to the radius of the points
        """
        if not (rew or q_val or qu_val):
            raise ValueError("Must select at least one of rew, q_val, or qu_val to draw simulated states")
        
        if rew:
            # Rewards are already normalized between 0 and 1, add 0.05 to make all rewards visible
            radius = node.reward
            
        elif q_val:
            # Q values are normalized between 0 and 1, add 0.05 to make all rewards visible
            radius = node.Q
            
        elif qu_val:
            radius = node.Q + node.U
            
        color = 'g' if radius >= 0 else 'r'
        radius = np.abs(radius) * scaling + bias
        radius = np.clip(radius, 0, max)
            
        # Place a point at the state of this node
        self.ui.draw_point(node.state[0][:2], color=color, radius=radius, alpha=0.2)
        
        # Draw the children recursively by calling this function
        for child in node.children.values():
            self.draw_simulated_states(child, rew=rew, q_val=q_val, qu_val=qu_val, scaling=scaling, bias=bias, max=max)

    
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
    
# if __name__ == '__main__':
#     # Create parser
#     parser = argparse.ArgumentParser(description='Run Toy Measurement Control')
    
#     # Add arguments
#     parser.add_argument('--one_iteration', type=bool)
#     parser.add_argument('--display_evaluation', type=bool)
#     parser.add_argument('--time_evaluation', type=bool)
#     parser.add_argument('--no_flask_server', type=bool)
    
#     # Parse arguments
#     args = parser.parse_args()
    
#     # Create an instance of MeasurementControlEnvironment using command line arguments
#     tmc = MeasurementControlEnvironment(one_iteration=args.one_iteration,
#                                 display_evaluation=args.display_evaluation,
#                                 time_evaluation=args.time_evaluation,
#                                 no_flask_server=args.no_flask_server)
#     tmc.run()