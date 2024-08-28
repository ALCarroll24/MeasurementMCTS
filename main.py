import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from ui import MatPlotLibUI
from car import Car
from ooi import OOI
from state_evaluation.evaluation import KDTreeEvaluation
from static_kf import StaticKalmanFilter, measurement_model
from utils import wrap_angle, min_max_normalize, get_interpolated_polygon_points, rotate_about_point
# from shapely.affinity import rotate
from mcts.mcts import Environment
from exploration_grid import ExplorationGrid
from flask_server import FlaskServer
import argparse
from copy import deepcopy
from time import sleep
import timeit
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

class MeasurementControlEnvironment(Environment):
    def __init__(self):
        # Time step for simulations
        self.simulation_dt = 0.3 # time step size for forward simulation search
        
        # MCTS search parameters
        self.horizon_length = 500 # length of the planning horizon
        self.final_cov_trace = 0.03 # Covariance trace threshold for stopping the episode (normalized from (0, initial trace)-> (0, 1))
        
        # MCTS Action space parameters
        long_acc_options = np.array([-1., -0.5, 0., 0.5, 1.]) # options for longitudinal acceleration (scaled from [-1, 1] to vehicle [-max_acc, max_acc])
        steering_acc_options = np.array([-1., -0.25, 0., 0.25, 1.]) # options for steering acceleration (scaled from [-1, 1] to vehicle [-max_steering_alpha, max_steering_alpha])
        self.action_space = np.array(np.meshgrid(long_acc_options, steering_acc_options)).T.reshape(-1, 2) # Generate all combinations using the Cartesian product of the two action spaces
        
        # Move the zero action to the front of the list (this is the default action)
        zero_action = np.array([0.0, 0.0])
        zero_action_index = np.where(np.all(self.action_space == zero_action, axis=1))[0][0]
        self.action_space = np.concatenate((self.action_space[zero_action_index:], self.action_space[:zero_action_index]))
        
        # Simulated car and attached sensor parameters
        initial_car_state = np.array([30.0, 20.0, 10., np.radians(45), 0.0, 0.0])
        # [X (m), Y (m), v_x (m/s), psi (rad), delta (rad), delta_dot (rad/s)] (x pos, y pos, longitudinal velocity, yaw, steering angle, steering angle rate)
        self.min_range = 5. # minimum range for sensor model
        sensor_max_range = 40. # meters
        self.min_bearing = 5. # minimum bearing for sensor model
        sensor_max_bearing = np.radians(60) # degrees
        
        # Evaluation parameters based on KDtree to quickly compute distance to corners and obstacles
        self.eval_steps = 3  # number of steps to evaluate the base policy
        self.eval_dt = 0.5  # time step size for evaluation
        obstacle_std_dev = 3. # Number of standard deviations to inflate obstacle circle around OOI corners
        corner_rew_norm_max = 0.1 # High normalization value map corner rewards to [0, corner_rew_norm_max] (make this smaller than the rewards on [0, 1] because exploration term balances this)
        obs_rew_norm_min = 12. # Amount to scale the obstacle punishment by (relative to rewards on [0, 1]) rewards in [-obs_rew_norm_min, 0]
        
        # Collision parameters
        self.soft_collision_buffer = 2. # buffer for soft collision (length from outline of OOI to new outline for all points)
        self.hard_collision_punishment = 1000  # punishment for hard collision
        self.soft_collision_punishment = 100  # punishment for soft collision
        
        # Kalman Filter parameters parameters
        initial_corner_std_dev = 0. # standard deviation for the noise between ground truth and first observation of corners
        range_dev = 1. # standard deviation for the range scaling
        bearing_dev = 0.5 # standard deviation for the bearing scaling
        initial_range_std_dev = 0.5  # standard deviation for the noise for the range of the OOI corners
        initial_bearing_std_dev = 0.5 # standard deviation for the noise for the bearing of the OOI corners
        
        # OOI Real location
        ooi_ground_truth_corners = np.array([[54., 52.], [54., 48.], [46., 48.], [46., 52.]]) # Ground truth corners of the OOI
        rotatation_angle = 45 # Rotate simulated OOI by this angle (degrees)
        ooi_ground_truth_corners = rotate_about_point(ooi_ground_truth_corners, np.radians(rotatation_angle), ooi_ground_truth_corners.mean(axis=0))
        
        # Use parameters to initialize noisy corners and calculate initial covariance matrix using measurement model
        ooi_noisy_corners = ooi_ground_truth_corners + np.random.normal(0, initial_corner_std_dev, (4, 2)) # Noisy corners of the OOI
        ooi_init_covariance = measurement_model(ooi_noisy_corners, np.arange(4), initial_car_state[:2], initial_car_state[3], # Initial Covariance matrix for the OOI
                                                min_range=self.min_range, min_bearing=self.min_bearing, range_dev=initial_range_std_dev, bearing_dev=initial_bearing_std_dev)
        self.covariance_trace_init = np.trace(ooi_init_covariance)
        
        # Exploration grid parameters
        padding = np.array([15,15]) # padding to add to the grid for x and y
        meters_per_pixel = 1 # meters per pixel of the grid
        self.explored_cell_reward = 0.0015 # reward for exploring a cell
        explore_grid_bounds = np.array([[5, 95], [5, 95]]) # bounds of the grid
                
        # Initial random state bounds (when environment is reset)
        #                                 X,           Y,          v_x,         psi,           delta,       delta_dot
        self.car_state_bounds = np.array([[10., 90.], [10., 90.], [-10., 10.], [-np.pi, np.pi], [-0.5, 0.5], [-0.5, 0.5]])
        self.mean_corners_bounds = np.array([[30., 70.], [30., 70.]]) # Mean of corners (x, y) bounds (corners picked around mean)
        self.length_and_width = np.array([8., 4.]) # Length and width of the OOI
        self.init_corner_std_dev = 0.5 # Standard deviation of gaussian distribution which will be used to sample corners locations
        self.rotate_corner_bounds = np.array([0, 2*np.pi]) # Rotation bounds for corners
        self.reset_trace_init = 64. # Initial trace of covariance matrix
        
        self.ui = MatPlotLibUI()
        
        # Create a car object
        self.car = Car(initial_car_state, max_bearing=sensor_max_bearing, max_range=sensor_max_range, ui=self.ui)
        
        # Create an OOI object
        self.ooi = OOI(ooi_ground_truth_corners, car_max_range=sensor_max_range, car_max_bearing=sensor_max_bearing, ui=self.ui)
        
        # Create a Static Vectorized Kalman Filter object
        self.skf = StaticKalmanFilter(ooi_noisy_corners, ooi_init_covariance, range_dev=range_dev, min_range=self.min_range,
                                      bearing_dev=bearing_dev, min_bearing=self.min_bearing, ui=self.ui)
        
        # Exploration grid which gives rewards for exploring the environment
        self.explore_grid = ExplorationGrid(explore_grid_bounds, meters_per_pixel, sensor_max_range, sensor_max_bearing, ui=self.ui)
        
        # Create a KDTree for evaluation TODO: This should be created in each main loop (one real time step)
        self.eval_kd_tree = KDTreeEvaluation([ooi_noisy_corners], num_steps=self.eval_steps, dt=self.eval_dt, max_range=sensor_max_range, 
                                             max_bearing=sensor_max_bearing, obstacle_std_dev=obstacle_std_dev, corner_rew_norm_max=corner_rew_norm_max,
                                             obs_rew_norm_min=obs_rew_norm_min, range_dev=range_dev, bearing_dev=bearing_dev,
                                             min_range=self.min_range, min_bearing=self.min_bearing, ui=self.ui)
        
        # Get and set OOI collision polygons
        # self.ooi_poly = self.ooi.get_collision_polygon()
        # self.ooi.soft_collision_polygon = self.ooi_poly.buffer(self.soft_collision_buffer)
        # self.ooi.soft_collision_points = get_interpolated_polygon_points(self.ooi.soft_collision_polygon, num_points=50)
        # self.soft_ooi_poly = self.ooi.soft_collision_polygon
        
        # Flag for whether goal has been reached
        self.done = False
        
        print("Toy Measurement Control Initialized")

    @property
    def N(self):
        """ Number of actions in the action space """
        return len(self.action_space)

    @N.setter
    def N(self, value):
        self.N = value

    def update(self, action):
        # Update the car's state based on the control inputs
        self.car.update(self.simulation_dt, action)
        
        observable_corners, indeces = self.ooi.get_noisy_observation(self.car.get_state(), draw=self.draw)

        # If there are no observable corners, skip the update
        if len(indeces) > 0:
            self.skf.update(observable_corners, indeces, self.car.get_state())
        
        # Check if we are done (normalized covariance trace is below threshold)
        done = self.check_done(self.get_state())
        if done:
            print("Goal Reached")
        
        # Print normalized covariance trace compared to final trace
        trace_normalized = min_max_normalize(np.trace(self.skf.P_k), 0, self.covariance_trace_init)
        print(f'Normalized Covariance Trace: {np.round(trace_normalized, 4)} Final: {self.final_cov_trace}')
        
    def reset(self) -> tuple:
        """
        Reset the environment to a random state
        
        returns (Tuple[Car, OOI mean corners, Covariance matrix]) the initial state of the environment
        """
        
        # Pick the car state at random within the bounds
        car_pos = np.random.uniform(self.car_state_bounds[:,0], self.car_state_bounds[:,1])
        car_yaw = np.random.uniform(-np.pi, np.pi)
        car_state = np.array([car_pos[0], car_pos[1], 0., car_yaw, 0.0, 0.0])
        
        # Pick the mean of the corners from the bounds
        center_mean_corners = np.random.uniform(self.mean_corners_bounds[:,0], self.mean_corners_bounds[:,1])
        x, y = center_mean_corners
        
        # Use the length and width to get the corners of the OOI and sample the corners from a gaussian distribution
        width, length = self.length_and_width
        perfect_corner_means = np.array([[x - width/2, y - length/2], # Bottom left corner
                                         [x + width/2, y - length/2], # Top left corner
                                         [x + width/2, y + length/2], # Top right corner
                                         [x - width/2, y + length/2]]) # Bottom right corner
        
        # Sample gaussian distributions for each corner around the mean to get initial corner means
        corner_means = np.random.normal(perfect_corner_means, self.init_corner_std_dev, (4, 2))
        
        # Update KD tree with new corners
        self.eval_kd_tree.create_kd_tree([corner_means])
        
        # Create initial covariance using initial trace
        initial_covariance = np.eye(8) * self.reset_trace_init/8
        
        # Use the initial grid to get the initial grid state
        grid = self.explore_grid.initial_grid
        
        # Return the initial state
        return (car_state, corner_means.flatten(), initial_covariance, grid, 0)
        
    def get_state(self, horizon=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        '''
        Returns full state -> Tuple[Car State, Corner Mean, Corner Covariance, Horizon]
        '''
        return self.car.get_state(), self.skf.get_mean(), self.skf.get_covariance(), self.explore_grid.get_state(), horizon
    
    def step(self, state, action, dt=None) -> Tuple[tuple, float, bool]:
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
        car_state, corner_means, corner_cov, grid, horizon = state
        
        # Increment the horizon
        horizon += 1
        
        # Apply the action to the car
        new_car_state = self.car.update(dt, action, starting_state=car_state)
        
        # Get the observation from the OOI, pass it to the KF for update
        observable_corners, indeces = self.ooi.get_observation(new_car_state, corners=corner_means.reshape(4,2), draw=False)
        new_mean, new_cov = self.skf.update(observable_corners, indeces, new_car_state,
                                            simulate=True, s_k_=corner_means, P_k_=corner_cov)
        
        # Update the exploration grid with the new state
        new_grid, num_explored = self.explore_grid.update(grid, new_car_state)
        
        # Find the reward based prior vs new covariance matrix and car collision with OOI
        reward, done = self.get_reward(corner_cov, new_cov, new_car_state, num_explored)
        
        # Combine the updated car state, mean, covariance and horizon into a new state
        new_state = (new_car_state, new_mean, new_cov, new_grid, horizon)
        
        # Return the reward and the new state
        return new_state, reward, done
    
    
    def get_reward(self, cov, new_cov, car_state, num_explored, print_rewards=False) -> Tuple[float, bool]:
        """
        Get the reward of the new state-action pair.
        
        :cov: (np.ndarray) the covariance matrix of the corners
        :new_cov: (np.ndarray) the new covariance matrix of the corners
        :car_state: (np.ndarray) the state of the car
        :num_explored: (int) the number of points in the field of view and range of the car
        :return: (float, bool) the reward of the state-action pair, and whether the episode is done
        """
        
        # Normalize the traces between 0 and 1 using the final trace as the min and initial trace as the max
        cov_trace = min_max_normalize(np.trace(cov), self.final_cov_trace, self.covariance_trace_init)
        new_cov_trace = min_max_normalize(np.trace(new_cov), self.final_cov_trace, self.covariance_trace_init)
        
        
        # Reward is how much the trace has decreased (higher is better)
        reward = cov_trace - new_cov_trace
        
        # Print trace for debugging
        if print_rewards:
            print(f'Trace Reward: {reward}')    
        
        # Get obstacle reward from KDTree evaluation (will be [-obs_rew_norm_min, 0])
        obs_reward = self.eval_kd_tree.get_obstacle_cost(car_state)
        # Add the obstacle reward to the trace reward
        reward += obs_reward
        
        reward += self.explored_cell_reward * num_explored
        
        # Find whether the episode is done based on the trace
        done = new_cov_trace < self.final_cov_trace
        
        # If done then add a 1 reward to the final state
        if done:
            reward += 1
        
        return reward, done
        
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
    
    # Same repeating action evaluation as in evaluation.py but using full environment step
    def full_evaluate(self, action, state, depth, draw=False) -> float:
        # Use the full environment step to evaluate the reward eval_steps times
        cumulative_reward = 0.
        for i in range(self.eval_steps):
            state, reward, done = self.step(state, action, dt=self.eval_dt)
            discounted_reward = reward * (self.discount_factor**(depth+i))
            cumulative_reward += discounted_reward
            
            # Draw the state if draw is True with size based on reward
            if draw:
                # self.ui.draw_circle(state[0][:2], 0.1, color='r')
                self.ui.draw_text(f'fl: {reward:.2f}', state[0][:2] + np.array([-0.2, 0.6]), color='black', fontsize=12)
            #     print(f'i={i} Fl Reward={reward}')
            # print()
        
        return cumulative_reward
    
    # Run evaluation with display from starting state for debugging
    def display_evaluation(self, state, pause_time=0.1, draw=False) -> None:
        # Run both evaluations for comparison on each action
        test_actions = np.array([[1.0, -1.0], [1.0, -0.5], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        # test_actions = np.array([[0., 0.]])
        # test_actions = np.array([[-1.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])
        
        for a in test_actions:
        # for a in self.action_space:
            print()
            print(f'Action: {a}')
            # Run the KDTree evaluation
            print('KD TREE EVALUATION')
            kd_cumulative_reward = self.evaluate(a, state, 0, draw=draw)
            
            # Run the full environment evaluation
            print('FULL EVALUATION')
            full_cumulative_reward = self.full_evaluate(a, state, 0, draw=draw)
            
        if draw:  
            # Draw the initial state
            self.car.draw_state(state[0])
            self.skf.draw_state(state[1], state[2])
            self.ooi.draw()
            self.eval_kd_tree.draw_obstacles()
            
            # Create plot for the UI
            self.ui.single_plot()
    
    def draw_state(self, state, title=None, explore_grid=True, plot=True, root_node=None, 
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
        # Draw the car state
        self.car.draw_state(state[0])
        
        # Draw the KF state
        self.skf.draw_state(state[1], state[2])
        
        # Draw the obstacles
        self.eval_kd_tree.draw_obstacles()
        
        if explore_grid:
            # Draw the exploration grid
            self.explore_grid.draw_grid(state[3])
        
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

    
    def draw_action_set(self, state, action_set):
        """
        Use matplotlib animate to create a video with the normal state display over time with actions
        params: state - initial state of the environment
                action_set - list of actions to take in the environment
        """
        # Function called by matplotlib animate to get a frame of the video
        def animate(i):
            # Use the state and axis from the parent function
            nonlocal state
            nonlocal ax
            nonlocal last_index
            
            # Clear all existing patches from the axis
            for patch in ax.patches:
                patch.remove()
            
            # Get the action
            action = action_set[i]
            
            # Draw the state create artists in UI class
            self.draw_state(state, plot=False)
            
            artists = self.ui.get_artists()
            
            for artist in artists:
                ax.add_patch(artist)
            
            # Update state for next iteration if it hasn't already been called
            if last_index != i:
                state, reward, done = self.step(state, action)
            last_index = i
            
            return ax.patches
        
        # Get the figure and axis from the UI
        fig, ax = self.ui.plot(get_fig_ax=True)
        plt.close()
        
        # Track the last index to avoid desyncing from the action set when matplotlib calls the same frame multiple times
        last_index = -1
        
        ani = FuncAnimation(fig, animate, frames=len(action_set), interval=200, blit=False)
            
        # Display the animation in the notebook
        display(HTML(ani.to_jshtml()))
    
if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(description='Run Toy Measurement Control')
    
    # Add arguments
    parser.add_argument('--one_iteration', type=bool)
    parser.add_argument('--display_evaluation', type=bool)
    parser.add_argument('--time_evaluation', type=bool)
    parser.add_argument('--no_flask_server', type=bool)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create an instance of MeasurementControlEnvironment using command line arguments
    tmc = MeasurementControlEnvironment(one_iteration=args.one_iteration,
                                display_evaluation=args.display_evaluation,
                                time_evaluation=args.time_evaluation,
                                no_flask_server=args.no_flask_server)
    tmc.run()