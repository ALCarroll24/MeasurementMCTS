import numpy as np
from typing import Tuple
from ui import MatPlotLibUI
from car import Car
from ooi import OOI
from state_evaluation.evaluation import KDTreeEvaluation
from static_kf import StaticKalmanFilter, measurement_model
from utils import wrap_angle, min_max_normalize, get_interpolated_polygon_points, rotate_about_point
from shapely.affinity import rotate
# from mcts.mcts import MCTS
# from mcts.hash import hash_action, hash_state
from mcts.mcts import mcts_search, Environment, parallel_mcts_search
from mcts.tree_viz import render_graph, render_pyvis
from flask_server import FlaskServer
import argparse
from copy import deepcopy
from time import sleep
import timeit

class MeasurementControlEnvironment(Environment):
    def __init__(self, notebook=False, one_iteration=False, display_evaluation=False, time_evaluation=False, no_flask_server=False, enable_ui=True):
        # Flag for whether to run one iteration for profiling
        self.one_iteration = one_iteration
        if self.one_iteration:
            print("Running one iteration for profiling")
            self.draw = False
        else:
            self.draw = True
        
        self.enable_flask_server = not no_flask_server
            
        # Determine update rate
        self.ui_update_rate = 40.0 # Hz (determines pause time for UI update)
        self.simulation_dt = 0.3 # time step size for forward simulation search
        
        # MCTS search parameters
        self.num_processes = 1 # number of processes to use for parallel MCTS search (1 for single threaded)
        self.horizon_length = 500 # length of the planning horizon
        self.learn_iterations = 1000 # number of learning iterations for MCTS (one set of: selection, expansion, evaluation, backpropagation)
        self.exploration_factor = 0.3  # exploration factor for MCTS
        self.discount_factor = 1.0  # discount factor for MCTS (1 means no discounting and 0 only considers immediate rewards)
        self.final_cov_trace = 0.03 # Covariance trace threshold for stopping the episode (normalized from (0, initial trace)-> (0, 1))
        
        # MCTS Action space parameters
        long_acc_options = np.array([-1., -0.5, 0., 0.5, 1.]) # options for longitudinal acceleration (scaled from [-1, 1] to vehicle [-max_acc, max_acc])
        steering_acc_options = np.array([-1., -0.25, 0., 0.25, 1.]) # options for steering acceleration (scaled from [-1, 1] to vehicle [-max_steering_alpha, max_steering_alpha])
        self.all_actions = np.array(np.meshgrid(long_acc_options, steering_acc_options)).T.reshape(-1, 2) # Generate all combinations using the Cartesian product of the two action spaces
        
        # Move the zero action to the front of the list (this is the default action)
        zero_action = np.array([0.0, 0.0])
        zero_action_index = np.where(np.all(self.all_actions == zero_action, axis=1))[0][0]
        self.all_actions = np.concatenate((self.all_actions[zero_action_index:], self.all_actions[:zero_action_index]))
        
        # Simulated car and attached sensor parameters
        initial_car_state = np.array([50.0, 30.0, 10., np.radians(90), 0.0, 0.0])
        # [X (m), Y (m), v_x (m/s), psi (rad), delta (rad), delta_dot (rad/s)] (x pos, y pos, longitudinal velocity, yaw, steering angle, steering angle rate)
        self.min_range = 5. # minimum range for sensor model
        sensor_max_range = 40. # meters
        self.min_bearing = 5. # minimum bearing for sensor model
        sensor_max_bearing = 60. # degrees
        
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
        
        # Create a plotter object unless we are profiling
        title = f'sim step size={self.simulation_dt}, Explore factor={self.exploration_factor}, Horizon={self.horizon_length}, Learn Iterations={self.learn_iterations}\n' + \
                f'Evaluation Steps={self.eval_steps}, Evaluation dt={self.eval_dt}, Obstacle rew min={-obs_rew_norm_min}'
        if enable_ui:
            self.ui = MatPlotLibUI(notebook=notebook, update_rate=self.ui_update_rate, title=title, single_plot=display_evaluation)
        else:
            self.ui = None
        self.ui_was_paused = False # Flag for whether the UI was paused before the last iteration
        
        # Create a car object
        self.car = Car(initial_car_state, max_bearing=sensor_max_bearing, max_range=sensor_max_range, ui=self.ui)
        
        # Create an OOI object
        self.ooi = OOI(ooi_ground_truth_corners, car_max_range=sensor_max_range, car_max_bearing=sensor_max_bearing, ui=self.ui)
        
        # Create a Static Vectorized Kalman Filter object
        self.skf = StaticKalmanFilter(ooi_noisy_corners, ooi_init_covariance, range_dev=range_dev, min_range=self.min_range,
                                      bearing_dev=bearing_dev, min_bearing=self.min_bearing, ui=self.ui)
        
        # Create a KDTree for evaluation TODO: This should be created in each main loop (one real time step)
        self.eval_kd_tree = KDTreeEvaluation([ooi_noisy_corners], num_steps=self.eval_steps, dt=self.eval_dt, max_range=sensor_max_range, 
                                             max_bearing=sensor_max_bearing, obstacle_std_dev=obstacle_std_dev, corner_rew_norm_max=corner_rew_norm_max,
                                             obs_rew_norm_min=obs_rew_norm_min, range_dev=range_dev, bearing_dev=bearing_dev,
                                             min_range=self.min_range, min_bearing=self.min_bearing, ui=self.ui)
        
        # Initial random state bounds (when environment is reset)
        #                                 X,           Y,          v_x,         psi,           delta,       delta_dot
        self.car_state_bounds = np.array([[10., 90.], [10., 90.], [-10., 10.], [-np.pi, np.pi], [-0.5, 0.5], [-0.5, 0.5]])
        self.mean_corners_bounds = np.array([[30., 70.], [30., 70.]]) # Mean of corners (x, y) bounds (corners picked around mean)
        self.length_and_width = np.array([8., 4.]) # Length and width of the OOI
        self.init_corner_std_dev = 0.5 # Standard deviation of gaussian distribution which will be used to sample corners locations
        self.rotate_corner_bounds = np.array([0, 2*np.pi]) # Rotation bounds for corners
        self.trace_init = 64. # Initial trace of covariance matrix

        # Get and set OOI collision polygons
        # self.ooi_poly = self.ooi.get_collision_polygon()
        # self.ooi.soft_collision_polygon = self.ooi_poly.buffer(self.soft_collision_buffer)
        # self.ooi.soft_collision_points = get_interpolated_polygon_points(self.ooi.soft_collision_polygon, num_points=50)
        # self.soft_ooi_poly = self.ooi.soft_collision_polygon
        
        # if true time difference between evaluations (KDTree vs Full environment step evaluation)
        if time_evaluation:
            start_time = timeit.default_timer()
            _ = self.evaluate(self.get_state(), draw=False)
            kd_tree_time_taken = timeit.default_timer() - start_time
            print(f'KDTree Evaluation Time: {kd_tree_time_taken} seconds')
            start_time = timeit.default_timer()
            _ = self.full_evaluate(self.all_actions[0,:], self.get_state(), depth=0, draw=False)
            full_time_taken = timeit.default_timer() - start_time
            print(f'Full Evaluation Time: {full_time_taken} seconds')
            times_faster = full_time_taken / kd_tree_time_taken
            print(f'KDTree Evaluation is {times_faster} times faster than full evaluation')
            exit()
        
        # If we are just plotting the evaluation, do that and exit
        if display_evaluation:
            self.display_evaluation(self.get_state(), draw=True)
            exit()
        
        # Save the last action (mainly used for relative manual control)
        self.last_action = np.array([0.0, 0.0])
        
        # Run flask server which makes web MCTS tree display and communicates clicked nodes
        if self.one_iteration or not self.enable_flask_server:
            self.flask_server = None
        else:
            self.flask_server = FlaskServer()
            self.flask_server.run_thread()
        
        # Initialize variables for clicked node display
        self.last_node_clicked = None
        self.last_node_hash_clicked = None
        
        # Flag for whether simulated state is being drawn (Used for pausing UI and clicking nodes to display simulated state)
        self.drawing_simulated_state = False
        
        # Flag for whether goal has been reached
        self.done = False
        
        print("Toy Measurement Control Initialized")

    @property
    def N(self):
        """ Number of actions in the action space """
        return len(self.all_actions)

    @N.setter
    def N(self, value):
        self.N = value

    def run(self): 
    # Loop until matplotlib window is closed (handled by the UI class)
        while(True):
            
            # Only update controls if the UI is not paused or if we are doing one iteration
            if (self.one_iteration is True) or (not self.ui.paused):
                # Reset the flag for whether the UI was paused before the last iteration
                self.ui_was_paused = False
                
                # Get the observation from the OOI, pass it to the KF for update
                start_time = timeit.default_timer()
                observable_corners, indeces = self.ooi.get_noisy_observation(self.car.get_state(), draw=self.draw)
                print(f'Observation Time: {timeit.default_timer() - start_time}')
                
                # If there are no observable corners, skip the update
                if len(indeces) > 0:
                    self.skf.update(observable_corners, indeces, self.car.get_state())
                
                # Check if we are done (normalized covariance trace is below threshold)
                done = self.check_done(self.get_state())
                if done:
                    print("Goal Reached")
                    self.ui.paused = True
                
                # Print normalized covariance trace compared to final trace
                trace_normalized = min_max_normalize(np.trace(self.skf.P_k), 0, self.covariance_trace_init)
                print(f'Normalized Covariance Trace: {np.round(trace_normalized, 4)} Final: {self.final_cov_trace}')
                
                ############################ AUTONOMOUS CONTROL ############################
                # Create an MCTS object
                # mcts = MCTS(initial_obs=self.get_state(), env=self, K=self.exploration_factor,
                #             _hash_action=hash_action, _hash_state=hash_state,
                #             horizon_length=self.horizon_length, discount_factor=self.discount_factor)                
                # start_time = time.time()
                # mcts.learn(self.learn_iterations, progress_bar=False)
                # print(f'MCTS Time: {time.time() - start_time}')
                
                # # Get the best action from the MCTS tree
                # action_vector = mcts.best_action()
                # print("MCTS Action: ", action_vector)
                
                # Run MCTS search
                if self.num_processes == 1:
                    # Single threaded MCTS search
                    start_time = timeit.default_timer()
                    action_index, root_node = mcts_search(self, self.get_state(), self.learn_iterations)
                    print(f'Single Threaded MCTS Time: {timeit.default_timer() - start_time}')
                else:
                    # Parallel MCTS search
                    start_time = timeit.default_timer()
                    action_index, root_node = parallel_mcts_search(self, self.get_state(), self.learn_iterations,
                                                                   num_processes=self.num_processes)
                    print(f'Parallel MCTS Time: {timeit.default_timer() - start_time}')
                
                # Get the best action from the MCTS tree
                action_vector = self.all_actions[action_index]
                
                # Get the next state by looking through tree based on action    
                # next_state = list(mcts.root.children[hash_action(action_vector)].children.values())[0].state
                
                # Call reward to print useful information about the state from reward function
                # reward, done = self.get_reward(next_state, action_vector, print_rewards=True)
                # print(f'Total Reward: {reward}')
                
                ############################ MANUAL CONTROL ############################
                # Get the control inputs from the arrow keys, pass them to the car for update
                # relative_action_vector = self.car.get_arrow_key_control()
                # action_vector = self.car.add_input(relative_action_vector, self.last_action)   # This adds the control input rather than directly setting it (easier for keyboard control)
                
                # Update the car's state based on the control inputs
                self.car.update(self.simulation_dt, action_vector)
                
                self.drawing_simulated_state = False
                
            else:
                # Only render the MCTS tree if the UI was not paused before the last iteration so that the tree is not rendered multiple times
                if not self.ui_was_paused:
                    render_pyvis(root_node)
                    self.ui_was_paused = True
                    
                # Check if a node has been clicked
                if self.last_node_hash_clicked != self.flask_server.node_clicked:
                    clicked_node = mcts.get_node(self.flask_server.node_clicked)
                    
                    if clicked_node is None:
                        print("Node not found")
                        print("Either refresh (viewing old tree) or click on decision node (not random node)")
                    else:
                        self.drawing_simulated_state = True
                        self.last_node_clicked = clicked_node
                        self.last_node_hash_clicked = self.flask_server.node_clicked
                        print("Node clicked: \n", self.last_node_clicked)
            
            # Draw either simulated state or current state
            if self.drawing_simulated_state:
                # Update displays based on clicked node
                self.car.draw_state(self.last_node_clicked.state[0])
                self.skf.draw_state(self.last_node_clicked.state[1], self.last_node_clicked.state[2])
                self.ooi.draw() # Just draw rectangles same ground truth
                self.eval_kd_tree.draw_obstacles()
            else:
                # Update current state displays
                self.car.draw()
                self.ooi.draw()
                self.skf.draw()
                self.eval_kd_tree.draw_obstacles()
                # self.draw_simulated_states(mcts)
                    
                # If we are only doing one iteration
                if self.one_iteration:
                    # Render the MCTS tree (viewable in browser by opening tree_visualization.html)
                    render_pyvis(root_node)
                    
                    # Create a single plot for the UI
                    self.ui.single_plot()
                    
                    # And exit
                    exit()
            
            # Update the ui to display the new state
            self.ui.update()
            
            # Check for shutdown flag
            if self.ui.shutdown and self.enable_flask_server:
                self.flask_server.stop_flask()
                break
            
    def reset(self) -> tuple:
        """
        Reset the environment to a random state
        
        returns (Tuple[Car, OOI mean corners, Covariance matrix]) the initial state of the environment
        """
        
        # Pick the car state at random within the bounds
        car_state = np.random.uniform(self.car_state_bounds[:,0], self.car_state_bounds[:,1])
        
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
        initial_covariance = np.eye(8) * self.trace_init/8
        
        # Return the initial state
        return (car_state, corner_means.flatten(), initial_covariance, 0)
        
        
            
    def get_state(self, horizon=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        '''
        Returns full state -> Tuple[Car State, Corner Mean, Corner Covariance, Horizon]
        '''
        return self.car.get_state(), self.skf.get_mean(), self.skf.get_covariance(), horizon
    
    def step(self, state, action, dt=None) -> Tuple[float, np.ndarray]:
        """
        Step the environment by one time step. The action is applied to the car, and the state is observed by the OOI.
        The observation is then passed to the KF for update.
        
        :param state: (np.ndarray) the state of the car and KF (Car state(x,y,yaw), corner means, corner covariances)
        :param action: (np.ndarray) the control input to the car (velocity, steering angle)
        :return: (float, np.ndarray) the reward of the state-action pair, and the new state
        """
        # If dt is not specified, use the default period
        if dt is None:
            dt = self.simulation_dt
        
        # Pull out the state elements
        car_state, corner_means, corner_cov, horizon = state
        
        # Increment the horizon
        horizon += 1
        
        # Apply the action to the car
        new_car_state = self.car.update(dt, action, starting_state=car_state)
        
        # Get the observation from the OOI, pass it to the KF for update
        observable_corners, indeces = self.ooi.get_observation(new_car_state, corners=corner_means.reshape(4,2), draw=False)
        new_mean, new_cov = self.skf.update(observable_corners, indeces, new_car_state,
                                            simulate=True, s_k_=corner_means, P_k_=corner_cov)
        
        # Find the reward based prior vs new covariance matrix and car collision with OOI
        reward, done = self.get_reward(new_cov, new_car_state)
        
        # Combine the updated car state, mean, covariance and horizon into a new state
        new_state = (new_car_state, new_mean, new_cov, horizon)
        
        # Return the reward and the new state
        return new_state, reward, done
    
    
    def get_reward(self, new_cov, car_state, print_rewards=False) -> Tuple[float, bool]:
        """
        Get the reward of the new state-action pair.
        
        :param new_state: (np.ndarray) the new state of the car and OOI (position(0:2), corner means(2:10), corner covariances(10:74))
        :param action: (np.ndarray) the control input to the car (velocity, steering angle)
        :return: (float, bool) the reward of the state-action pair, and whether the episode is done
        """
        
        # Normalize the trace between 0 and 1 using the final trace as the min and initial trace as the max
        new_cov_trace = min_max_normalize(np.trace(new_cov), self.final_cov_trace, self.covariance_trace_init)
        
        # Reward is mirrored over 0.5 (1 is now the final covariance and 0 is the initial covariance)
        reward = 1-new_cov_trace
        
        # Print trace for debugging
        if print_rewards:
            print(f'Trace Reward: {reward}')    
        
        # Get obstacle reward from KDTree evaluation (will be [-obs_rew_norm_min, 0])
        obs_reward = self.eval_kd_tree.get_obstacle_cost(car_state)
        # Add the obstacle reward to the trace reward
        reward += obs_reward
        
        # Find whether the episode is done based on the trace
        done = new_cov_trace < self.final_cov_trace
        
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
        for n, action in enumerate(self.all_actions):
            # Pass the car state to the KDTree evaluation to get the reward
            prior_reward[n] = self.eval_kd_tree.evaluate(action, state[0], norm_point_traces, state[3], self.discount_factor, draw=draw)
            
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
        # for a in self.all_actions:
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
    
    def draw_state(self, state) -> None:
        """
        Draw the state on the UI.
        
        :param state: (np.ndarray) the state of the car and OOI (position, corner means, corner covariances)
        """
        # Draw the car state
        self.car.draw_state(state[0])
        
        # Draw the KF state
        self.skf.draw_state(state[1], state[2])
        
        # Draw the obstacles
        self.eval_kd_tree.draw_obstacles()
        
        self.ui.single_plot()
    
    def draw_simulated_states(self, mcts_tree, radius=0.1, color='y'):
        # Recursively go through tree of simulated states and draw them, width is based on average reward
        
        # Get the root node
        root = mcts_tree.root # root decision node
        
        def draw_child_states(decision_node):
            # Place a point at the state of this node
            self.ui.draw_point(decision_node.state[0][:2], color=color, radius=radius)
            
            # Draw the children
            for child_random_node in decision_node.children.values():
                if len(child_random_node.children.values()) > 0:
                    draw_child_states(next(iter(child_random_node.children.values())))
                    
        # Recuriousely draw the children
        draw_child_states(root)
                
    
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