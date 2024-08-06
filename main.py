import numpy as np
from typing import Tuple
from ui import MatPlotLibUI
from car import Car
from ooi import OOI
from static_kf import StaticKalmanFilter, measurement_model
from utils import wrap_angle, min_max_normalize, get_interpolated_polygon_points, rotate_about_point
from shapely.affinity import rotate
from mcts.mcts import MCTS
from mcts.hash import hash_action, hash_state
from mcts.tree_viz import render_graph, render_pyvis
from flask_server import FlaskServer
import argparse
from copy import deepcopy
from time import sleep
import time

class ToyMeasurementControl:
    def __init__(self, one_iteration=False, display_evaluation=False, no_flask_server=False):
        # Flag for whether to run one iteration for profiling
        self.one_iteration = one_iteration
        if self.one_iteration:
            print("Running one iteration for profiling")
            self.draw = False
        else:
            self.draw = True
        
        self.enable_flask_server = not no_flask_server
            
        # Determine update rate
        self.ui_update_rate = 20.0 # Hz
        self.simulation_dt = 0.1 # time step size for forward simulation search
        
        # MCTS search parameters
        self.horizon = 5 # length of the planning horizon
        self.learn_iterations = 50  # number of learning iterations for MCTS
        self.exploration_factor = np.sqrt(2)  # exploration factor for MCTS (Using sqrt(2) as recommended for rewards in [0,1])
        self.discount_factor = 0.9  # discount factor for MCTS
        self.final_cov_trace = 0.03 # Covariance trace threshold for stopping the episode (normalized from (0, initial trace)-> (0, 1))
        
        # MCTS Action space parameters
        long_acc_options = np.array([-1., -0.5, 0, 0.5, 1.]) # options for longitudinal acceleration (scaled from [-1, 1] to vehicle [-max_acc, max_acc])
        steering_acc_options = np.array([-1., -0.5, 0, 0.5, 1.]) # options for steering acceleration (scaled from [-1, 1] to vehicle [-max_steering_alpha, max_steering_alpha])
        self.all_actions = np.array(np.meshgrid(long_acc_options, steering_acc_options)).T.reshape(-1, 2) # Generate all combinations using the Cartesian product of the two action spaces
        
        # Simulated car and attached sensor parameters
        # TODO: Placing the car far away causes it to pick the first action and just spin in a circle
        initial_car_state = np.array([10.0, 20.0, 0.0, 0.0, 0.0, 0.0]) 
        # [X (m), Y (m), v_x (m/s), psi (rad), delta (rad), delta_dot (rad/s)] (x pos, y pos, longitudinal velocity, yaw, steering angle, steering angle rate)
        self.sensor_max_bearing = 60 # degrees
        self.sensor_max_range = 40 # meters
        
        # New evaluation parameters based on base policy
        self.eval_path_buffer = 7.  # buffer around OOI for evaluation path
        self.eval_steps = 10  # number of steps to evaluate the base policy
        self.eval_dt = 0.2  # time step size for evaluation
        self.lookahead_distance = 4.  # distance to look ahead for path following
        eval_path_rotation = 90  # degrees to rotate the evaluation path (This helps the car get more observations of corners)
        
        # Collision parameters
        self.soft_collision_buffer = 2. # buffer for soft collision (length from outline of OOI to new outline for all points)
        self.hard_collision_punishment = 1000  # punishment for hard collision
        self.soft_collision_punishment = 100  # punishment for soft collision
        
        # Kalman Filter parameters parameters
        range_dev = 1. # standard deviation of the range scaling
        bearing_dev = 1. # standard deviation of the bearing scaling
        initial_corner_std_dev = 1.5 # standard deviation of the noise for the OOI corners
        initial_range_std_dev = 0.5  # standard deviation of the noise for the range of the OOI corners
        initial_bearing_std_dev = 0.5 # standard deviation of the noise for the bearing of the OOI corners
        
        # OOI Real location
        ooi_ground_truth_corners = np.array([[54., 52.], [54., 48.], [46., 48.], [46., 52.]]) # Ground truth corners of the OOI
        rotatation_angle = 45 # Rotate simulated OOI by this angle (degrees)
        ooi_ground_truth_corners = rotate_about_point(ooi_ground_truth_corners, np.radians(rotatation_angle), ooi_ground_truth_corners.mean(axis=0))
        
        # Use parameters to initialize noisy corners and calculate initial covariance matrix using measurement model
        ooi_noisy_corners = ooi_ground_truth_corners + np.random.normal(0, initial_corner_std_dev, (4, 2)) # Noisy corners of the OOI
        ooi_init_covariance = measurement_model(ooi_noisy_corners, np.arange(4), initial_car_state[:2], initial_car_state[3], # Initial Covariance matrix for the OOI
                                                range_dev=initial_range_std_dev, bearing_dev=initial_bearing_std_dev)
        self.covariance_trace_init = np.trace(ooi_init_covariance)
        
        # Create a plotter object unless we are profiling
        if self.one_iteration:
            self.ui = None
        else:
            title = f'sim step size={self.simulation_dt}, Explore factor={self.exploration_factor}, Horizon={self.horizon}, Learn Iterations={self.learn_iterations}\n' + \
                    f'Evaluation Steps={self.eval_steps}, Evaluation dt={self.eval_dt}'
            self.ui = MatPlotLibUI(update_rate=self.ui_update_rate, title=title)
            self.ui_was_paused = False # Flag for whether the UI was paused before the last iteration
        
        # Create a car object
        self.car = Car(self.ui, initial_car_state, max_bearing=self.sensor_max_bearing, max_range=self.sensor_max_range)
        
        # Create an OOI object
        self.ooi = OOI(self.ui, ooi_ground_truth_corners, car_max_range=self.car.max_range, car_max_bearing=self.car.max_bearing)
        
        # Create a Static Vectorized Kalman Filter object
        self.skf = StaticKalmanFilter(ooi_noisy_corners, ooi_init_covariance, range_dev=range_dev, bearing_dev=bearing_dev)
        
        # Get and set OOI collision polygons
        self.ooi_poly = self.ooi.get_collision_polygon()
        self.ooi.soft_collision_polygon = self.ooi_poly.buffer(self.soft_collision_buffer)
        self.ooi.soft_collision_points = get_interpolated_polygon_points(self.ooi.soft_collision_polygon, num_points=50)
        self.soft_ooi_poly = self.ooi.soft_collision_polygon
        
        # Get evaluation path for circling around OOI and collecting observations
        eval_poly = self.ooi_poly.buffer(self.eval_path_buffer)
        eval_poly = rotate(eval_poly, eval_path_rotation, origin='centroid')
        self.eval_path = get_interpolated_polygon_points(eval_poly, num_points=200)
        
        # If we are just plotting the evaluation, do that and exit
        if display_evaluation:
            self.display_evaluation(self.get_state(), pause_time=0.1)
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


    def run(self): 
    # Loop until matplotlib window is closed (handled by the UI class)
        while(True):
            
            # Only update controls if the UI is not paused or if we are doing one iteration
            if (self.one_iteration is True) or (not self.ui.paused):
                # Reset the flag for whether the UI was paused before the last iteration
                self.ui_was_paused = False
                
                # Get the observation from the OOI, pass it to the KF for update
                observable_corners, indeces = self.ooi.get_noisy_observation(self.car.get_state(), draw=self.draw)
                
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
                mcts = MCTS(initial_obs=self.get_state(), env=self, K=self.exploration_factor,
                            _hash_action=hash_action, _hash_state=hash_state,
                            discount_factor=self.discount_factor)
                mcts.learn(self.learn_iterations, progress_bar=False)
                
                # If we are doing one iteration for profiling, exit
                if self.one_iteration:
                    render_pyvis(mcts.root)
                    exit()
                    
                # Get the best action from the MCTS tree
                action_vector = mcts.best_action()
                print("MCTS Action: ", action_vector)
                
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
                    render_pyvis(mcts.root)
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
                self.skf.draw_state(self.last_node_clicked.state[1], self.last_node_clicked.state[2], self.ui)
                self.ooi.draw() # Just draw rectangles same ground truth
            else:
                # Update current state displays
                self.car.draw()
                self.ooi.draw()
                self.skf.draw(self.ui)
            
            # Update the ui to display the new state
            self.ui.update()
            
            # Check for shutdown flag
            if self.ui.shutdown and self.enable_flask_server:
                self.flask_server.stop_flask()
                break
            
    def get_state(self, horizon=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        '''
        Returns full state -> Tuple[Car State, Corner Mean, Corner Covariance, Horizon]
        '''
        return self.car.get_state(), self.skf.get_mean(), self.skf.get_covariance(), horizon
    
    def step(self, state, action, dt=None, done_by_horizon=True) -> Tuple[float, np.ndarray]:
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
        
        # Combine the updated car state, mean, covariance and horizon into a new state
        new_state = (new_car_state, new_mean, new_cov, horizon)
        
        # Find the reward based prior vs new covariance matrix and car collision with OOI
        reward, done = self.get_reward(state[2], new_state[2], car_state)
        
        # Check if we are at the end of the horizon
        if done_by_horizon and not done and horizon == self.horizon:
            done = True
        
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
    
    def get_reward(self, prior_cov, new_cov, car_state, print_rewards=False) -> Tuple[float, bool]:
        """
        Get the reward of the new state-action pair.
        
        :param new_state: (np.ndarray) the new state of the car and OOI (position(0:2), corner means(2:10), corner covariances(10:74))
        :param action: (np.ndarray) the control input to the car (velocity, steering angle)
        :return: (float, bool) the reward of the state-action pair, and whether the episode is done
        """
        
        # Normalize the trace between 0 and 1 (in this case this just divides by initial variance times the dimensions)
        prior_cov_trace = min_max_normalize(np.trace(prior_cov), 0, self.covariance_trace_init)
        new_cov_trace = min_max_normalize(np.trace(new_cov), 0, self.covariance_trace_init)
        
        # Reward is the difference in trace (higher difference is better)
        # This is the amount of information gained by the KF (by covariance getting smaller)
        reward = prior_cov_trace - new_cov_trace
        
        # Print trace for debugging
        if print_rewards:
            print(f'Trace Reward: {reward}')    
        
        # Remove large reward when car enters hard or soft boundary
        car_poly = self.car.get_collision_polygon(car_state)
        
        # If car is in collision with OOI, give a large negative reward
        if car_poly.overlaps(self.ooi_poly):
            reward -= self.hard_collision_punishment
            
            if print_rewards:
                print("Hard Collision")
            
        # If car comes very close to OOI (soft collision), give a less large negative reward based on distance inside soft boundary
        if car_poly.overlaps(self.soft_ooi_poly):
            # Find the car polygon's distance to the hard boundary
            dist_to_ooi = car_poly.distance(self.ooi_poly)
            
            # Find the distance inside the soft boundary
            dist_in_soft = self.soft_collision_buffer - dist_to_ooi
            
            # Reward is the percentage of the buffer distance inside the soft boundary squared
            reward -= self.soft_collision_punishment * (dist_in_soft / self.soft_collision_buffer)**2
            
            if print_rewards:
                print("Soft Collision")
        
        # Find whether the episode is done TODO: done needs to also account for horizon length
        done = new_cov_trace < self.final_cov_trace
        
        return reward, done

    # New base policy based evaluation function
    def evaluate(self, state, draw=False) -> float:
        cumulative_reward = 0.
        for i in range(self.eval_steps):
            # Get the action from the base policy
            action = self.car.get_action_follow_path(self.eval_path, self.lookahead_distance, simulate=True, starting_state=state[0])
            
            # Step the environment
            state, reward, done = self.step(state, action, dt=self.eval_dt, done_by_horizon=False)
            
            # Calculate and add discounted reward based on depth (steps in horizon) of evaluation
            cumulative_reward += self.discount_factor**state[3] * reward
            
            # If we are done, break
            if done:
                return cumulative_reward
            
        return cumulative_reward
    
    # Run evaluation with display from starting state for debugging
    def display_evaluation(self, state, pause_time=0.1) -> None:
        # Loop through the evaluation steps
        cumulative_reward = 0.
        for i in range(self.eval_steps):
            # Get the action from the base policy
            action, target_point= self.car.get_action_follow_path(self.eval_path, self.lookahead_distance, simulate=True, starting_state=state[0], return_target_point=True)
            print(f'car state: {state[0]}')
            print(f'action: {action}')
            
            # Step the environment
            state, reward, done = self.step(state, action, dt=self.eval_dt)
            print(f'Reward: {reward}')
            cumulative_reward += reward
            
            # Draw the state
            self.car.draw_state(state[0])
            self.skf.draw_state(state[1], state[2], self.ui)
            self.ooi.draw()
            
            # Draw the target point
            self.ui.draw_point(target_point, color='r')
            
            # Draw the evaluation path
            self.ui.draw_polygon(self.eval_path, color='g')
            
            # Update the UI
            self.ui.update()
            
            # Pause for a specified amount of time
            sleep(pause_time)
        
        print(f'Total Cumulative Reward: {cumulative_reward}')
    
    
if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(description='Run Toy Measurement Control')
    
    # Add arguments
    parser.add_argument('--one_iteration', type=bool)
    parser.add_argument('--display_evaluation', type=bool)
    parser.add_argument('--no_flask_server', type=bool)
    
    # Parse arguments
    args = parser.parse_args()
    print(args)
    
    # Create an instance of ToyMeasurementControl using command line arguments
    tmc = ToyMeasurementControl(one_iteration=args.one_iteration,
                                display_evaluation=args.display_evaluation,
                                no_flask_server=args.no_flask_server)
    tmc.run()