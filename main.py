import numpy as np
from typing import Tuple
from ui import MatPlotLibUI
from car import Car
from ooi import OOI
from vkf import VectorizedStaticKalmanFilter
from utils import wrap_angle, min_max_normalize, angle_difference
from shapely import Point, LineString, Polygon
from mcts.mcts import MCTS
from mcts.hash import hash_action, hash_state
from mcts.tree_viz import render_graph, render_pyvis
from flask_server import FlaskServer
import time
import argparse
import pickle
from copy import deepcopy

class ToyMeasurementControl:
    def __init__(self, one_iteration=False):
        # Flag for whether to run one iteration for profiling
        self.one_iteration = one_iteration
        if self.one_iteration:
            print("Running one iteration for profiling")
            self.draw = False
        else:
            self.draw = True
            
        # Determine update rate
        self.hz = 20.0
        self.period = 1.0 / self.hz
        
        # Parameters
        self.final_cov_trace = 0.1
        self.velocity_options = 3  # number of discrete options for velocity
        self.steering_angle_options = 3  # number of discrete options for steering angle
        self.reverse_option = False # whether to include a reverse option in the action space
        self.reverse_speed = 5.0  # speed for reverse option
        
        self.action_space_sample_heuristic = 'uniform_discrete'
        self.horizon = 5 # length of the planning horizon
        self.expansion_branch_factor = -1  # number of branches when expanding a node (at least two, -1 for all possible actions)
        self.learn_iterations = 100  # number of learning iterations for MCTS
        self.exploration_factor = 0  # exploration factor for MCTS
        
        self.eval_dist_scale = 0.2      # for evaluation, the weight of the distance error
        self.eval_bearing_scale = 0.8   # for evaluation, the weight of the heading error
        self.eval_ocl_turn_scale = 0.  # for evaluation, the weight of the occlusion turn steps
        self.eval_ocl_dist_scale = 0.  # for evaluation, the weight of the occlusion distance steps
        self.evaluation_multiplier = 0.  # multiplier for evaluation function
        
        self.soft_collision_buffer = 4.0  # buffer for soft collision (length from outline of OOI to new outline for all points)
        self.hard_collision_punishment = 1e8  # punishment for hard collision
        self.soft_collision_punishment = 1e8  # punishment for soft collision

        # Create a plotter object unless we are profiling
        if self.one_iteration:
            self.ui = None
        else:
            self.ui = MatPlotLibUI(update_rate=self.hz)
            self.ui_was_paused = False # Flag for whether the UI was paused before the last iteration
        
        # Create a car object
        self.car = Car(self.ui, np.array([30.0, 60.0]), 90)
        
        # Create an OOI object
        self.ooi = OOI(self.ui, position=(50,50), car_max_range=self.car.max_range, car_max_bearing=self.car.max_bearing)
        
        # Get and set OOI collision polygons
        self.ooi_poly = self.ooi.get_collision_polygon()
        self.ooi.soft_collision_polygon = self.ooi_poly.buffer(self.soft_collision_buffer)
        self.soft_ooi_poly = self.ooi.soft_collision_polygon

        # Create a Static Vectorized Kalman Filter object
        # self.vkf = VectorizedStaticKalmanFilter(np.array([50.0]*8), np.diag([8.0]*8), 4.0)
        self.vkf = VectorizedStaticKalmanFilter(np.array(self.ooi.get_corners()).flatten(), np.diag([8.0]*8), 1.0)
        # self.vkf = VectorizedStaticKalmanFilter(np.array(self.ooi.get_corners()).flatten(), np.diag([8.0, 8.0, 0.001, 0.001, 0.001, 0.001, 8.0, 8.0]), 4.0)
        
        # Compute all possible actions
        self.all_actions = self.get_all_actions()
        
        # Save the last action (mainly used for relative manual control)
        self.last_action = np.array([0.0, 0.0])
        
        # Run flask server which makes web MCTS tree display and communicates clicked nodes
        if self.one_iteration:
            self.flask_server = None
        else:
            self.flask_server = FlaskServer()
            self.flask_server.run_thread()
            
        self.last_node_clicked = None
        self.last_node_hash_clicked = None
        
        # Flag for whether simulated state is being drawn
        self.drawing_simulated_state = False


    def run(self): 
    # Loop until matplotlib window is closed (handled by the UI class)
        while(True):
            
            # Only update controls if the UI is not paused or if we are doing one iteration
            if (self.one_iteration is True) or (not self.ui.paused):
                # Reset the flag for whether the UI was paused before the last iteration
                self.ui_was_paused = False
                
                # Get the observation from the OOI, pass it to the KF for update
                observable_corners, indeces = self.ooi.get_noisy_observation(self.car.get_state(), draw=self.draw)
                self.vkf.update(observable_corners, indeces, self.car.get_state())
                
                ############################ AUTONOMOUS CONTROL ############################
                # Create an MCTS object
                # print("NEW MCTS ITERATION-------------------")
                # print()
                # print("Initial State: ", self.get_state())
                mcts = MCTS(initial_obs=self.get_state(), env=self, K=self.exploration_factor,
                            _hash_action=hash_action, _hash_state=hash_state,
                            expansion_branch_factor=self.expansion_branch_factor)
                mcts.learn(self.learn_iterations, progress_bar=False)
                
                # If we are doing one iteration for profiling, exit
                if self.one_iteration:
                    render_pyvis(mcts.root)
                    exit()
                    
                # Get the best action from the MCTS tree
                action_vector = mcts.best_action()
                print("MCTS Action: ", action_vector)
                
                # Get the next state by looking through tree based on action    
                next_state = list(mcts.root.children[hash_action(action_vector)].children.values())[0].state
                
                # Call reward to print useful information about the state from reward function
                reward, done = self.get_reward(next_state, action_vector, print_rewards=True)
                print(f'Total Reward: {reward}')
                
                ############################ MANUAL CONTROL ############################
                # Get the control inputs from the arrow keys, pass them to the car for update
                # relative_action_vector = self.car.get_arrow_key_control()
                # action_vector = self.car.add_input(relative_action_vector, self.last_action)   # This adds the control input rather than directly setting it (easier for keyboard control)
                
                # Update the car's state based on the control inputs
                self.car.update(self.period, action_vector)
                
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
                self.vkf.draw_state(self.last_node_clicked.state[1], self.last_node_clicked.state[2], self.ui)
                self.ooi.draw() # Just draw rectangles for now
            else:
                # Update current state displays
                self.car.draw()
                self.ooi.draw()
                self.vkf.draw(self.ui)
            
            # Update the ui to display the new state
            self.ui.update()
            
            # Check for shutdown flag
            if self.ui.shutdown:
                self.flask_server.stop_flask()
                break
            
    def get_state(self, horizon=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        '''
        Returns full state -> Tuple[Car State, Corner Mean, Corner Covariance, Horizon]
        '''
        return self.car.get_state(), self.vkf.get_mean(), self.vkf.get_covariance(), horizon
    
    def step(self, state, action) -> Tuple[float, np.ndarray]:
        """
        Step the environment by one time step. The action is applied to the car, and the state is observed by the OOI.
        The observation is then passed to the KF for update.
        
        :param state: (np.ndarray) the state of the car and KF (Car state(x,y,yaw), corner means, corner covariances)
        :param action: (np.ndarray) the control input to the car (velocity, steering angle)
        :return: (float, np.ndarray) the reward of the state-action pair, and the new state
        """
        # print("Starting forward simulation state: ", state)
        
        # Pull out the state elements
        car_state, corner_means, corner_cov, horizon = state
        
        # Increment the horizon
        horizon += 1
        
        # Apply the action to the car
        new_car_state = self.car.update(self.period, action, simulate=True, starting_state=car_state)
        
        # Get the observation from the OOI, pass it to the KF for update
        observable_corners, indeces = self.ooi.get_observation(new_car_state, draw=False) # TODO: In forward simulation, observations should come from mean of KF
        new_mean, new_cov = self.vkf.update(observable_corners, indeces, new_car_state, 
                                            simulate=True, s_k_=corner_means, P_k_=corner_cov)
        
        # Combine the updated car state, mean, covariance and horizon into a new state
        new_state = (new_car_state, new_mean, new_cov, horizon)
        
        # Find the reward based on updated state
        reward, done = self.get_reward(new_state, action)
        
        # Check if we are at the end of the horizon
        if not done and horizon == self.horizon:
            done = True
        
        # Return the reward and the new state
        return new_state, reward, done
    
    def get_reward(self, state, action, print_rewards=False) -> Tuple[float, bool]:
        """
        Get the reward of the new state-action pair.
        
        :param new_state: (np.ndarray) the new state of the car and OOI (position(0:2), corner means(2:10), corner covariances(10:74))
        :param action: (np.ndarray) the control input to the car (velocity, steering angle)
        :return: (float, bool) the reward of the state-action pair, and whether the episode is done
        """
        # Pull out the state elements
        car_state, corner_means, corner_cov, horizon = state
        
        # Find the sum of the diagonals
        trace = np.trace(corner_cov)
        
        # Find whether the episode is done TODO: done needs to also account for horizon length
        done = trace < self.final_cov_trace

        # Find the reward
        reward = -trace
        
        # Print trace for debugging
        if print_rewards:
            print(f'Trace Reward: {reward}')    
        
        # Remove large reward when car enters hard or soft boundary
        car_poly = self.car.get_car_polygon(car_state)
        
        # If car is in collision with OOI, give a large negative reward
        if car_poly.overlaps(self.ooi_poly):
            reward -= self.hard_collision_punishment
            
            if print_rewards:
                print("Hard Collision")
            
        # If car comes very close to OOI (soft collision), give a less large negative reward
        if car_poly.overlaps(self.soft_ooi_poly):
            reward -= self.soft_collision_punishment
            
            if print_rewards:
                print("Soft Collision")
        
        
        return reward, done
                        
    def evaluate(self, state, draw=False) -> float:
        """
        Evaluates a DecisionNode by using a convex combination of distance to closest OOI point and angle to OOI.
        
        :param state: (np.ndarray) the state of the car and OOI (position(0:2), corner means(2:10), corner covariances(10:74))
        :return: (float) the cumulative reward observed during the tree traversing.
        """
        #### Basic data extraction
        # Pull out the car position and yaw from the state
        car_state = state[0]
        car_pos = car_state[0:2]
        car_yaw = car_state[2]
        
        # Pull out the OOI corner positions from the state
        ooi_state = state[1]
        ooi_reshaped = np.reshape(ooi_state, (4, 2))
        
        # Pull out covariances of corners and get trace for each corner
        cov_state = state[2]
        cov_diag = np.diag(cov_state)
        corner_traces = np.zeros((int(cov_diag.shape[0]/2),))
        
        # Iterate through covariance diagonal elements
        for i in range(corner_traces.shape[0]):
            # Calculate trace of covariance matrix for each corner
            corner_traces[i] = cov_diag[i*2] + cov_diag[(i*2) + 1]
        

        #### Find what corners are occluded by front of object using shapely
        # Create line of sight from the car center to each corner of the OOI
        car_pt = Point(car_pos)
        lines_of_sight = [LineString([car_pos, Point(corner)]) for corner in ooi_reshaped]
        
        # Check if each line of sight cross the OOI (which means it is not observable)
        ooi_polygon = Polygon(ooi_reshaped)
        intersection_bools = [line.crosses(ooi_polygon) for line in lines_of_sight]
        
        
        #### Calculate squared distance and bearing to each corner
        # Calculate squared distance of Car to OOI points
        squared_dists = np.sum((ooi_reshaped - car_pos)**2, axis=1)
        
        # Create scaled cumulative distance, bearing, occlusion turn steps, and occlusion distance steps
        cum_dist = 0.
        cum_bearing = 0.
        cum_occlusion = 0.
        
        # Iterate through each corner trace
        for i in range(corner_traces.shape[0]):
            # If corner is occluded use closest observable point weighting
            if intersection_bools[i]:
                # Return car position projected to corner observable line
                extension_length = 100
                
                # Extend vector from corner to prev and next corners by subtracting tail, multiplying by extension and then adding tail back
                prev_corner_extended = (ooi_reshaped[i-1] - ooi_reshaped[i]) * extension_length + ooi_reshaped[i-1]
                next_corner_extended = (ooi_reshaped[(i+1)%4] - ooi_reshaped[i]) * extension_length + ooi_reshaped[(i+1)%4]
                
                # Make these into Line Strings to use shapely functions
                obs_line1 = LineString([ooi_reshaped[i], prev_corner_extended])
                obs_line2 = LineString([ooi_reshaped[i], next_corner_extended])
                
                # Project the car position onto the line, get length along line
                obs_len1 = obs_line1.project(car_pt)
                obs_len2 = obs_line2.project(car_pt)
                
                # Use length along line to get projected point
                obs_pt1 = obs_line1.interpolate(obs_len1)
                obs_pt2 = obs_line2.interpolate(obs_len2)
                
                # Draw the projected closest observation points
                if draw:
                    self.ui.draw_arrow(car_pos, *obs_pt1.coords)
                    self.ui.draw_arrow(car_pos, *obs_pt2.coords)
                    
                # Convert into a numpy array
                obs_pts = np.array([*obs_pt1.coords, *obs_pt2.coords])
                
                # Iterate through each observable point to find the lowest cost point
                obs_pt_costs = np.zeros((obs_pts.shape[0],))
                for j, obs_pt in enumerate(obs_pts):
                    ##### First weighting is based on the number of timesteps to reach the observable point
                    
                    # Start with the difference in angle between the car and direction to observe the corner (extension point to corner angle)
                    yaw_err = angle_difference(car_yaw, np.arctan2(obs_pt[1] - ooi_reshaped[i][1], obs_pt[0] - ooi_reshaped[i][0]))
                    
                    # Subtract half of the car fov since we can see the corner if it is within half of the fov
                    yaw_err -= np.deg2rad(self.car.max_bearing / 2)
                    
                    # Make sure this is positive
                    yaw_err = max(yaw_err, 0)
                    
                    # Find maximum angle car can turn in one timestep
                    max_yaw = self.car.max_velocity * np.tan(np.deg2rad(self.car.max_steering_angle)) / self.car.length
                    
                    # Get number of timesteps to turn to be able to observe the corner
                    turn_steps = yaw_err / max_yaw
                    
                    #### Second weighting is based on the distance to the observable point
                    # Find the squared distance to the observable point
                    dist_err = np.sum((obs_pt - car_pos)**2)
                    
                    # Get the number of steps to reach that point
                    dist_steps = np.sqrt(dist_err) / self.car.max_velocity
                    
                    # print("Turn steps:", turn_steps, "Dist steps:", dist_steps)
                    
                    #### Normalize the turn and distance steps to a max of 10 steps
                    turn_steps = min_max_normalize(turn_steps, 0, 10)
                    dist_steps = min_max_normalize(dist_steps, 0, 10)
                    
                    # Add the weighted turn and distance steps to the observable point costs to later find the minimum
                    obs_pt_costs[j] = self.eval_ocl_turn_scale * turn_steps + self.eval_ocl_dist_scale * dist_steps
                    
                # Add the weighted minimum cost of the observable points to the cumulative occlusion
                cum_occlusion += corner_traces[i] * np.min(obs_pt_costs)

                
            # Otherwise weight based on bearing and distance to corner
            else:
                # Find bearing to current corner
                ooi_bearing = np.arctan2(ooi_reshaped[i, 1] - car_pos[1], ooi_reshaped[i, 0] - car_pos[0])
                
                # Subtract the car yaw to get the relative bearing to the OOI point
                bearing_delta = abs(ooi_bearing - car_yaw)
                
                # Normalize both the distance and bearing to be between 0 and 1
                norm_dist = min_max_normalize(squared_dists[i], 0., 2000.)
                norm_bearing = min_max_normalize(bearing_delta, 0., np.pi)
                
                # Add covariance trace weighted bearing and squared distance to cumulative distance and bearing
                cum_dist += corner_traces[i] * norm_dist
                cum_bearing += corner_traces[i] * norm_bearing
        
        # Return weighted convex combination (alpha and beta add to 1) of cumulative covariance weighted distance and bearing
        # This is negative because we want to minimize the distance and bearing (punishment for being far away or off bearing)
        evaluation_value = -self.evaluation_multiplier * (self.eval_dist_scale * cum_dist + self.eval_bearing_scale * cum_bearing + cum_occlusion)

        return evaluation_value
    
    def action_space_sample(self) -> np.ndarray:
        """
        Sample an action from the action space.
        
        :return: (np.ndarray) the sampled action
        """
        # Uniform sampling in continuous space
        if self.action_space_sample_heuristic == 'uniform_continuous':
            velocity = np.random.uniform(0, self.car.max_velocity)
            steering_angle = np.random.uniform(-self.car.max_steering_angle, self.car.max_steering_angle)
            
        # Uniform Discrete sampling with a specified number of options
        if self.action_space_sample_heuristic == 'uniform_discrete':
            velocity = np.random.choice(np.linspace(0, self.car.max_velocity, self.velocity_options))
            
            # If velocity is 0, steering angle must be 0 as well since the car cannot turn without moving
            if velocity == 0:
                steering_angle = 0
            else:
                steering_angle = np.random.choice(np.linspace(-self.car.max_steering_angle, self.car.max_steering_angle, self.steering_angle_options))
            
        return np.array([velocity, steering_angle])
    
    
    def get_all_actions(self) -> np.ndarray:
        """
        Get all possible actions in the action space.
        
        :return: (np.ndarray) the possible actions
        """
        # Create a meshgrid of all possible actions
        velocity = np.linspace(0, self.car.max_velocity, self.velocity_options)
        steering_angle = np.linspace(-self.car.max_steering_angle, self.car.max_steering_angle, self.steering_angle_options)
        actions = np.array(np.meshgrid(velocity, steering_angle)).T.reshape(-1, 2)
            
        # Iterate through the actions and remove the 0 velocity actions with non-zero steering angle
        possible_actions = deepcopy(actions)
        deletion_rows = []
        for i in range(len(actions)):
            if actions[i][0] == 0 and actions[i][1] != 0:
                deletion_rows.append(i)
            
        # Remove the rows with 0 velocity and non-zero steering angle
        possible_actions = np.delete(possible_actions, deletion_rows, axis=0)
        
        # # If reverse option is enabled, add a reverse option
        # if self.reverse_option:
        #     reverse_option = np.array([self.reverse_speed, 0.0])
        #     possible_actions = np.vstack((possible_actions, reverse_option))
            
        return possible_actions
    
    
if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(description='Run Toy Measurement Control')
    
    # Add arguments
    parser.add_argument('--one_iteration', type=bool)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create an instance of ToyMeasurementControl using command line arguments
    tmc = ToyMeasurementControl(one_iteration=args.one_iteration)
    tmc.run()