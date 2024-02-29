import numpy as np
from typing import Tuple
from ui import MatPlotLibUI
from car import Car
from ooi import OOI
from vkf import VectorizedStaticKalmanFilter
from mcts.mcts import MCTS
from mcts.hash import hash_action, hash_state
import time

class ToyMeasurementControl:
    def __init__(self):
        # Determine update rate
        self.hz = 20.0
        self.period = 1.0 / self.hz
        
        # Parameters
        self.final_cov_trace = 0.1
        self.action_space_sample_heuristic = 'uniform_discrete'
        self.velocity_options = 5  # number of discrete options for velocity
        self.steering_angle_options = 5  # number of discrete options for steering angle
        
        # Create a plotter object
        self.ui = MatPlotLibUI(update_rate=self.hz)
        
        # Create a car object
        self.car = Car(self.ui, np.array([30.0, 30.0]), 90, self.hz)
        
        # Create an OOI object
        self.ooi = OOI(self.ui, car_max_range=self.car.max_range, car_max_bearing=self.car.max_bearing)
        
        # Create a Static Vectorized Kalman Filter object
        self.vkf = VectorizedStaticKalmanFilter(np.array([50]*8), np.diag([8]*8), 4.0)


    def run(self): 
    # Loop until matplotlib window is closed (handled by the UI class)
        while(True):
        
            # Get the observation from the OOI, pass it to the KF for update
            observable_corners, indeces = self.ooi.get_noisy_observation(self.car.get_state())
            self.vkf.update(observable_corners, indeces, self.car.get_state())
            
            ############################ MANUAL CONTROL ############################
            # Get the control inputs from the arrow keys, pass them to the car for update
            vel_steering_tuple = self.car.get_arrow_key_control()
            self.car.add_input(vel_steering_tuple)   # This adds the control input rather than directly setting it (easier for keyboard control)
            
            
            ############################ AUTONOMOUS CONTROL ############################
            # Create an MCTS object
            mcts = MCTS(initial_obs=self.get_state(), env=self, K=0.3**5,
                        _hash_action=hash_action, _hash_state=hash_state)
            mcts.learn(100, progress_bar=False)
            action_vector = mcts.best_action()
            print("Past MCTS")
            
            self.car.update(self.period)
        
            # Update the displays, and pause for the period
            self.car.draw()
            self.ooi.draw()
            self.vkf.draw(self.ui)
            self.ui.update()
            
    # Returns full state -> Tuple[Car State, Corner Mean, Corner Covariance, Done]:
    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.car.get_state(), self.vkf.get_mean(), self.vkf.get_covariance()
    
    def step(self, state, action) -> Tuple[float, np.ndarray]:
        """
        Step the environment by one time step. The action is applied to the car, and the state is observed by the OOI.
        The observation is then passed to the KF for update.
        
        :param state: (np.ndarray) the state of the car and KF (Car state(x,y,yaw), corner means, corner covariances)
        :param action: (np.ndarray) the control input to the car (velocity, steering angle)
        :return: (float, np.ndarray) the reward of the state-action pair, and the new state
        """
        # Pull out the state elements
        car_state, corner_means, corner_cov = state
        
        # Apply the action to the car
        new_car_state = self.car.update(self.period, action, simulate=True, starting_state=car_state)
        
        # Get the observation from the OOI, pass it to the KF for update
        observable_corners, indeces = self.ooi.get_noisy_observation(new_car_state)
        new_mean, new_cov = self.vkf.update(observable_corners, indeces, new_car_state, 
                                            simulate=True, s_k=corner_means, P_k=corner_cov)
        
        # Combine the new car state and the new mean and covariance into a new state
        new_state = (new_car_state, new_mean, new_cov)
        
        # Find the reward based on updated state
        reward, done = self.get_reward(new_state, action)
        
        # Return the reward and the new state
        return new_state, reward, done
    
    def get_reward(self, state, action) -> Tuple[float, bool]:
        """
        Get the reward of the new state-action pair.
        
        :param new_state: (np.ndarray) the new state of the car and OOI (position(0:2), corner means(2:10), corner covariances(10:74))
        :param action: (np.ndarray) the control input to the car (velocity, steering angle)
        :return: (float, bool) the reward of the state-action pair, and whether the episode is done
        """
        # Pull out the state elements
        car_state, corner_means, corner_cov = state
        
        # Find the sum of the diagonals
        trace = np.trace(corner_cov)
        
        # Find whether the episode is done TODO: done needs to also account for horizon length
        done = trace < self.final_cov_trace

        # Find the reward
        reward = -trace
        
        return reward, done
    
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
            steering_angle = np.random.choice(np.linspace(-self.car.max_steering_angle, self.car.max_steering_angle, self.steering_angle_options))
            
        return np.array([velocity, steering_angle])
    
    
if __name__ == '__main__':  
    tmc = ToyMeasurementControl()
    tmc.run()