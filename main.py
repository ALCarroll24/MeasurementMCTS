import numpy as np
from typing import Tuple
from ui import MatPlotLibUI
from car import Car
from ooi import OOI
from vkf import VectorizedStaticKalmanFilter

class ToyMeasurementControl:
    def __init__(self):
        # Determine update rate
        self.hz = 20.0
        self.period = 1.0 / self.hz
        
        # Parameters
        self.final_cov_trace = 0.1
        
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
            
            # Get the control inputs from the arrow keys, pass them to the car for update
            vel_steering_tuple = self.car.get_arrow_key_control()
            self.car.add_input(vel_steering_tuple)   # This adds the control input rather than directly setting it (easier for keyboard control)
            
            self.car.step(self.period)
        
            # Update the displays, and pause for the period
            self.car.draw()
            self.ooi.draw()
            self.vkf.draw(self.ui)
            self.ui.update()
            
    
    def step(self, state, action) -> Tuple[float, np.ndarray]:
        """
        Step the environment by one time step. The action is applied to the car, and the state is observed by the OOI.
        The observation is then passed to the KF for update.
        
        :param state: (np.ndarray) the state of the car and OOI (position(0:2), corner means(2:10), corner covariances(10:74))
        :param action: (np.ndarray) the control input to the car (velocity, steering angle)
        :return: (float, np.ndarray) the reward of the state-action pair, and the new state
        """
        # Apply the action to the car
        new_car_state = self.car.step(self.period, action, simulate=True, starting_state=state)
        
        # Get the observation from the OOI, pass it to the KF for update
        observable_corners, indeces = self.ooi.get_noisy_observation(new_car_state)
        new_mean, new_cov = self.vkf.update(observable_corners, indeces, new_car_state, 
                                            simulate=True, s_k=state[2:10], P_k=state[10:74])
        
        # Combine the new car state and the new mean and covariance into a new state
        new_state = np.concatenate((new_car_state, new_mean, new_cov))
        
        # Find the reward based on updated state
        reward, done = self.get_reward(new_state, action)
        
        # Return the reward and the new state
        return new_state, reward, done
    
    def get_reward(self, new_state, action) -> Tuple[float, bool]:
        """
        Get the reward of the new state-action pair.
        
        :param new_state: (np.ndarray) the new state of the car and OOI (position(0:2), corner means(2:10), corner covariances(10:74))
        :param action: (np.ndarray) the control input to the car (velocity, steering angle)
        :return: (float, bool) the reward of the state-action pair, and whether the episode is done
        """
        # Pull out the portions of the state
        car_position = new_state[0:2]
        corner_means = new_state[2:10]
        corner_cov = new_state[10:74]
        
        # Convert the corner covariance into a 8x8 matrix and find the trace
        corner_cov = corner_cov.reshape((8,8))
        trace = np.trace(corner_cov)
        
        # Find whether the episode is done
        done = trace < self.final_cov_trace

        # Find the reward
        reward = -trace
        
        return reward, done
    
if __name__ == '__main__':  
    tmc = ToyMeasurementControl()
    tmc.run()