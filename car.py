import numpy as np
import threading
from ui import MatPlotLibUI
from utils import wrap_angle, sat_value, rotate
import time
from typing import Tuple
from shapely import Polygon

class Car:
    def __init__(self, ui, position, yaw, length=4., width=2., max_range=100., max_bearing=45., max_velocity=10., max_steering_angle=45., range_arrow_length=10.0):
        self.ui = ui
        self.position = position  # x, y
        self.yaw = np.radians(yaw)  # Orientation
        self.length = length
        self.width = width
        self.max_range = max_range
        self.max_bearing = max_bearing
        self.max_velocity = max_velocity
        self.max_steering_angle = max_steering_angle
        self.velocity = 0.0
        self.steering_angle = 0.0
        self.range_arrow_length = range_arrow_length

    def update(self, dt, action,  simulate=False, starting_state=None):
        # Pull the inputs from the action tuple
        velocity, steering_angle = action
            
        # If we are doing forward simulation, we need to pass in the starting state
        # MUY IMPORTANTE - take a copy of the state, otherwise we will be modifying the original state object
        if simulate is not None and starting_state is not None:
            position = np.copy(starting_state[0:2])
            yaw = np.copy(starting_state[2])
        else:
            position = self.position
            yaw = self.yaw
        
        # Update car state using the bicycle model
        position[0] += velocity * np.cos(yaw) * dt
        position[1] += velocity * np.sin(yaw) * dt
        yaw += (velocity / self.length) * np.tan(steering_angle) * dt
        
        # Saturate inputs
        velocity = sat_value(velocity, self.max_velocity)
        steering_angle = sat_value(steering_angle, np.radians(self.max_steering_angle))
        
        # Keep angle between [-pi, pi]
        yaw = wrap_angle(yaw)
        
        # Only update state if we are not simulating
        if not simulate:
            self.position = position
            self.yaw = yaw
            
        # Otherwise return the new state
        else:
            return np.array([position[0], position[1], yaw])
            
    def get_car_polygon(self, car_state=None):
        # If we are calculating the polygon for a different state, use that state
        if car_state is not None:
            position = car_state[0:2]
            yaw = car_state[2]
            
        # Otherwise, use the current state
        else:
            position = self.position
            yaw = self.yaw
        
        points_no_yaw = [[position[0] + self.width / 2, position[1] + self.length / 2],
                         [position[0] + self.width / 2, position[1] - self.length / 2],
                         [position[0] - self.width / 2, position[1] - self.length / 2],
                         [position[0] - self.width / 2, position[1] + self.length / 2]]
        
        # Rotate the points by the yaw
        points = rotate(np.array(points_no_yaw) - position, yaw - np.radians(90)) + position
        
        return Polygon(points)
        

    def draw(self):
        # Draw the car as a rectangle in the UI
        self.ui.draw_rectangle(self.position, self.length, self.width, self.yaw)
        
        # Draw range and bearing indicators
        self.ui.draw_arrow(self.position, self.position + np.array([np.cos(self.yaw + np.radians(self.max_bearing))*self.range_arrow_length,
                                                                    np.sin(self.yaw + np.radians(self.max_bearing))*self.range_arrow_length]))
        self.ui.draw_arrow(self.position, self.position + np.array([np.cos(self.yaw - np.radians(self.max_bearing))*self.range_arrow_length,
                                                                    np.sin(self.yaw - np.radians(self.max_bearing))*self.range_arrow_length]))
        
        # # Draw the car's polygon to check it is working
        # x, y = self.get_car_polygon().exterior.xy
        # for x, y in zip(x, y):
        #     self.ui.draw_circle((x, y), 0.3)

        
    def draw_state(self, state):
        # Draw the car as a rectangle in the UI
        self.ui.draw_rectangle(state[0:2], self.length, self.width, state[2])
                               
        # Draw range and bearing indicators
        self.ui.draw_arrow(state[0:2], state[0:2] + np.array([np.cos(state[2] + np.radians(self.max_bearing))*self.range_arrow_length,
                                                                np.sin(state[2] + np.radians(self.max_bearing))*self.range_arrow_length]))
        self.ui.draw_arrow(state[0:2], state[0:2] + np.array([np.cos(state[2] - np.radians(self.max_bearing))*self.range_arrow_length,
                                                                np.sin(state[2] - np.radians(self.max_bearing))*self.range_arrow_length]))
        
    def test_actions(self):
        # Test drive the car around using a for loop
        self.velocity = 5.0
        self.steering_angle = np.radians(5.0)
        for i in range(100):
            self.update(0.1)
            time.sleep(0.1)
            
    def get_arrow_key_control(self):
        velocity = 0.0
        steering_angle = 0.0
        
        # Grab inputs from arrow keys
        if 'up' in self.ui.keys:
            velocity += 0.4
        if 'down' in self.ui.keys:
            velocity -= 0.4
        if 'left' in self.ui.keys:
            steering_angle += np.radians(5.0)
        if 'right' in self.ui.keys:
            steering_angle -= np.radians(5.0)
        
        # print("Keys: ", self.ui.keys)
        # print("Velocity: ", self.velocity, "Steering angle: ", np.degrees(self.steering_angle))
        
        # Return the tuple of inputs
        return (velocity, steering_angle)
          
    def set_input(self, vel_steering):
        self.velocity = vel_steering[0]
        self.steering_angle = vel_steering[1]
        
    def add_input(self, relative_input: Tuple[float, float], last_input: Tuple[float, float]) -> Tuple[float, float]:
        new_input = last_input + relative_input
        
        # If there are no inputs from the arrow keys, decay the velocity and steering angle
        if len(self.ui.keys) == 0:
            new_input[0] *= 0.95    # Decay the velocity
            new_input[1] *= 0.6     # Decay the steering angle
            
        # Return the new input
        return new_input
        
    def get_state(self):
        return np.array([self.position[0], self.position[1], self.yaw])

if __name__ == '__main__':
    
    # Determine update rate
    hz = 10.0
    period = 1.0 / hz
    
    # Create a plotter object
    ui = MatPlotLibUI(update_rate=hz)
    
    # Create a car object
    car = Car(ui, np.array([10.0, 10.0]), 0, hz)
    
    # Test drive the car around using a for loop
    # car.velocity = 3.0
    # car.steering_angle = np.radians(2.0)
    # for i in range(100):
    #     car.update(period)
    #     ui.update()
    
    # Start the car action thread
    # car_action_thread = threading.Thread(target=car.test_actions)
    # car_action_thread.start()
    
    # Start the ui plotting loop
    # ui.start_async_loop()
    