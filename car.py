import numpy as np
import threading
from ui import MatPlotLibUI
from utils import wrap_angle, sat_value
import time

class Car:
    def __init__(self, ui, position, yaw, update_rate, length=4.0, width=2.0, max_range=20.0, max_bearing=45.0, max_velocity=10.0, max_steering_angle=45.0):
        self.ui = ui
        self.position = position  # x, y
        self.yaw = np.radians(yaw)  # Orientation
        self.hz = update_rate
        self.period = 1.0 / self.hz
        self.length = length
        self.width = width
        self.max_range = max_range
        self.max_bearing = max_bearing
        self.max_velocity = max_velocity
        self.max_steering_angle = max_steering_angle
        self.velocity = 0.0
        self.steering_angle = 0.0

    def step(self, dt, action=None,  simulate=False, starting_state=None):
        # Let's assume that the action is a tuple of (velocity, steering_angle)
        if action is not None:
            self.velocity = action[0]
            self.steering_angle = action[1]
            
        # If we are doing forward simulation, we need to pass in the starting state
        if simulate is not None and starting_state is not None:
            position = starting_state[0:2]
            yaw = starting_state[2]
        else:
            position = self.position
            yaw = self.yaw
        
        # Update car state using the bicycle model
        position[0] += self.velocity * np.cos(self.yaw) * dt
        position[1] += self.velocity * np.sin(self.yaw) * dt
        yaw += (self.velocity / self.length) * np.tan(self.steering_angle) * dt
        
        # Saturate inputs
        self.velocity = sat_value(self.velocity, self.max_velocity)
        self.steering_angle = sat_value(self.steering_angle, np.radians(self.max_steering_angle))
        
        # Keep angle between [-pi, pi]
        self.yaw = wrap_angle(self.yaw)
        
        # Only update state if we are not simulating
        if not simulate:
            self.position = position
            self.yaw = yaw
            
        # Otherwise return the new state
        else:
            return np.array([position[0], position[1], yaw])
            
        

    def draw(self):
        # Draw the car as a rectangle in the UI
        self.ui.draw_rectangle(self.position, self.length, self.width, self.yaw)
        
        # Draw range and bearing indicators
        self.ui.draw_arrow(self.position, self.position + np.array([np.cos(self.yaw + np.radians(self.max_bearing))*self.max_range,
                                                                    np.sin(self.yaw + np.radians(self.max_bearing))*self.max_range]))
        self.ui.draw_arrow(self.position, self.position + np.array([np.cos(self.yaw - np.radians(self.max_bearing))*self.max_range,
                                                                    np.sin(self.yaw - np.radians(self.max_bearing))*self.max_range]))
        
        
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
        
    def add_input(self, vel_steering):
        self.velocity += vel_steering[0]
        self.steering_angle += vel_steering[1]
        
        # If there are no inputs from the arrow keys, decay the velocity and steering angle
        if len(self.ui.keys) == 0:
            self.velocity *= 0.95
            self.steering_angle *= 0.6
        
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
    