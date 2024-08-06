import numpy as np
import threading
from ui import MatPlotLibUI
from utils import wrap_angle, sat_value, rotate, angle_difference
import time
from typing import Tuple
from shapely import Polygon

class Car:
    def __init__(self, ui, initial_state, max_range=100., max_bearing=45., max_velocity=10., range_arrow_length=10.0):
        self.ui = ui
        
        # Save initial state
        self.state = initial_state
        
        # Save parameters
        self.max_range = max_range # m
        self.max_bearing = np.radians(max_bearing) # Max sensor fov in radians (converted from degrees)
        self.max_velocity = max_velocity # m/s
        self.range_arrow_length = range_arrow_length
        
        ###### Jeep Grand Cherokee Trailhawk Parameters ######
        ### Car dimension parameters
        self.wheelbase = 9.575 * 0.3048  # Front axle to back axle in feet (converted to meters)
        self.length = 15.825 * 0.3048  # Total length of car in feet (converted to meters)
        self.width = 7.0 * 0.3048  # Total width of car in feet (converted to meters)
        
        ### Longitudinal parameters
        max_engine_torque = 260.0 * 1.356  # Maximum engine torque in Nm (converted from lb-ft)
        gear_ratios = np.array([-3.0, 3.0, 1.67, 1.0, 0.75, 0.67]) # Gear ratios
        final_drive_ratio = 3.47  # Final drive ratio
        gears_to_average = [1, 2]  # Gears to average to calculate max torque (Most time for this problem is spent in these gears)
        average_gear_ratio = np.mean(gear_ratios[gears_to_average]) * final_drive_ratio  # Average gear ratio
        torque_ratio = 1/3  # Percentage of torque to assume is readily available from max (for largest acceleration action limit)
        max_wheel_torque = torque_ratio * max_engine_torque * average_gear_ratio * final_drive_ratio  # Maximum torque available at wheels in Nm
        tire_radius = 32.0 * 0.5 * 0.0254  # Tire radius in meters (converted from inches diameter)
        max_longitudinal_force = max_wheel_torque / tire_radius  # Maximum longitudinal force available at wheels in Nm
        gross_vehicle_mass = 6000.0 * 0.453592  # Gross vehicle mass (passengers+cargo) in kg (converted from lbs)
        
        ### Longitudinal output class variables
        self.max_acceleration = max_longitudinal_force / gross_vehicle_mass  # Maximum acceleration given F=ma in m/s^2
        self.min_acceleration = -0.8 * 9.81  # Maximum deceleration with brakes in m/s^2
        
        ### Lateral Parameters
        max_steering_wheel_turns = 3.2  # Maximum steering wheel turns from lock to lock (far left to far right)
        steering_ratio = np.mean([15.7, 18.9])  # Steering wheel turns to wheel turns (averaging center and at lock)
        self.max_steering_angle = np.radians(0.5 * max_steering_wheel_turns * 360 / steering_ratio)  # Maximum steering angle in radians
        quarter_rotation_time = 0.5  # Time to rotate steering wheel 90 degrees (used to calculate acceleration limit)
        
        # Lateral output class variables
        # From rearanging: theta = 1/2 * alpha * t^2, we get alpha = 2 * theta / t^2:
        self.max_steering_alpha = np.pi / (quarter_rotation_time**2) / steering_ratio  # Maximum steering angular acceleration in rad/s^2
        self.max_steering_angle = max_steering_wheel_turns * np.pi / steering_ratio  # Maximum steering angle from center in radians

        
    # Vehicle model Matrices
    def get_A_matrix(self, V_xt, delta_t, dt):
        # Maps state space to new state (non-linear by function parameters)
        A = np.array([[1, 0, np.cos(delta_t) * dt, 0, 0, 0],
                      [0, 1, np.sin(delta_t) * dt, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, V_xt * np.tan(delta_t)/self.wheelbase, 0],
                      [0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 1]])
        return A
    
    def get_B_matrix(self, dt):
        # Maps control inputs (linear acceleration and angular acceleration) to state space
        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, 0],
                      [0, 0],
                      [0, dt]])
        return B
    
    # Given action and state compute new state and return
    def car_model(self, action_vec, state_vec, dt):
        # State vector: [X, Y, v_x, psi, delta, delta_dot] (x pos, y pos, longitudinal velocity, yaw, steering angle, steering angle rate)
        # Action vector: [v_x_dot, delta_dot_dot] (linear acceleration, steering angle acceleration)
        
        # Copy action so we can modify it (can't modify input because it's a tuple)
        action_vec = np.copy(action_vec)
        
        # Actions are between [-1, 1] so scale them to the max values
        if action_vec[0] > 0:
            action_vec[0] *= self.max_acceleration
            
        # Negate since minimum acceleration is negative and so is the action
        else:
            action_vec[0] *= -self.min_acceleration
            
        # For steering the output max and min are the same so just scale it by max steering acceleration (also between [-1, 1] initially)
        action_vec[1] *= self.max_steering_alpha
        
        # Compute new state
        new_state = self.get_A_matrix(state_vec[2], state_vec[4], dt) @ state_vec + self.get_B_matrix(dt) @ action_vec
        
        # Keep vehicle yaw angle between [-pi, pi] (wrap around)
        new_state[3] = wrap_angle(new_state[3])
        
        # Keep steering angle between [-max_steering_angle, max_steering_angle]
        new_state[4] = np.clip(new_state[4], -self.max_steering_angle, self.max_steering_angle)
        
        return new_state
    
    # Update the car class or return the new state (when given a starting state) based on the action
    def update(self, dt, action, starting_state=None):        
        # If we are doing forward simulation, we need to pass in the starting state
        # MUY IMPORTANTE - take a copy of the state, otherwise we will be modifying the original state object
        if starting_state is not None:
            state = np.copy(starting_state)
        else:
            state = self.state
        
        # Use the car model to update the state
        new_state = self.car_model(action, state, dt)
        
        # Only update class state if we are not simulating
        if starting_state is None:
            self.state = new_state
            
        # Otherwise return the new state
        else:
            return new_state
        
    def get_action_pure_pursuit(self, target_point: np.ndarray, simulate: bool = False, starting_state: np.ndarray = None):
        # Use the update function to update the car state based on pure pursuit and target point

        # Retrieve the current or starting state
        if simulate and starting_state is not None:
            position = np.copy(starting_state[0:2])
            yaw = np.copy(starting_state[3])
        else:
            position = self.state[0:2]
            yaw = self.state[3]

        # Calculate the vector to the target point in the global frame
        vector_to_target = target_point - position

        # Rotate the vector into the car's frame of reference
        rotated_vector = rotate(vector_to_target, -yaw)
        dx, dy = rotated_vector

        # Compute the lookahead distance (distance to the target point)
        lookahead_distance = np.hypot(dx, dy)

        # Ensure lookahead_distance is not zero to avoid division by zero
        if lookahead_distance == 0:
            return

        # Calculate the curvature of the path to the target point
        curvature = 2 * dy / (lookahead_distance ** 2)

        # Compute the required steering angle based on the curvature
        required_steering_angle = np.arctan(curvature * self.length)

        # Limit the steering angle to the car's maximum steering angle
        required_steering_angle = np.clip(required_steering_angle, -np.radians(self.max_steering_angle), np.radians(self.max_steering_angle))

        # Determine the velocity; in pure pursuit, it is often kept constant or can be adjusted
        velocity = self.max_velocity

        return velocity, np.degrees(required_steering_angle)
    
    def get_action_follow_path(self, path: np.ndarray, look_ahead_dist: float, return_target_point: bool = False, simulate: bool = False, starting_state: np.ndarray = None):
        # Use the update function to update the car state based on following a path

        # Retrieve the current or starting state
        if simulate and starting_state is not None:
            position = np.copy(starting_state[0:2])
            yaw = np.copy(starting_state[3])
        else:
            position = self.state[0:2]
            yaw = self.state[3]

        # Find the index of the closest point
        closest_point_index = np.argmin(np.linalg.norm(path - position, axis=1))

        # See which direction we are going in the path
        high_index_yaw = np.arctan2((path[closest_point_index + 1] - path[closest_point_index])[1], 
                                    (path[closest_point_index + 1] - path[closest_point_index])[0])
        low_index_yaw = np.arctan2((path[closest_point_index - 1] - path[closest_point_index])[1], 
                                   (path[closest_point_index - 1] - path[closest_point_index])[0])
        
        # Find the direction most aligned with car yaw and set the iteration direction
        min_direction = np.argmin([angle_difference(yaw, high_index_yaw), 
                                   angle_difference(yaw, low_index_yaw)])
        iter_direction = 1 if min_direction == 0 else -1
        
        # Find the lookahead point on the path
        lookahead_index = closest_point_index
        while np.linalg.norm(path[lookahead_index] - position) < look_ahead_dist:
            lookahead_index += iter_direction
            
            # If we reach the end or beginning of the path wrap the index
            if lookahead_index >= len(path):
                lookahead_index = 0
            elif lookahead_index < 0:
                lookahead_index = len(path) - 1
        
        # If returning target point
        if return_target_point:
            return self.get_action_pure_pursuit(path[lookahead_index], simulate, starting_state), path[lookahead_index]

        # Use the update_pure_pursuit method to update the car state based on the lookahead point
        return self.get_action_pure_pursuit(path[lookahead_index], simulate, starting_state)

    def get_car_polygon(self, car_state=None):
        # If we are calculating the polygon for a different state, use that state
        if car_state is not None:
            position = car_state[0:2]
            yaw = car_state[2]
            
        # Otherwise, use the current state
        else:
            position = self.state[:2]
            yaw = self.state[3]
        
        points_no_yaw = [[position[0] + self.width / 2, position[1] + self.length / 2],
                         [position[0] + self.width / 2, position[1] - self.length / 2],
                         [position[0] - self.width / 2, position[1] - self.length / 2],
                         [position[0] - self.width / 2, position[1] + self.length / 2]]
        
        # Rotate the points by the yaw
        points = rotate(np.array(points_no_yaw) - position, yaw - np.radians(90)) + position
        
        return Polygon(points)

    def draw(self):
        # Draw the car as a rectangle in the UI
        self.ui.draw_rectangle(self.state[:2], self.length, self.width, self.state[3])
        
        # Draw range and bearing indicators
        self.ui.draw_arrow(self.state[:2], self.state[:2] + np.array([np.cos(self.state[3] + self.max_bearing)*self.range_arrow_length,
                                                                    np.sin(self.state[3] + self.max_bearing)*self.range_arrow_length]))
        self.ui.draw_arrow(self.state[:2], self.state[:2] + np.array([np.cos(self.state[3] - self.max_bearing)*self.range_arrow_length,
                                                                    np.sin(self.state[3] - self.max_bearing)*self.range_arrow_length]))
        
        # # Draw the car's polygon to check it is working
        # x, y = self.get_car_polygon().exterior.xy
        # for x, y in zip(x, y):
        #     self.ui.draw_circle((x, y), 0.3)

        
    def draw_state(self, state):
        # Draw the car as a rectangle in the UI
        self.ui.draw_rectangle(state[0:2], self.length, self.width, state[2])
                               
        # Draw range and bearing indicators
        self.ui.draw_arrow(state[0:2], state[0:2] + np.array([np.cos(state[2] + self.max_bearing)*self.range_arrow_length,
                                                                np.sin(state[2] + self.max_bearing)*self.range_arrow_length]))
        self.ui.draw_arrow(state[0:2], state[0:2] + np.array([np.cos(state[2] - self.max_bearing)*self.range_arrow_length,
                                                                np.sin(state[2] - self.max_bearing)*self.range_arrow_length]))
        
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
        return self.state

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
    