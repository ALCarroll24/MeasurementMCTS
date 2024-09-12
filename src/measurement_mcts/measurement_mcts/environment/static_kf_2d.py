import numpy as np
from math import *
from measurement_mcts.utils.utils import wrapped_angle_diff, get_ellipse_scaling

# Measurement noise, size needs to be 2 * number of observations for x and y (row length = column length)
def measurement_model(z, car_pos, car_yaw, min_range=5., min_bearing=5., range_dev=1., bearing_dev=0.5):    
    # Calculate the range and bearing to the corner
    dist = max(np.linalg.norm(z - car_pos), min_range)  # Distance to observation
    abs_bearing = np.arctan2(z[1] - car_pos[1], z[0] - car_pos[0]) - car_yaw  # Direction of observation
    sensor_bearing = wrapped_angle_diff(abs_bearing, car_yaw)  # Angle between center of sensor and observation
    bearing_scale = np.clip(np.abs(sensor_bearing), np.radians(min_bearing), None)  # Lateral scaling in magnitude matrix
    
    # Rotation matrix to rotate distribution before scaling
    G = np.array([[np.cos(sensor_bearing), -np.sin(sensor_bearing)], [np.sin(sensor_bearing), np.cos(sensor_bearing)]])
    
    # Magnitude matrix to scale distribution
    M = np.diag([range_dev**2 * dist, bearing_dev**2 * np.pi * dist * bearing_scale])
    
    # Calculate covariance matrix for this corner
    return G @ M @ G.T

class StaticKalmanFilter:
    def __init__(self, range_dev=1., min_range=5., bearing_dev=0.5, min_bearing=5., ui=None):
        # Load noise scaling values and minimum values for measurement model
        self.range_dev = range_dev  # Range scaling value for measurement model
        self.min_range = min_range  # Minimum range value (m) for measurement model
        self.bearing_dev = bearing_dev  # Bearing scaling value for measurement model
        self.min_bearing = min_bearing  # Minimum bearing value (degrees) for measurement model
        self.ui = ui  # UI object for plotting

        # Dimension of state and observation vectors
        self.s_dim = 2  # Now just for one point (x, y)
        self.z_dim = 2
        
        # State transition matrix, s_dim dimensional identity matrix because state is fully observable
        self.F = np.eye(self.s_dim)

        # State to observation matrix, s_dim x z_dim matrix
        self.H = np.eye(self.s_dim)
    
    # Update step of KF, get posterior distribution, function of measurement and sensor
    def update(self, mean, cov, z, car_state):        
        # Pull out car position and yaw
        car_pos = car_state[0:2]
        car_yaw = car_state[3]
        
        # Take copies of the mean and covariance to be safe (avoid changing the original)
        mean = mean.copy()
        cov = cov.copy()

        # Get the measurement matrix R
        R = measurement_model(z, car_pos, car_yaw, min_range=self.min_range, range_dev=self.range_dev,
                              min_bearing=self.min_bearing, bearing_dev=self.bearing_dev)
        I = np.eye(self.s_dim)

        # Kalman gain
        S_k = self.H @ cov @ self.H.T + R
        K_k = cov @ self.H.T @ np.linalg.inv(S_k)

        # Measurement residual
        z_bar = z - self.H @ mean

        # Update state and covariance
        mean = mean + K_k @ z_bar
        cov = (I - K_k @ self.H) @ cov
        
        return mean, cov
    
    # Draw the state on the UI
    def draw_state(self, mean, cov):
        if self.ui is None:
            raise ValueError('UI object is not set')
        
        for i in range(0, len(mean), 2):            
            # Get the scalings and angle of ellipse
            scalings, angle = get_ellipse_scaling(cov)
            
            # Draw the mean point and covariance ellipse
            self.ui.draw_point(mean[i:i+2], color='g')
            self.ui.draw_ellipse(mean[i:i+2], scalings[0], scalings[1], angle=angle, color='b')
            
        # Draw a polygon between the points
        self.ui.draw_polygon(mean.reshape(4, 2), color='r', closed=True, linestyle='-')

if __name__ == '__main__':
    # Initial state vector, uncertainty and time
    mean = np.array([0, 0])
    cov = np.diag([0.1, 0.1])

    # Create KF object
    kf = StaticKalmanFilter()

    # Create measurement vector
    z = np.array([0.1, 0.1])
    z2 = np.array([0.2, 0.2])

    # Update KF
    mean, cov = kf.update(mean, cov, z, car_state=[0, 0, 0, 0])
    mean, cov = kf.update(mean, cov, z2, car_state=[0, 0, 0, 0])

    # Print results
    print("Updated mean:", mean)
    print("Updated covariance:", cov)
