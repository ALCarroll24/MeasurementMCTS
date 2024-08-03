import numpy as np
from math import *
from utils import wrapped_angle_diff

# Measurement noise, size needs to be 2 * number of observations for x and y (row length = column length)
def measurement_model(z, obs_indices, car_pos, car_yaw, min_dist=1, range_dev=1., bearing_dev=1.):
    # Create output array of length 2 * number of observations x 2 * number of observations
    R = np.zeros((2 * len(obs_indices), 2 * len(obs_indices)))
    
    # Apply range bearing sensor model for each of the corners in the OOI
    for i in range(len(obs_indices)):
        # Calculate the range and bearing to the corner
        dist = max(np.linalg.norm(car_pos - z[i,:]), min_dist) # Distance to observation
        abs_bearing = np.arctan2(z[i, 1] - car_pos[1], z[i, 0] - car_pos[0]) - car_yaw # Direction of observation
        bearing_diff = wrapped_angle_diff(abs_bearing, car_yaw) # Angle between center of sensor and observation
        
        # Rotation matrix to rotate distribution before scaling
        G = np.array([[np.cos(abs_bearing), -np.sin(abs_bearing)], [np.sin(abs_bearing), np.cos(abs_bearing)]])
        
        # Magnitude matrix to scale distribution
        M = np.diag([range_dev**2 * dist, bearing_dev**2 * np.pi * dist])
        
        # Calculate covariance matrix for this corner
        r = G @ M @ G.T
        
        # Place into the measurement noise matrix
        R[i*2:i*2+2, i*2:i*2+2] = r
        
    return R

class StaticKalmanFilter:
    def __init__(self, _s, _P, range_dev=1., bearing_dev=1.):
        # Load noise scaling values
        self.range_dev = range_dev
        self.bearing_dev = bearing_dev

        # Dimension of state and observation vectors
        self.s_dim = 8
        self.z_dim = 8

        # Load in the initial state vector and covariance matrix
        self.s_k = _s.flatten()
        self.P_k = _P
        
    # State transition matrix, s_dim dimensional identity matrix
    def F(self):
        # Start with a two dimensional identity matrix
        F = np.eye(self.s_dim)

        return F
    
    # Maps state space to observation space,
    # The column length is given by state dimension, but the row length is given by 2 * number of observations for x and y
    def H(self, obs_indices):
        H = np.zeros((2 * len(obs_indices), self.s_dim)) # 2 * len(obs_indices) since each observation has an x and y component
        for i, index in enumerate(obs_indices):
            col_index = 2 * index      # Function mapping index of observation to index of diagonal point list of (x,y)
            
            H[2 * i, col_index] = 1        # Assignment to x portion
            H[2 * i + 1, col_index+1] = 1  # Assignment to y portion
        
        return H
    
    # Update step of KF, get posterior distribution, function of measurement and sensor
    def update(self, z, obs_indices, car_state, simulate=False, s_k_=None, P_k_=None):
        # Make z flat for comparison to flat state vector (4,2) -> (8,)
        z_flat = z.flatten()
        
        # Pull out car position and yaw
        car_pos = car_state[0:2]
        car_yaw = car_state[2]
        
        # If we are not simulating use the class variables, otherwise use the passed in variables
        if not simulate:
            s_k = np.copy(self.s_k)
            P_k = np.copy(self.P_k)
        else:
            # MUY IMPORTANTE - take a copy of the state, otherwise we will be modifying the original state object
            s_k = np.copy(s_k_)
            P_k = np.copy(P_k_)

        # Get function outputs beforehand
        H = self.H(obs_indices)
        R = measurement_model(z, obs_indices, car_pos, car_yaw, range_dev=self.range_dev, bearing_dev=self.bearing_dev)
        I = np.eye(self.s_dim)
        
        # Kalman gain
        S_k = H @ P_k @ H.T + R
        K_k = P_k @ H.T @ np.linalg.inv(S_k)

        # Measurement residual
        z_bar = z_flat - H @ s_k

        # Update state and covariance
        s_k = s_k + K_k @ z_bar
        P_k = (I - K_k @ H) @ P_k
        
        # If we are not simulating update the class variables
        if not simulate:
            self.s_k = s_k
            self.P_k = P_k
        # Otherwise return the updated state and covariance
        else:
            return s_k, P_k
        
    # Draw the mean and covariance on the UI
    def draw(self, ui):
        for i in range(0, self.s_dim, 2):
            # Find the eigenvectors and eigenvalues of the covariance matrix
            eig_vals, eig_vecs = np.linalg.eig(self.P_k[i:i+2, i:i+2])
            
            # Calculate the angle of the eigenvector with the largest eigenvalue
            angle = np.arctan2(eig_vecs[1, np.argmax(eig_vals)], eig_vecs[0, np.argmax(eig_vals)])
            
            # Draw the mean and covariance ellipse
            ui.draw_point(self.s_k[i:i+2], color='g')
            ui.draw_ellipse(self.s_k[i:i+2], eig_vals[0], eig_vals[1], angle=angle, color='b')
        
        # Draw a polygon between the points
        ui.draw_polygon(self.s_k.reshape(4, 2), color='r', closed=True, linestyle='-')
    
    # Draw the state on the UI
    def draw_state(self, mean, cov, ui):
        for i in range(0, self.s_dim, 2):
            # For now average the variance of the x and y components
            avg_var = np.mean([cov[i, i], cov[i+1, i+1]])
            
            # Find the eigenvectors and eigenvalues of the covariance matrix
            eig_vals, eig_vecs = np.linalg.eig(cov[i:i+2, i:i+2])
            
            # Calculate the angle of the eigenvector with the largest eigenvalue
            angle = np.arctan2(eig_vecs[1, np.argmax(eig_vals)], eig_vecs[0, np.argmax(eig_vals)])
            
            # Draw the mean point and covariance ellipse
            ui.draw_point(mean[i:i+2], color='g')
            ui.draw_ellipse(mean[i:i+2], eig_vals[0], eig_vals[1], angle=angle, color='b')
            
            
    def get_mean(self):
        return self.s_k
    
    def get_covariance(self):
        return self.P_k

if __name__ == '__main__':
    # Initial state vector, uncertainty and time
    s = np.array([0, 0])
    P = np.diag([0.1, 0.1])
    std_dev = 0.1
    P_vec = P.flatten()

    # Create KF object
    kf = VectorizedStaticKalmanFilter(s, P, std_dev)
    kf_vec = VectorizedStaticKalmanFilter(s, P_vec, std_dev)

    # Create measurement vector
    z = np.array([0.1, 0.1])
    z2 = np.array([0.2, 0.2])

    # Update KF
    kf.update(z)
    kf.update(z2)
    kf_vec.update_vectorized(z)
    kf_vec.update_vectorized(z2)

    # Print results
    print("Normal KF:")
    print("s_k:", kf.s_k)
    print("P_k:", kf.P_k)

    print()

    # Print results
    print("Vectorized KF:")
    print("s_k:", kf.s_k)
    print("P_k:", kf.P_k)