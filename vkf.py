import numpy as np
from math import *

class VectorizedStaticKalmanFilter:
    def __init__(self, _s, _P, p_dev):
        # Load noise scaling values
        self.p_dev = np.array(p_dev)

        # Dimension of state and observation vectors
        self.s_dim = 8
        self.z_dim = 8

        # Load in the initial state vector and covariance matrix
        self.s_k = _s
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
            col_index = (index + 1) * 2 - 2     # Function mapping index of observation to index of diagonal point list of (x,y)
            
            H[2 * i, col_index] = 1             # Assignment to x portion
            H[2 * i + 1, col_index+1] = 1       # Assignment to y portion
        return H
    
    # Measurement noise, size needs to be 2 * number of observations for x and y (row length = column length)
    def R(self, obs_indices):
        return self.p_dev**2 * np.eye(len(obs_indices) * 2)
    
    # Update step of KF, get posterior distribution, function of measurement and sensor
    def update(self, z, obs_indices):
        S_k = self.H(obs_indices) @ self.P_k @ self.H(obs_indices).T + self.R(obs_indices)
        K_k = self.P_k @ self.H(obs_indices).T @ np.linalg.inv(S_k)

        z_bar = z - self.H(obs_indices) @ self.s_k

        self.s_k = self.s_k + K_k @ z_bar
        self.P_k = (np.eye(self.s_dim) - K_k @ self.H(obs_indices)) @ self.P_k

    # Update step of KF, but with vectorized matrices
    def update_vectorized(self, z, obs_indices):
        S_k = self.H(obs_indices) @ self.P_k.reshape((self.s_dim,self.s_dim)) @ self.H(obs_indices).T + self.R(obs_indices)
        K_k = self.P_k.reshape((self.s_dim,self.s_dim)) @ self.H(obs_indices).T @ np.linalg.inv(S_k)

        z_bar = z - self.H(obs_indices) @ self.s_k
        
        # Create kronecker product matrix which allows for vectorized matrix operations
        O = np.kron(np.eye(self.s_dim) - K_k @ self.H(obs_indices), np.eye(self.s_dim))

        self.s_k = self.s_k + K_k @ z_bar
        self.P_k = O @ self.P_k
        
    # Draw the mean and covariance on the UI
    def draw(self, ui):
        for i in range(0, self.s_dim, 2):
            # For now average the variance of the x and y components
            avg_var = np.mean([self.P_k[i, i], self.P_k[i+1, i+1]])
            
            ui.draw_point(self.s_k[i:i+2], color='g')
            ui.draw_circle(self.s_k[i:i+2], avg_var, color='g')

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