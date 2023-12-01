import numpy as np
from math import *

class VectorizedStaticKalmanFilter:
    def __init__(self, _s, _P, p_dev):
        # Load noise scaling values
        self.p_dev = np.array(p_dev)

        # Dimension of state and observation vectors
        self.s_dim = 2
        self.z_dim = 2

        # Load in the initial state vector, uncertainty and time
        self.s_k = _s
        self.P_k = _P
        
    # State transition matrix, 2 dimensional identity matrix
    def F(self):
        # Start with a two dimensional identity matrix
        F = np.eye(self.s_dim)

        return F
    
    # Maps state space to measurement space,
    # self.z_dim dimensional identity matrix since state is fully observable
    def H(self):
        H = np.eye(self.z_dim)

        return H
    
    # Measurement noise (currently constant)
    def R(self):
        return self.p_dev**2 * np.eye(self.z_dim)
    
    # Update step of KF, get posterior distribution, function of measurement and sensor
    def update(self, z):
        S_k = self.H() @ self.P_k @ self.H().T + self.R()
        K_k = self.P_k @ self.H().T @ np.linalg.inv(S_k)

        z_bar = z - self.H() @ self.s_k

        self.s_k = self.s_k + K_k @ z_bar
        self.P_k = (np.eye(self.s_dim) - K_k @ self.H()) @ self.P_k

    # Update step of KF, but with vectorized matrices
    def update_vectorized(self, z):
        S_k = self.H() @ self.P_k.reshape((self.s_dim,self.s_dim)) @ self.H().T + self.R()
        K_k = self.P_k.reshape((self.s_dim,self.s_dim)) @ self.H().T @ np.linalg.inv(S_k)

        z_bar = z - self.H() @ self.s_k
        
        # Create kronecker product matrix which allows for vectorized matrix operations
        O = np.kron(np.eye(self.s_dim) - K_k @ self.H(), np.eye(self.s_dim))

        self.s_k = self.s_k + K_k @ z_bar
        self.P_k = O @ self.P_k

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