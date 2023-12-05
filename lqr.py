import numpy as np
from numpy.linalg import inv

class MeasurementLQRController:
    def __init__(self, start_state:tuple, goal_state:tuple, H = 10, c0=0.05, c1=0.1, c2=0.3, dt=1, 
                 vmin=0, vmax=20, amin=-5, amax=12):
        # Cost constants
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        
        # State and control limits
        self.vmin_real = vmin
        self.vmax_real = vmax
        self.amin = amin
        self.amax = amax
        
        self.H = H  # Length of horizon (steps into future to optimize for)
        
        # Time resolution
        self.dt = dt
        
        self.x0 = np.array([start_state[0], start_state[1]]).T  # Initialize the state vector [x, v]
        self.xg = np.array([goal_state[0], goal_state[1]]).T  # Initialize the goal state vector [x, v]
        
        self.ug = 0  # Initialize the goal control input (input which causes the agent to stay in the goal state)
        
        self.B = np.array([[0], [self.dt]])             # Control to state matrix
        self.Q = np.array([[self.c0, 0], [0, self.c2]])   # State cost matrix
        self.R = np.array([[self.c1]])                  # Control cost matrix
        
        # Track the current state and control input
        self.x = self.x0 - self.xg
        self.u = None
        
        # Velocity min and max values must also be adjusted by the goal state
        self.vmin = self.vmin_real - self.xg[1]
        self.vmax = self.vmax_real - self.xg[1]
        
        self.x_real = self.x0  # Track the real state of the system (not subtracted by state goal)
        
        # Track the list of states and controls
        self.x_u_list = [(self.x_real[0], self.x_real[1], self.u)]

    # Adaptive State transistion matrix
    def A_k(self, O):
        pass
        

    def Q_k(self, k):
        # If k is greater than H, horizion, then return 0
        print('k: ', k)
        if k >= self.H:
            return np.zeros((2, 2))
        
        # If this is the terminal state then the cost to go is 0
        Q_k_prior = self.Q_k(k+1)
        Q_k_post = (self.Q + self.A.T @ Q_k_prior @ self.A - self.A.T @ Q_k_prior @ self.B @ 
                    inv(self.R + self.B.T @ Q_k_prior @ self.B) @ self.B.T @ Q_k_prior @ self.A)
        
        return Q_k_post
        
    def K_k(self, k):
        Q_k_plus_1 = self.Q_k(k+1)
        
        return -inv(self.R + self.B.T @ Q_k_plus_1 @ self.B) @ self.B.T @ Q_k_plus_1 @ self.A

    def update(self, k):
        u = self.K_k(k) @ self.x # 1x1 matrix
        self.u = np.array([sat_value(u[0], self.amin, self.amax)])
        
        self.x = self.A @ self.x + self.B @ self.u
        self.x[1] = sat_value(self.x[1], self.vmin, self.vmax)
        
        # self.x_real = self.x + self.xg
        self.x_real = self.A @ self.x_real + self.B @ self.u
        self.x_real[1] = sat_value(self.x_real[1], self.vmin_real, self.vmax_real)
        
        self.x_u_list.append((self.x_real[0], self.x_real[1], self.u[0]))
        
        
def sat_value(value, min_value, max_value):
    return min(max_value, max(min_value, value))

# Initialize the LQR Controller class with the given A, B, Q, R matrices and K_values
lqr = MeasurementLQRController((-100, 10), (0, 0), H=100)

# Simulate for K steps
for k in range(100):
    lqr.update(k)
    
    # If the l2 norm of the state is less than 1e-2, then we have reached the terminal state
    if np.linalg.norm(lqr.x) < 1e-2:
        print('Terminal state reached at step: ', k)
        break
    
print("Printing the list of states and controls: ")
for xu in lqr.x_u_list:
    print(xu)
    
lqr.plot()
