import numpy as np
from scipy.spatial import KDTree
from copy import deepcopy
from car import Car
from utils import min_max_normalize, wrapped_angle_diff
from ui import MatPlotLibUI

# Class utilized for a speedier evalution of the MCTS state
# Mean point of obstacles / OOI corners are used to create a KDTree
# The kdtree is queried from the location of the car stating state to find near corners and obstacles
# The car is simulated for num_steps with the same action, each state from simulations has reward computed and then is summed
# Reward is calculated based on range and bearing to corner rewards and obstacle distance based punishments
# The reward is normalized to [-obs_rew_norm_min, 0] and [0, corner_reward_norm_max] based on expected maxes and mins
# The positive corner reward is designed to match the full environment step rewards
# Obstacle checking is done utilizing this class in the full environment step so it is already matched
class KDTreeEvaluation:
    def __init__(self, oois: np.ndarray, num_steps: int, dt: float, max_range: float = 40,
                 max_bearing: float = np.pi/3, obstacle_std_dev: int = 2.5, corner_rew_norm_max: float = 0.2,
                 obs_rew_norm_min: float = 1, range_dev: float = 1., bearing_dev: float = 0.5,
                 min_range: float = 5., min_bearing: float = 5., ui: MatPlotLibUI = None):
        
        # Simulation parameters
        self.num_steps = num_steps # Number of times to step the car model and evaluate the reward
        self.dt = dt # Time step for the car model
        
        # Sensor parameters
        self.max_range = max_range # Maximum range of the sensor
        self.max_bearing = np.radians(max_bearing) # Maximum bearing (FOV) of the sensor
        
        # Output reward range parameters [0, corner_reward_norm_max] and [-obs_rew_norm_min, 0]
        self.corner_rew_norm_max = corner_rew_norm_max # High normalization value to match full environment step rewards
        self.obs_rew_norm_min = obs_rew_norm_min # Amount to scale the obstacle punishment by
        
        # Measurement model parameters
        self.range_dev = range_dev # Scaling factor for range in reward calculation
        self.bearing_dev = bearing_dev # Scaling factor for bearing in reward calculation
        self.min_range = min_range # Minimum range for sensor model
        self.min_bearing = min_bearing # Minimum bearing for sensor model
        
        # The ui object which allows for drawing on matplotlib visualization
        self.ui = ui
        
        # Create a car to use simulation update (state initialization does nothing here)
        self.car = Car(None, np.zeros(6))
        
        # Create needed vectors
        kd_tree_pts = np.zeros((len(oois) * 5, 2)) # 4 corner points + mean obstacle point rows and (x, y) columns
        obstacle_radii = np.zeros((len(oois), 1)) # Radius of obstacle (based on std deviation)
        
        # First calculate circular obstacles at mean of oois
        for i, ooi in enumerate(oois):
            mean = np.mean(ooi, axis=0) # Calculate mean of ooi corners
            std_dev = obstacle_std_dev * np.std(ooi, axis=0) # Calculate std deviation of ooi corners scaled by std_devs
            
            # Add obstacle mean to points for kd tree (beginning is for obstacles)
            kd_tree_pts[i, :] = mean
            obstacle_radii[i] = np.linalg.norm(std_dev) # Calculate radius of obstacle
            
        # Add ooi corner points to kd tree
        # Stack the oois list into a numpy array
        ooi_stacked = np.vstack(oois)
        kd_tree_pts[len(oois):, :] = ooi_stacked

        # Create a KDTree from the points
        self.kd_tree = KDTree(kd_tree_pts)
        
        # Flatten obstacle radii, now (N,) array
        self.obstacle_radii = obstacle_radii.flatten()
        
        # Calculate the minimum possible obstacle reward (assuming obstacles are seperated and we can't be on two at once)
        self.obs_rew_min = max(self.obstacle_radii)**2
        
        # Calculate the maximum possible corner reward based on the min range and min bearing (times 4 for 4 corners)
        self.corner_rew_max = 4 * (self.range_dev**2 * self.max_range + self.bearing_dev**2 * np.pi * self.max_range * self.max_bearing)

    def get_corner_points(self):
        return self.kd_tree.data[len(self.obstacle_radii):]
    
    def get_obstacle_points(self):
        return self.kd_tree.data[:len(self.obstacle_radii)]
    
    def get_obstacle_radii(self):
        return self.obstacle_radii

    # Get nearest points within a radius of a point from kdtree
    def get_nearest_points(self, tree, point, radius):
        # Find all points within the radius of the point (returns points sorted by distance and their indices)
        distances, indices = tree.query(point, k=8, distance_upper_bound=radius)
        
        # Find the length of the non-infinite points
        non_inf_len = np.shape(distances[distances != np.inf])[0] # Length of non-inf points
        
        # Remove points with infinite distance (which have index larger than kd tree size)
        distances = distances[:non_inf_len]
        indices = indices[:non_inf_len]
        
        return distances, indices

    # Query the kd tree to find points within a radius of a point, then order by increasing index
    def query_kd_tree(self, point, radius):
        # Query the KDTree to find points within the range
        dists, indices = self.get_nearest_points(self.kd_tree, point, radius)
        
        # Order the distances and indices to have increasing index
        ordered_dists = dists[np.argsort(indices)]
        ordered_indices = indices[np.argsort(indices)]
        
        return ordered_dists, ordered_indices
    
    def get_obstacle_cost(self, car_state, ordered_dists=None, ordered_indices=None):
        # If the kdtree query was not passed, then query the kd tree
        if ordered_dists is None or ordered_indices is None:
            # Query the KDTree to find points within the largest obstacle radius
            ordered_dists, ordered_indices = self.get_nearest_points(self.kd_tree, car_state[:2], np.max(self.obstacle_radii))
        
        # Now find the number of obstacle points in the near indices
        obs_dists = ordered_dists[ordered_indices < len(self.obstacle_radii)]
        obs_indices = ordered_indices[ordered_indices < len(self.obstacle_radii)]
        
        # Only take the points that are within the obstacle radii
        is_offending_obstacle = obs_dists < self.obstacle_radii[obs_indices]
        offend_obs_dists = obs_dists[is_offending_obstacle]
        offend_obs_indices = obs_indices[is_offending_obstacle]
        offend_obs_radii = self.obstacle_radii[offend_obs_indices]
        
        # And the punishment (negative reward) for being within the obstacle radii (squared to make it ramp based on distance)
        obs_reward = np.sum((offend_obs_radii - offend_obs_dists)**2)
        
        # Normalize the obstacle rewards to [-obs_rew_norm_min, 0] based on expected maxes and mins
        min_clipped_obs_reward = min(obs_reward, self.obs_rew_min) # Clip the obs reward to min
        norm_obs_reward = -self.obs_rew_norm_min * min_max_normalize(min_clipped_obs_reward, 0, self.obs_rew_min)
        
        return norm_obs_reward

    # Query kd tree to find closest points, then calculate reward based on distances to corners and obstacles
    def get_evaluation_reward(self, car_state, point_traces):
        # Query the KDTree to find points within the car max range
        ordered_dists, ordered_indices = self.get_nearest_points(self.kd_tree, car_state[:2], self.max_range)
        
        # Get the obstacle cost, since we already queried the kd tree, pass in the ordered indices and distances
        norm_obs_reward = self.get_obstacle_cost(car_state, ordered_dists, ordered_indices)
        
        # Now find the number of obstacle points in the near indices
        obs_pts_count = np.sum(ordered_indices < len(self.obstacle_radii)) # returns number of True values
        
        # Now we can get the indices and distances which are from the corners (not obstacles)
        corner_indices = ordered_indices[obs_pts_count:]
        corner_dists = ordered_dists[obs_pts_count:]
        corner_pts = self.kd_tree.data[corner_indices]
        
        # Now remove any points that are out of the sensor fov
        corner_bearings = np.arctan2(corner_pts[:, 1] - car_state[1], corner_pts[:, 0] - car_state[0]) # Calculate bearing to each point
        rel_bearings = np.abs(wrapped_angle_diff(corner_bearings, car_state[3])) # Calculate absolute value of relative bearing difference
        in_fov_bools = rel_bearings < self.max_bearing # Check if relative bearing is within fov
        
        # If there are no corners in fov, return only obstacle reward
        if np.sum(in_fov_bools) == 0:
            return norm_obs_reward
        
        # Get the indices, distances, and bearings that are in fov
        in_fov_indices = corner_indices[in_fov_bools] # Only take the indices that are in fov
        in_fov_ranges = np.clip(corner_dists[in_fov_bools], self.min_range, None) # Only take the distances that are in fov (clip to min)
        in_fov_rel_bearings = np.clip(rel_bearings[in_fov_bools], self.min_bearing, None) # Only take the bearings that are in fov (clip to min)
        
        # Now we can calculate the reward based on the distances to the corners and trace of that corner covariance
        cov_indices = in_fov_indices - len(self.obstacle_radii) # Subtract the number of obstacles to get the correct index
        corner_reward = np.sum(point_traces[cov_indices] * # Corner point covariance trace weight
                               (self.range_dev**2 * in_fov_ranges +          # Range scaling to match measurement model
                               self.bearing_dev**2 * np.pi * in_fov_ranges * in_fov_rel_bearings))   # Bearing scaling to match measurement model
        
        # Normalize the corner rewards to [0, corner_reward_norm_max] based on expected maxes and mins
        clipped_corner_reward = min(corner_reward, self.corner_rew_max) # Clip the corner reward to max
        norm_corner_reward = self.corner_rew_norm_max * min_max_normalize(clipped_corner_reward, 0, self.corner_rew_max)
        flipped_corner_reward = self.corner_rew_norm_max - norm_corner_reward # Make into a reward (higher number was worse before)
        
        
        # Add the two rewards which is now [-obs_rew_norm_min, corner_reward_norm_max]
        return flipped_corner_reward + norm_obs_reward
    
    # Given a car state and action, apply that action repetitively and evaluate the reward
    def evaluate(self, action, car_state, corner_cov, depth, discount_factor, draw=False):
        # Make a deep copy of the car
        state = deepcopy(car_state)
        
        # Maintain cumulative reward
        cumulative_reward = 0.0

        # Loop and apply actions
        for i in range(self.num_steps):
            # Get new car state with starting state and action (does not update car object)
            state = self.car.update(self.dt, action, starting_state=state)

            # Calculate the reward
            reward = self.get_evaluation_reward(state, corner_cov)
            discounted_reward = reward * (discount_factor ** (depth + i))
            cumulative_reward += discounted_reward
            
            # Draw the state if draw is True with size based on reward
            if draw:
                self.ui.draw_circle(state[:2], 0.1, color='b')
                # self.ui.draw_text(f'{action}', state[:2] + np.array([-0.6, 0.3]), color='black', fontsize=12)
                self.ui.draw_text(f'kd: {reward:.2f}', state[:2] + np.array([-0.2, 0.2]), color='black', fontsize=12)
                self.ui.draw_arrow(state[:2], state[:2] + 0.5 * np.array([np.cos(state[3]), np.sin(state[3])]), color='b')
                
        # Average and return the cumulative reward
        return cumulative_reward / self.num_steps

    def draw_obstacles(self):
        for i, radius in enumerate(self.obstacle_radii):
            self.ui.draw_circle(self.kd_tree.data[i], radius, color='y')
