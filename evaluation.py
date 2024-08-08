import numpy as np
from scipy.spatial import KDTree
from copy import deepcopy
from car import Car
from utils import min_max_normalize, wrapped_angle_diff
from ui import MatPlotLibUI

class KDTreeEvaluation:
    def __init__(self, oois: list, num_steps: int, dt: float, max_range: float = 40, max_bearing: float = np.pi/3,
                  std_devs: int = 3, corner_rew_max: float = 3000, obs_rew_min: float = 100,
                  corner_reward_scale: float = 1., obs_pun_scale: float = 1, ui: MatPlotLibUI = None):
        # Save parameters
        self.num_steps = num_steps
        self.dt = dt
        self.max_range = max_range
        self.max_bearing = np.radians(max_bearing)
        self.corner_rew_max = corner_rew_max
        self.obs_rew_min = obs_rew_min
        self.corner_reward_scale = corner_reward_scale
        self.obs_pun_scale = obs_pun_scale
        self.ui = ui
        
        # Create a car to use simulation update (state initialization does nothing here)
        self.car = Car(None, np.zeros(6))
        
        # Create needed vectors
        kd_tree_pts = np.zeros((len(oois) * 5, 2)) # 4 corner points + mean obstacle point rows and (x, y) columns
        obstacle_radii = np.zeros((len(oois), 1)) # Radius of obstacle (based on std deviation)
        
        # First calculate circular obstacles at mean of oois
        for i, ooi in enumerate(oois):
            mean = np.mean(ooi, axis=0) # Calculate mean of ooi corners
            std_dev = std_devs * np.std(ooi, axis=0) # Calculate std deviation of ooi corners scaled by std_devs
            
            # Add obstacle mean to points for kd tree (beginning is for obstacles)
            kd_tree_pts[i, :] = mean
            obstacle_radii[i] = np.linalg.norm(std_dev) # Calculate radius of obstacle
            
        # Add ooi corner points to kd tree
        # Stack the oois list into a numpy array
        ooi_stacked = np.vstack(oois)
        kd_tree_pts[len(oois):, :] = ooi_stacked

        # Create a KDTree from the points
        self.kd_tree = KDTree(kd_tree_pts)
        
        # Flatten obstacle radii
        self.obstacle_radii = obstacle_radii.flatten()

    # Get nearest points within a radius of a point from kdtree
    def get_nearest_points(self, tree, point, radius):
        # Find all points within the radius of the point (returns points sorted by distance and their indeces)
        distances, indeces = tree.query(point, k=8, distance_upper_bound=radius)
        
        # Find the length of the non-infinite points
        non_inf_len = np.shape(distances[distances != np.inf])[0] # Length of non-inf points
        
        # Remove points with infinite distance (which have index larger than kd tree size)
        distances = distances[:non_inf_len]
        indeces = indeces[:non_inf_len]
        
        return distances, indeces

    # Query kd tree to find closest points, then calculate reward based on distances to corners and obstacles
    def get_evaluation_reward(self, state, corner_cov):
        # Query the KDTree to find points within the car max range
        dists, indeces = self.get_nearest_points(self.kd_tree, state[:2], self.max_range)
        
        # Order the distances and indeces to have increasing index
        ordered_dists = dists[np.argsort(indeces)]
        ordered_indeces = indeces[np.argsort(indeces)]
        
        # Now find the number of obstacle points in the near indeces
        obs_pts_count = np.sum(ordered_indeces < len(self.obstacle_radii)) # returns number of True values
        
        # Now we can split the ordered indeces and distances into obstacle and corner vectors
        obs_indices = ordered_indeces[:obs_pts_count]
        obs_dists = ordered_dists[:obs_pts_count]
        corner_indices = ordered_indeces[obs_pts_count:]
        corner_dists = ordered_dists[obs_pts_count:]
        corner_pts = self.kd_tree.data[corner_indices]``
        
        # Now remove any points that are out of the sensor fov
        corner_bearings = np.arctan2(corner_pts[:, 1] - state[1], corner_pts[:, 0] - state[0]) # Calculate bearing to each point
        in_fov_bools = np.abs(wrapped_angle_diff(corner_bearings, state[3])) < self.max_bearing # Check if relative bearing is within fov
        in_fov_indeces = corner_indices[in_fov_bools] # Only take the indeces that are in fov
        in_fov_corner_dists = corner_dists[in_fov_bools] # Only take the distances that are in fov
        
        # Now we can calculate the reward based on the distances to the corners
        corner_reward = self.corner_reward_scale * np.sum(in_fov_corner_dists**2) # * corner_traces
        
        # Only take the points that are within the obstacle radii
        close_obs_indices = obs_indices[obs_dists < self.obstacle_radii[obs_indices]]
        close_obs_dists = obs_dists[obs_dists < self.obstacle_radii[obs_indices]]
        
        # And the punishment for being within the obstacle radii
        obs_reward = -self.obs_pun_scale * np.sum((self.obstacle_radii[close_obs_indices] - close_obs_dists)**2)
        
        # Normalize the rewards to [-1, 0] and [0, 1] based on expected maxes and mins
        clipped_corner_reward = min(corner_reward, 4 * self.max_range**2) # Clip the corner reward to max (should never actually reach this)
        norm_corner_reward = min_max_normalize(clipped_corner_reward, 0, 4 * self.max_range**2) # Normalize to [0, 1]
        flipped_corner_reward = 1 - norm_corner_reward # Make into a reward (higher number was worse before)
        
        min_clipped_obs_reward = max(obs_reward, -self.obs_rew_min) # Clip the obs reward to min
        norm_obs_reward = min_max_normalize(min_clipped_obs_reward, -self.obs_rew_min, 0) # Normalize to [-1, 0]
        
        print(f'Num corners in range: {len(in_fov_indeces)}')
        print(f'Num obstacles in range: {len(close_obs_indices)}')
        print(f'Corner reward: {corner_reward}')
        print(f'Obstacle reward: {obs_reward}')
        print(f'Normalized corner reward: {flipped_corner_reward}')
        print(f'Normalized obstacle reward: {norm_obs_reward}')
        # print(f'Total reward: {flipped_corner_reward + norm_obs_reward}')
        # print()
        
        # Return the reward
        return flipped_corner_reward + norm_obs_reward
    
    # Given a car state and action, apply that action repetitively and evaluate the reward
    def evaluate(self, action, car_state, corner_cov, draw=False):
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
            cumulative_reward += reward
            
            # Draw the state if draw is True with size based on reward
            if draw:
                self.ui.draw_circle(state[:2], reward, color='b')
                # self.ui.draw_text(f'{action}', state[:2] + np.array([-0.6, 0.3]), color='black', fontsize=12)
            
                print(f'i={i} KD Reward={reward}')
            print()
        print()
        # Average and return the cumulative reward
        return cumulative_reward / self.num_steps

    def draw_obstacles(self):
        for i, radius in enumerate(self.obstacle_radii):
            self.ui.draw_circle(self.kd_tree.data[i], radius, color='y')
            
    def get_obstacle_cost(self, car_state):
        # Query the KDTree to find points within the up to the max obstacle radius
        dists, indeces = self.get_nearest_points(self.kd_tree, car_state[:2], self.obstacle_radii.max())
        
        # Order the distances and indeces to have increasing index
        ordered_dists = dists[np.argsort(indeces)]
        ordered_indeces = indeces[np.argsort(indeces)]
        
        # Now find the number of obstacle points in the near indeces
        obs_dists = ordered_dists[ordered_indeces < len(self.obstacle_radii)]
        obs_indeces = ordered_indeces[ordered_indeces < len(self.obstacle_radii)]
        
        # Only take the points that are within the obstacle radii
        is_offending_obs = obs_dists < self.obstacle_radii[obs_indeces]
        offending_obs_dists = obs_dists[is_offending_obs]
        offending_obs_indices = obs_indeces[is_offending_obs]
        offending_obs_radii = self.obstacle_radii[offending_obs_indices]
        
        # And the punishment for being within the obstacle radii
        obs_reward = -self.obs_pun_scale * np.sum((offending_obs_radii - offending_obs_dists)**2)
        print(f'Number of obstacles in range: {len(offending_obs_indices)}')
        print(f'Distances to offending obstacles: {offending_obs_dists}')
        
        return obs_reward