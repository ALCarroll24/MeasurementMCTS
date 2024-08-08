import numpy as np
from scipy.spatial import KDTree
from copy import deepcopy
from car import Car
from utils import min_max_normalize
from ui import MatPlotLibUI

class KDTreeEvaluation:
    def __init__(self, oois: list, num_steps: int, dt: float, max_range: float = 40, std_devs: int = 3,
                 corner_rew_max: float = 3000, obs_rew_min: float = 100, corner_reward_scale: float = 1.,
                 obs_pun_scale: float = 1, ui: MatPlotLibUI = None):
        # Save parameters
        self.num_steps = num_steps
        self.dt = dt
        self.max_range = max_range
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
    def get_evaluation_reward(self, state):
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
        
        # Now we can calculate the reward based on the distances to the corners
        corner_reward = self.corner_reward_scale * np.sum(corner_dists**2) # * corner_traces
        
        # Only take the points that are within the obstacle radii
        close_obs_indices = obs_indices[obs_dists < self.obstacle_radii[obs_indices]]
        close_obs_dists = obs_dists[obs_dists < self.obstacle_radii[obs_indices]]
        
        # And the punishment for being within the obstacle radii
        obs_reward = -self.obs_pun_scale * np.sum((self.obstacle_radii[close_obs_indices] - close_obs_dists)**2)
        
        # Normalize the rewards to [-1, 0] and [0, 1] based on expected maxes and mins
        max_clipped_corner_reward = min(corner_reward, self.corner_rew_max) # Clip the corner reward to max
        norm_corner_reward = min_max_normalize(max_clipped_corner_reward, 0, self.max_range**2) # Normalize to [0, 1]
        flipped_corner_reward = 1 - norm_corner_reward # Make into a reward (higher number was worse before)
        
        min_clipped_obs_reward = max(obs_reward, -self.obs_rew_min) # Clip the obs reward to min
        norm_obs_reward = min_max_normalize(min_clipped_obs_reward, -self.obs_rew_min, 0) # Normalize to [-1, 0]
        
        # print(f'Num corners in range: {len(corner_indices)}')
        # print(f'Num obstacles in range: {len(close_obs_indices)}')
        # print(f'Corner reward: {corner_reward}')
        # print(f'Obstacle reward: {obs_reward}')
        # print(f'Normalized corner reward: {norm_corner_reward}')
        # print(f'Normalized obstacle reward: {norm_obs_reward}')
        # print(f'Total reward: {flipped_corner_reward + norm_obs_reward}')
        # print()
        
        # Return the reward
        return flipped_corner_reward + norm_obs_reward
    
    # Given a car state and action, apply that action repetitively and evaluate the reward
    def evaluate(self, action, car_state, draw=False):
        # Make a deep copy of the car
        state = deepcopy(car_state)
        
        # Maintain cumulative reward
        cumulative_reward = 0.0

        # Loop and apply actions
        for i in range(self.num_steps):
            # Get new car state with starting state and action (does not update car object)
            state = self.car.update(self.dt, action, starting_state=state)

            # Calculate the reward
            reward = self.get_evaluation_reward(state)
            cumulative_reward += reward
            
            # Draw the state if draw is True with size based on reward
            if draw:
                self.ui.draw_circle(state[:2], reward, color='b')
                # self.ui.draw_text(f'{action}', state[:2] + np.array([-0.6, 0.3]), color='black', fontsize=12)
            
                print(f'i={i} KD Reward={reward}')
        print()
        # Average and return the cumulative reward
        return cumulative_reward / self.num_steps
            
        