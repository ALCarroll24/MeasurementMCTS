import numpy as np
from measurement_mcts.utils.utils import wrap_angle, min_max_normalize, rotate_about_point, angle_difference

class HERTG:
    def __init__(self, state, env, method, reward_scale=0.1):
        self.env = env
        self.method = method
        
        # Scale the reward based on the horizon length in the environment
        self.reward_scale = reward_scale / env.horizon_length
        
        # Initialize the best ooi index
        self.set_best_ooi(state)
        
        self.root_state = state
        
        if self.method == 'static':
            self.target_point = get_hertg_past_ooi_target_point(state, self.env, self.best_ooi_idx)
            
        elif self.method == 'dynamic':
            self.target_point = get_hertg_dynamic_target_point(state, self.env, self.best_ooi_idx)
        
        else:
            raise ValueError('Invalid HERTG method')
            
        
    def get_reward(self, state):
        return get_distance_reward(state, self.target_point, self.root_state, scale=self.reward_scale)
    
    def update_root_state(self, state):
        self.root_state = state
    
    def update_best_ooi(self, state):
        # Check if the ooi is completed and update the best ooi
        if self.check_ooi_completion(state) is True:
            self.set_best_ooi(state)
            
            # Static is only updated if the ooi is completed
            if self.method == 'static':
                self.target_point = get_hertg_past_ooi_target_point(state, self.env, self.best_ooi_idx)
                
        # Dynamic target point is updated even if the ooi is not completed
        if self.method == 'dynamic':
            self.target_point = get_hertg_dynamic_target_point(state, self.env, self.best_ooi_idx)

    def draw_target_point(self, state):
        if self.target_point is not None:
            self.env.ui.draw_point(self.target_point, 'green', radius=0.25)
        
    def set_best_ooi(self, state, dist_weight=0.1, bearing_weight=2):
        # Pull out state elements
        car_state, ooi_means, ooi_covs, horizon = state
        car_pos, car_yaw = car_state[:2], car_state[3]
        
        # Compute centers of each ooi and their range and bearing to the car
        ooi_centers = np.mean(ooi_means, axis=1)
        car_to_center_dists = np.linalg.norm(ooi_centers - car_pos, axis=1)
        car_to_center_bearings = np.abs(wrap_angle(np.arctan2(ooi_centers[:, 1] - car_pos[1], ooi_centers[:, 0] - car_pos[0]) - car_yaw))
        
        # Compute covariance trace sums for each ooi and normalize
        ooi_trace_sums = np.trace(ooi_covs, axis1=2, axis2=3)
        ooi_not_fully_observed = np.any(ooi_trace_sums > self.env.final_corner_cov_trace, axis=1).astype(int)

        # Mask off fully observed oois by giving them infinite cost
        ooi_mask = np.where(ooi_not_fully_observed == 0, np.inf, ooi_not_fully_observed)

        # Compute cost for each ooi and mask off fully observed oois
        ooi_cost = ooi_mask * (dist_weight * car_to_center_dists + bearing_weight * car_to_center_bearings)
        
        # Pick the lowest cost ooi as the best one
        self.best_ooi_idx = np.argmin(ooi_cost)
        
    def check_ooi_completion(self, state):
        best_ooi_covs = state[2][self.best_ooi_idx]
        traces = np.trace(best_ooi_covs, axis1=1, axis2=2)
        
        if np.any(traces > self.env.final_corner_cov_trace):
            return False
        else:
            return True
        
def get_velocity_vector_reward(state, target_point, scale=0.01):
    """
    Compute the reward for the HERTG algorithm based on the velocity and angle to the target point.
    params:
        state: tuple - the state of the car
        env: object - the environment object
        target_point: np.array - the target point to use for the reward
        scale: float - the scaling factor for the reward
    returns:
        float - the heuristic reward
    """
    # Now compute the reward based on distance and angle to the target point
    car_pos = state[0][:2]
    car_yaw = state[0][3]
    car_vel = state[0][2]
    car_steering_angle = state[0][4]
    
    # Compute vector from car to target point
    car_to_target_point = target_point - car_pos
    unit_car_to_target_point = car_to_target_point / np.linalg.norm(car_to_target_point)
    
    # Compute velocity velocity vector with direction of car tires
    if car_vel >= 0:
        velocity_vector = car_vel * np.array([np.cos(car_yaw + car_steering_angle), np.sin(car_yaw + car_steering_angle)])
        
    # Account for vehicle travelling velocity vector in reverse
    else:
        velocity_vector = car_vel * np.array([np.cos(car_yaw - car_steering_angle), np.sin(car_yaw - car_steering_angle)])
        # punish travelling in reverse somewhat
        velocity_vector = 0. * velocity_vector
    
    # Take the dot product to get the heuristic reward
    return scale * unit_car_to_target_point @ velocity_vector

def get_distance_reward(state, target_point, root_state, scale=0.01, max_distance=30):
    """
    Compute the reward for the HERTG algorithm based on the distance to the target point.
    params:
        state: tuple - the state of the car
        env: object - the environment object
        target_point: np.array - the target point to use for the reward
        scale: float - the scaling factor for the reward
    returns:
        float - the heuristic reward
    """
    # Now compute the reward based on distance and angle to the target point
    car_pos = state[0][:2]
    
    # Compute the distance to the target point
    distance_to_target = np.linalg.norm(target_point - car_pos)
    print(f'Distance to target: {distance_to_target}')
    
    # Calculate distance from root to target
    root_pos = root_state[0][:2]
    distance_from_root = np.linalg.norm(target_point - root_pos)
    print(f'Distance from root to target: {distance_from_root}')
    
    # Reward based on only decreasing the distance from root to target
    distance_reduced = np.clip(distance_from_root - distance_to_target, 0, max_distance)
    print(f'Distance reduced: {distance_reduced}')
    
    # Return the scaled distance as the reward
    reward = min_max_normalize(distance_reduced, 0, max_distance)
    print(f'Reward: {reward}')
    reward = scale * reward
    print(f'Scaled reward: {reward}')
    return reward
    
def get_hertg_dynamic_target_point(state, env, best_ooi_idx, lookahead_distance=25, ooi_circle_space=4, draw=False):
    car_pos, car_yaw = state[0][:2], state[0][3]
    ooi_means = state[1]
    ooi_centers = np.mean(ooi_means, axis=1)
    
    # Project the target point from car to best ooi
    best_ooi_center = ooi_centers[best_ooi_idx]
    car_to_best_ooi = best_ooi_center - car_pos
    car_to_best_ooi_norm = np.linalg.norm(car_to_best_ooi)
    unit_car_to_best_ooi = car_to_best_ooi / car_to_best_ooi_norm
    
    # Cap the lookahead distance to the distance to the best ooi
    lookahead_distance = min(lookahead_distance, car_to_best_ooi_norm)
    target_point = car_pos + lookahead_distance * unit_car_to_best_ooi
    
    # Now to check for collision calculate the radius of each ooi and model as a circle
    ooi_radii = np.linalg.norm(ooi_means - ooi_centers[:, None], axis=2)
    ooi_max_radii = np.max(ooi_radii, axis=1)
    
    # Check if the target point is near an ooi circle
    target_point_near_ooi = np.linalg.norm(target_point - ooi_centers, axis=1) < ooi_max_radii + ooi_circle_space

    # Get indices where true
    near_ooi_indices = np.where(target_point_near_ooi)[0]
    
    # Angle to iterate with
    angle_iter = np.radians(5)
    
    if len(near_ooi_indices) > 1:
        raise ValueError('Multiple OOIs near target point')
    elif len(near_ooi_indices) == 1:
        # print('Near OOI')
        # Pick direction to rotate around based on car heading relative to ooi
        car_to_close_ooi = ooi_centers[near_ooi_indices[0]] - car_pos
        car_to_ooi_angle = np.arctan2(car_to_close_ooi[1], car_to_close_ooi[0])
        angle_diff = angle_difference(car_to_ooi_angle, car_yaw)
        if angle_diff > 0:
            angle_direction = -1
        else:
            angle_direction = 1
        
        # First place the target_point on the ooi circle
        unit_car_to_close_ooi = car_to_close_ooi / np.linalg.norm(car_to_close_ooi)
        target_point = ooi_centers[near_ooi_indices[0]] + (ooi_circle_space + ooi_max_radii[near_ooi_indices[0]]) * -unit_car_to_close_ooi
        
        # Rotate target point around ooi circle until the distance from car to target point is greater than lookahead
        while np.linalg.norm(target_point - car_pos) < lookahead_distance:
            target_point = rotate_about_point(target_point, angle_direction * angle_iter, ooi_centers[near_ooi_indices[0]])
        
        if draw is True:
            env.ui.draw_point(target_point, 'green', radius=0.25)
            
        return target_point
    
    # Now check if the target point is inside an obstacle/occlusion
    obs_means = np.vstack((env.object_manager.obstacle_means, env.object_manager.occlusion_means))
    obs_radii = np.hstack((env.object_manager.obstacle_radii, env.object_manager.occlusion_radii))
    target_point_to_obs_dists = np.linalg.norm(target_point - obs_means, axis=1)
    target_point_inside_obs = target_point_to_obs_dists < obs_radii + env.car_collision_radius
    collision_indices = np.where(target_point_inside_obs)[0]
    
    if len(collision_indices) > 1:
        raise ValueError('Multiple obstacles/occlusions near target point')
    elif len(collision_indices) == 1:
        # print('Inside Obstacle/Occlusion')
        # Pick direction based on target_point heading relative to obstacle
        car_to_close_obs = obs_means[collision_indices[0]] - car_pos
        car_to_obs_angle = np.arctan2(car_to_close_obs[1], car_to_close_obs[0])
        car_to_target_point = target_point - car_pos
        car_to_target_point_angle = np.arctan2(car_to_target_point[1], car_to_target_point[0])
        angle_diff = angle_difference(car_to_obs_angle, car_to_target_point_angle)
        if angle_diff > 0:
            angle_direction = -1
        else:
            angle_direction = 1
        
        # First place the target_point on the obstacle circle
        unit_car_to_close_obs = car_to_close_obs / np.linalg.norm(car_to_close_obs)
        target_point = obs_means[collision_indices[0]] + (obs_radii[collision_indices[0]] + env.car_collision_radius) * -unit_car_to_close_obs
        
        # Rotate target point around ooi circle until the distance from car to target point is greater than lookahead
        while np.linalg.norm(target_point - car_pos) < lookahead_distance:
            target_point = rotate_about_point(target_point, angle_direction * angle_iter, obs_means[collision_indices[0]])
        
        if draw is True:
            env.ui.draw_point(target_point, 'green', radius=0.25)
            
        return target_point

    if draw is True:
        env.ui.draw_point(target_point, 'green', radius=0.25)
        
    return target_point

def get_hertg_past_ooi_target_point(state, env, best_ooi_idx, spacing=-2, draw=False):
    """
    Get the target point behind the best ooi based on the spacing provided.
    params:
        state: tuple - the state of the car
        env: object - the environment object
        best_ooi_idx: int - the index of the best ooi
        spacing: float - the spacing behind the best ooi
        draw: bool - whether to draw the target point
    """
    car_pos, car_yaw = state[0][:2], state[0][3]
    ooi_means = state[1]
    
    # Get the center and radius of the best ooi
    best_ooi = ooi_means[best_ooi_idx]
    best_ooi_center = np.mean(best_ooi, axis=0)
    best_ooi_max_radius = np.max(np.linalg.norm(best_ooi - best_ooi_center, axis=1))
    
    # Project the target point from car to best ooi
    car_to_best_ooi = best_ooi_center - car_pos
    unit_car_to_best_ooi = car_to_best_ooi / np.linalg.norm(car_to_best_ooi)
    
    # Compute the target point behind the best ooi
    target_point = best_ooi_center + (best_ooi_max_radius + env.car_collision_radius + spacing) * unit_car_to_best_ooi
    
    # Draw the target point if requested
    if draw is True:
        env.ui.draw_point(target_point, 'green', radius=0.25)
        
    return target_point
    

def get_target_point_follow_action(car_state, env, target_point, no_turn_degrees=10, max_turn_degrees=20):
    car_pos = car_state[:2]
    car_yaw = car_state[3]
    steering_options = env.steering_acc_options[2:] # Take the 0 and positive options
    
    # Compute vector from car to target point
    car_to_target_point = target_point - car_pos
    car_to_target_point_angle = np.arctan2(car_to_target_point[1], car_to_target_point[0])
    
    # Compute angle difference between car heading and target point
    angle_diff = wrap_angle(car_to_target_point_angle - car_yaw)
    angle_sign = np.sign(angle_diff)
    
    # Determine action based on angle difference
    if np.abs(angle_diff) < np.radians(no_turn_degrees):
        return [1, steering_options[0]]                 # Accelerate, no turning
    elif np.abs(angle_diff) < np.radians(max_turn_degrees):
        return [1, angle_sign * steering_options[1]]    # Accelerate, turn slightly
    else:
        return [1, angle_sign * steering_options[2]]    # Accelerate, turn max
    
