import numpy as np
from shapely import Polygon, Point, LineString
from utils import wrap_angle, wrapped_angle_diff, angle_in_interval
from typing import Union
from scipy.spatial import KDTree

class OOI:
    def __init__(self, ui, corner_locations, std_dev=0.5,
                 car_max_range=20.0, car_max_bearing=45.0):
        self.mean = np.mean(corner_locations, axis=0)
        self.corners = corner_locations
        self.ui = ui
        self.std_dev = std_dev
        
        self.max_range = car_max_range # meters
        self.max_bearing = np.radians(car_max_bearing) # radians from degrees
        
        # Calculate the collision polygon
        self.collision_polygon = self.get_collision_polygon()
        self.soft_collision_polygon = None # This is set by main class
        self.soft_collision_points = None # This is set by main class
        
        # Maintain a kd-tree for quick nearest neighbor queries
        self.tree = KDTree(self.corners)
        
    def get_corners(self):
        return self.corners
    
    # Return the collision polygon for the OOI
    def get_collision_polygon(self) -> Polygon:
        return Polygon(self.corners)
        
    def draw(self):
        # Close rectangle and draw it
        self.ui.draw_polygon(self.corners, color='r', closed=True, alpha=0.5)
        
        # Draw soft collision polygon if available
        # self.ui.draw_polygon(self.soft_collision_points, color='b', closed=True, linestyle='--', alpha=0.2)
    
    # Return the corners of the OOI which are observable from the car (observation vectors do not intersect walls)
    def get_observation(self, car_state, corners=None, draw=True):
        # If corners are not provided, use the corners of the OOI
        if corners is None:
            corners = self.corners
        
        # Get needed parameters from car
        car_position = car_state[0:2]
        car_yaw = car_state[3]
        car_range = self.max_range
        
        # Do initial distance check to see if we can see any corners
        # If not, return empty list (avoid needless computation)
        distances = np.linalg.norm(corners - car_position, axis=1)
        if np.all(distances >= car_range):
            return np.array([]), []
        
        # Filter out observations of corners that are out of sensor fov
        # Calculate bearing to each corner
        bearings = np.arctan2(corners[:, 1] - car_position[1], corners[:, 0] - car_position[0])
        
        # Old non-vectorized code
        # corner_in_fov = np.full(corners.shape[0], False, dtype=bool) # corners x 1 length boolean array for fov check
        # for i, bearing in enumerate(bearings):
        #     # If the absolute value of the angle difference is less than the max sensor fov, the corner is in fov
        #     corner_in_fov[i] = np.abs(wrapped_angle_diff(bearing, car_yaw)) < self.max_bearing
        
        # Vectorized check for corners in fov
        corner_in_fov = np.abs(wrapped_angle_diff(bearings, car_yaw)) < self.max_bearing
        
        #### Find what corners are occluded by front of object using shapely
        # Create line of sight from the car center to each corner of the OOI
        lines_of_sight = [LineString([car_position, Point(corner)]) for corner in corners]
        
        # Check if each line of sight cross the OOI (which means it is not observable)
        ooi_polygon = Polygon(corners)
        not_intersect_ooi = np.array([not line.crosses(ooi_polygon) for line in lines_of_sight])
        
        # Get the usable corners by taking the element-wise AND of the two boolean arrays
        observable_corner_idx = np.logical_and(corner_in_fov, not_intersect_ooi)
        
        # Take only the rows (corners) that are not occluded
        observable_corners = corners[observable_corner_idx == True]
        
        if draw:
            # Draw observable corners
            for corner in observable_corners:
                self.ui.draw_circle(corner, 0.6, color='g')
        
        return observable_corners, np.where(observable_corner_idx)[0]
    
    # Call get_observation and add noise to the observation
    def get_noisy_observation(self, car_state, corners=None, draw=True):
        observable_corners, indeces = self.get_observation(car_state, corners=corners, draw=False)
        
        # Add noise to the observation
        mean = 0
        noisy_observable_corners = observable_corners + np.random.normal(mean, self.std_dev, observable_corners.shape)
        
        if draw:
            # Draw noisy observable corners
            for corner in noisy_observable_corners:
                self.ui.draw_circle(corner, 0.6, color='g')
        
        return noisy_observable_corners, indeces
    
    # Get simulated observation quickly using kd-tree
    def get_fast_simulated_observation(self, car_state, corners=None, draw=False):
        # If corners are not provided, use the class kd-tree
        if corners is None:
            tree = self.tree
        else:
            tree = KDTree(corners)
            
        # Get car position
        car_position = car_state[0:2]
        car_yaw = car_state[3]
        
        # Find the 8 (from two closest rectangles) nearest points to the car within sensor range
        tree_query = tree.query(car_position, k=8, distance_upper_bound=self.max_range)

        # Construct an array of rows (x,y) for each point (ordered by smallest distance first)
        non_inf_len = np.shape(tree_query[0][tree_query[0] != np.inf])[0] # Length of non-inf points
        xy_idx_points = np.hstack((tree.data[tree_query[1][:non_inf_len]], tree_query[1][:non_inf_len].reshape(-1, 1)))
        
        # Filter out points that are out of sensor fov
        # Calculate bearing to each point
        point_bearings = np.arctan2(xy_idx_points[:, 1] - car_position[1], xy_idx_points[:, 0] - car_position[0])
        
        # If the absolute value of the angle difference is less than the max sensor fov, the point is in fov
        points_in_fov = []
        got_first_point = False
        for i, bearing in enumerate(point_bearings):
            # Only append points if the absolute value of the angle difference is less than the max sensor fov
            if np.abs(wrapped_angle_diff(bearing, car_yaw)) <= self.max_bearing:
                points_in_fov.append(xy_idx_points[i])
                got_first_point = True
        
        # Convert to numpy array
        points_in_fov = np.array(points_in_fov)
        
        # If there are no points return an empty array
        if not got_first_point:
            return np.array([]), np.array([])
        
        # If there is 4 or more points in fov, return only the first 3 (The 4th point should be occluded by front of object)
        if points_in_fov.shape[0] >= 4:
            points_in_fov = points_in_fov[:3]
        
        if draw:
            # Draw the points in fov
            for point in points_in_fov:
                self.ui.draw_circle(point[:2], 0.6, color='g')
        
        # Order based on indeces in third column
        points_in_fov = points_in_fov[points_in_fov[:, 2].argsort()]
        
        # Take out indeces and convert to int array
        obs_indeces = points_in_fov[:, 2].astype(int)
        
        # Return the points and indeces
        return points_in_fov[:, :2], obs_indeces
        
if __name__ == '__main__':
    ooi = OOI(None)
    
    print(ooi.corners)
    
    for corner in ooi.corners:
        print(corner)