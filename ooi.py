import numpy as np
from shapely import Polygon, Point, LineString
from utils import wrap_angle, wrapped_angle_diff, angle_in_interval
from typing import Union

class OOI:
    def __init__(self, ui, corner_locations, std_dev=0.5,
                 car_max_range=20.0, car_max_bearing=45.0):
        self.mean = np.mean(corner_locations, axis=0)
        self.corners = corner_locations
        self.ui = ui
        self.std_dev = std_dev
        
        self.max_range = car_max_range
        self.max_bearing = car_max_bearing
        
        # Calculate the collision polygon
        self.collision_polygon = self.get_collision_polygon()
        self.soft_collision_polygon = None # This is set by main class
        self.soft_collision_points = None # This is set by main class
        
    def get_corners(self):
        return self.corners
    
    # Return the collision polygon for the OOI
    def get_collision_polygon(self) -> Polygon:
        return Polygon(self.corners)
        
    def draw(self):
        # Close rectangle and draw it
        self.ui.draw_polygon(self.corners, color='r', closed=True, alpha=0.5)
        
        # Draw soft collision polygon if available
        self.ui.draw_polygon(self.soft_collision_points, color='b', closed=True, linestyle='--', alpha=0.2)
    
    # Return the corners of the OOI which are observable from the car (observation vectors do not intersect walls)
    def get_observation(self, car_state, draw=True):
        # Get needed parameters from car
        car_position = car_state[0:2]
        car_range = self.max_range
        
        # Do initial distance check to see if we can see any corners
        # If not, return empty list (avoid needless computation)
        distances = np.linalg.norm(self.corners - car_position, axis=1)
        if np.all(distances >= car_range):
            return np.array([]), []
        
        # Filter out observations of corners that are out of sensor fov
        # Calculate bearing to each corner
        car_yaw = car_state[2]
        bearings = np.arctan2(self.corners[:, 1] - car_position[1], self.corners[:, 0] - car_position[0])
        corner_in_fov = np.full(self.corners.shape[0], False, dtype=bool) # corners x 1 length boolean array for fov check
        for i, bearing in enumerate(bearings):
            # If the absolute value of the angle difference is less than the max sensor fov, the corner is in fov
            corner_in_fov[i] = np.abs(wrapped_angle_diff(bearing, car_yaw)) < np.radians(self.max_bearing)
        
        #### Find what corners are occluded by front of object using shapely
        # Create line of sight from the car center to each corner of the OOI
        lines_of_sight = [LineString([car_position, Point(corner)]) for corner in self.corners]
        
        # Check if each line of sight cross the OOI (which means it is not observable)
        ooi_polygon = Polygon(self.corners)
        not_intersect_ooi = np.array([not line.crosses(ooi_polygon) for line in lines_of_sight])
        
        # Get the usable corners by taking the element-wise AND of the two boolean arrays
        observable_corner_idx = np.logical_and(corner_in_fov, not_intersect_ooi)
        
        # Take only the rows (corners) that are not occluded
        observable_corners = self.corners[observable_corner_idx == True]
        
        if draw:
            # Draw observable corners
            for corner in observable_corners:
                self.ui.draw_circle(corner, 0.6, color='g')
        
        return observable_corners, np.where(observable_corner_idx)[0]
    
    # Call get_observation and add noise to the observation
    def get_noisy_observation(self, car_state, draw=True):
        observable_corners, indeces = self.get_observation(car_state, draw=False)
        
        # Add noise to the observation
        mean = 0
        noisy_observable_corners = observable_corners + np.random.normal(mean, self.std_dev, observable_corners.shape)
        
        if draw:
            # Draw noisy observable corners
            for corner in noisy_observable_corners:
                self.ui.draw_circle(corner, 0.6, color='g')
        
        return noisy_observable_corners, indeces
        
if __name__ == '__main__':
    ooi = OOI(None)
    
    print(ooi.corners)
    
    for corner in ooi.corners:
        print(corner)