import numpy as np
from shapely import Polygon, Point, LineString
from utils import wrap_angle, wrapped_angle_diff, angle_in_interval
from typing import Union

class OOI:
    def __init__(self, ui, position=(50,50), length=4, width=8, std_dev=0.5,
                 car_max_range=20.0, car_max_bearing=45.0):
        self.position = position
        
        # TODO: support yaw
        # self.yaw = yaw
        
        self.ui = ui
        self.length = length
        self.width = width
        self.std_dev = std_dev
        
        self.max_range = car_max_range
        self.max_bearing = car_max_bearing
        
        # Calculate corner positions
        self.corners = self.get_corners()
        
        # Calculate the collision polygon
        self.collision_polygon = self.get_collision_polygon()
        
    def get_corners(self):
        # Calculate corner positions
        corners = np.zeros((4, 2))
        corners[0, :] = self.position + np.array([self.width / 2, self.length / 2])
        corners[1, :] = self.position + np.array([self.width / 2, -self.length / 2])
        corners[2, :] = self.position + np.array([-self.width / 2, -self.length / 2])
        corners[3, :] = self.position + np.array([-self.width / 2, self.length / 2])
        
        # Find the angle range with which this corner is observable
        self.corner_bearing_ranges = np.zeros((4, 2))
        for i, corner in enumerate(corners):
            # print("Corner", i, "position:", corner)
            # Draw a vector from this corner to the center of the OOI
            corner_to_center = self.position - corner
            
            # Break out components and multiply by 2 to get the full range
            next_x_corner = np.array([corner_to_center[0] * 2, 0])
            next_y_corner = np.array([0, corner_to_center[1] * 2])
            
            # Draw for debugging
            # self.ui.draw_arrow(corner, corner + next_x_corner)
            # self.ui.draw_arrow(corner, corner + next_y_corner)
            
            # Find the angle that each vector makes
            x_angle = np.arctan2(next_x_corner[1], next_x_corner[0])
            y_angle = np.arctan2(next_y_corner[1], next_y_corner[0])
            
            # Find the larger interval (this is the side with the 270 degree fov)
            if x_angle > np.pi/2 and y_angle < 0:
                self.corner_bearing_ranges[i, 0] = y_angle
                self.corner_bearing_ranges[i, 1] = x_angle
            elif x_angle > np.pi/2 and y_angle > 0:
                self.corner_bearing_ranges[i, 0] = x_angle
                self.corner_bearing_ranges[i, 1] = y_angle
            elif x_angle < np.pi/2 and y_angle > 0:
                self.corner_bearing_ranges[i, 0] = y_angle
                self.corner_bearing_ranges[i, 1] = x_angle
            elif x_angle < np.pi/2 and y_angle < 0:
                self.corner_bearing_ranges[i, 0] = x_angle
                self.corner_bearing_ranges[i, 1] = y_angle
            
        # print("Corner bearing ranges:", np.degrees(corner_bearing_ranges))
        
        # Rotate corners TODO: This rotates around (0,0) and needs to be done some other way
        # rotation_matrix = np.array([[np.cos(self.yaw), -np.sin(self.yaw)], [np.sin(self.yaw), np.cos(self.yaw)]])
        # corners = np.matmul(rotation_matrix, corners.T).T
        return corners
    
    # Return the collision polygon for the OOI
    def get_collision_polygon(self) -> Polygon:
        # Calculate corner positions
        corners = self.get_corners()
        
        # Create an OOI polygon from the corners
        return Polygon(corners)
    
    def simulate_observation(self, car_state, nearest_observable=False, draw=False):
        car_pos = Point(car_state[0:2])
        
        # Get the coordinates of the corners of the OOI
        ooi_corners = list(self.collision_polygon.exterior.coords)[:-1]  # Exclude the repeated first/last point

        # Create line of sight from the car center to each corner of the OOI
        lines_of_sight = [LineString([car_pos, Point(corner)]) for corner in ooi_corners]
        
        # Check if each line of sight cross the OOI (which means it is not observable)
        intersections = [line.crosses(ooi) for line in lines_of_sight]
        
        for i, corner in enumerate(ooi_corners):
            # If point is non-observable and nearest_observable is set
            if not intersections[i] and nearest_observable:
                # Return car position projected to corner observable line
                extension_length = 100
                
                # Get direction vector and extend it then add back the initial corner
                prev_corner = (ooi_corners[i-1] - corner) * extension_length + ooi_corners[i-1]
                next_corner = (ooi_corners[(i+1)%4] - corner) * extension_length + ooi_corners[(i+1)%4]
                
                # Make these into Line Strings to do math
                obs_line1 = LineString([corner, prev_corner])
                obs_line2 = LineString([corner, next_corner])
                
                # Project the car position onto the line, get length along line
                obs_len1 = obs_line1.project(car_pos)
                obs_len2 = obs_line2.project(car_pos)
                
                # Use length along line to get projected point
                obs_pt1 = obs_line1.interpolate(obs_len1)
                obs_pt2 = obs_line2.interpolate(obs_len2)
                
                # Draw the projected points
                if draw:
                    self.ui.draw_arrow(car_pos, obs_pt1)
                    self.ui.draw_arrow(car_pos, obs_pt2)
                

        
        
    def draw(self):
        # Draw main rectangle
        self.ui.draw_rectangle(self.position, self.width, self.length, 0)
        
        # Draw soft collision polygon if available
        if self.soft_collision_polygon is not None:
            for x, y in self.soft_collision_polygon.exterior.coords:
                self.ui.draw_circle((x, y), 0.12)
        
        # Draw the collision polygon to ensure it matches
        # for x, y in self.get_collision_polygons().exterior.coords:
        #     print(x, y)
        #     self.ui.draw_circle((x, y), 1.5)
        
        # Draw corners
        # for corner in self.corners:
        #     self.ui.draw_circle(corner, 0.6)
        
    def get_observation(self, car_state, draw=True):
        # Get needed parameters from car
        car_position = car_state[0:2]
        car_yaw = car_state[2]
        car_range = self.max_range
        car_max_bearing = np.radians(self.max_bearing)
        
        # Do initial distance check to see if we can see any corners
        # If not, return empty list (avoid needless computation)
        distances = np.linalg.norm(self.corners - car_position, axis=1)
        if np.all(distances >= car_range):
            return np.array([]), []
        
        # Find which corners are observable
        observable_corners = []
        indeces = []
        for i, corner in enumerate(self.corners):
            # Calculate distance and bearing to corner
            distance = np.linalg.norm(corner - car_position)
            bearing = wrapped_angle_diff(np.arctan2(corner[1] - car_position[1], corner[0] - car_position[0]), car_yaw)
            opposite_bearing = wrap_angle(np.arctan2(car_position[1] - corner[1], car_position[0] - corner[0]))
            # print("Car yaw:", np.degrees(car_yaw), "Corner bearing:", np.degrees(np.arctan2(corner[1] - car_position[1], corner[0] - car_position[0])))
            # print("Corner:", i, "bearing:", np.degrees(bearing), "opp_bearing:", np.degrees(opposite_bearing), "interval:", np.degrees(self.corner_bearing_ranges[i, 0]), np.degrees(self.corner_bearing_ranges[i, 1]), "in_interval:", angle_in_interval(opposite_bearing, self.corner_bearing_ranges[i, 0], self.corner_bearing_ranges[i, 1]))
            
            # Check if corner is observable
            if distance >= car_range:
                # print("Corner is ", distance - car_range, "m too far away")
                pass
            # Check if corner is within the car's field of view
            elif np.abs(bearing) >= car_max_bearing:
                # print("Bearing is", np.degrees(np.abs(bearing) - car_max_bearing), "degrees too large")
                pass
            # Check if any observation vectors go through one of the walls
            elif not angle_in_interval(opposite_bearing, self.corner_bearing_ranges[i, 0], self.corner_bearing_ranges[i, 1]):
                # print("Bearing from corner:", np.degrees(opposite_bearing), "is out of interval", np.degrees(self.corner_bearing_ranges[i, 0]), np.degrees(self.corner_bearing_ranges[i, 1]))
                pass
            else:
                observable_corners.append(corner)
                indeces.append(i)
                
        if draw:
            # Draw observable corners
            for corner in observable_corners:
                self.ui.draw_circle(corner, 0.6)
                
        # Convert to numpy array for later usage
        obs_corners_vec = np.array(observable_corners).flatten()
        
        return obs_corners_vec, indeces
    
    # Call get_observation and add noise to the observation
    def get_noisy_observation(self, car_state, draw=True):
        observable_corners, indeces = self.get_observation(car_state, draw=draw)
        
        # Add noise to the observation
        mean = 0
        noisy_observable_corners = observable_corners + np.random.normal(mean, self.std_dev, observable_corners.shape)
        
        return noisy_observable_corners, indeces
        
if __name__ == '__main__':
    ooi = OOI(None)
    
    print(ooi.corners)
    
    for corner in ooi.corners:
        print(corner)