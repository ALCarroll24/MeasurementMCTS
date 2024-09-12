import numpy as np
from copy import deepcopy as copy
from typing import List, Tuple
from matplotlib.colors import ListedColormap
import pandas as pd
from utils import wrapped_angle_diff, get_pixels_and_values

class ExplorationGrid:
    def __init__(self, bounds: np.ndarray, meters_per_pixel: float,
                 car_max_range: float, car_max_bearing: float, grid=None, ui=None):
        """
        Initialize the exploration grid.
        :param bounds: Numpy array with the bounds of the grid. [[x_min, x_max], [y_min, y_max]]
        :param meters_per_pixel: Meters per pixel of the grid.
        """
        # Grid origin is the minimum x and y values
        self.grid_origin = bounds[:, 0]

        # Get conversion between pixels and meters
        self.meters_per_pixel = meters_per_pixel
        self.len_meters_xy = bounds[:, 1] - bounds[:, 0] # This gives length and height in meters
        self.len_pixels_xy = 1/meters_per_pixel * self.len_meters_xy

        # Get the car parameters we need
        self.car_max_range = car_max_range
        self.car_max_bearing = car_max_bearing
        
        # Use the given grid, if it is none raise an error in functions using grid
        self.grid = grid
        
        # Store the ui
        self.ui = ui
        
    def reset(self, object_df: pd.DataFrame=None):
        """
        Reset the grid maintained by the class to all ones and return it.
        
        params:
        object_df: Dataframe with the objects to calculate occlusions for if desired
        """
        # Create grid using pixel height and width
        grid = np.ones((int(self.len_pixels_xy[0]), int(self.len_pixels_xy[1])))
        
        if object_df is not None:
            grid = self.clear_object_cells(grid, object_df)
        
        return grid
        
    def get_grid(self):
        """
        Get the grid maintained by the class.
        :return: Tuple of the grid and the number of points in range and field of view.
        """
        return self.grid
    
    def set_grid(self, grid: np.ndarray):
        """
        Set the grid maintained by the class.
        :param grid: Grid to set.
        """
        self.grid = grid
        
    def clear_object_cells(self, grid: np.ndarray, object_df: pd.DataFrame):
        # Take a copy of the grid to avoid modifying the original
        grid = copy(grid)
        
        # Get the pixel indices and values of the grid
        pixel_indices, values = get_pixels_and_values(grid)
        
        # Convert into world coordinates 
        # (vector to bottom left of grid + bottom left of bounding box + (pixel indices min pixels of bounding box)
        world_coords = self.grid_origin + self.meters_per_pixel * pixel_indices
        
        # Iterate through each object (row of dataframe)
        for i, row in object_df.iterrows():
            # Find the distance between the object and all the points
            distances = np.linalg.norm(world_coords - row['mean'], axis=1)
            
            # Find the points that are within the radius of the object
            in_range = distances < row['radius']
            
            # Update the grid points that are in range with the value of 0 since there is no need to explore them
            grid[pixel_indices[in_range, 0], pixel_indices[in_range, 1]] = 0
            
        return grid
    
    def draw_grid(self, grid: np.ndarray):
        """
        Draw grid in the background of the plot using imshow.
        :param grid: 2D numpy array representing the occupancy grid.
        :param color: Color to represent occupied cells in the grid.
        """
        if self.ui is None:
            raise ValueError("UI is not set.")
        if self.grid is None:
            raise ValueError("Grid is not set, call reset().")
        
        # Create a color map based on the input color
        cmap = ListedColormap(['white', '#98FB98'])
        
        # Take the transpose of the grid to match the image coordinates
        grid_transpose = grid.T
        
        # Convert the grid to an image
        grid_image = cmap(grid_transpose)  # This converts the grid to an RGBA image
        
        # Plot the grid using imshow
        extent=(self.grid_origin[0], 
                self.grid_origin[0] + grid.shape[1] * self.meters_per_pixel,
                self.grid_origin[1] + grid.shape[0] * self.meters_per_pixel, 
                self.grid_origin[1])
        self.ui.draw_background_image(grid_image, extent, alpha=0.4)

            
    def update(self, grid: np.ndarray, car_state: np.ndarray, object_df: pd.DataFrame=None, return_occlusions=False):
        """
        Update the grid with the state.
        :param grid: 2D numpy array representing the occupancy grid.
        :param state: Tuple (x, y) for the state.
        :param object_df: Dataframe with the objects to calculate occlusions for
        :return Tuple of updated grid and number of points which were explored.
        """
        if self.grid is None:
            raise ValueError("Grid is not set, call reset().")
        # Pull out the elements of the car state
        car_pos, car_yaw = car_state[:2], car_state[3]
        
        # Get a copy before modifying the given grid
        new_grid = copy(grid)
        
        # Create a rectangular bounding box for the car sensor range to minimize number of cell checks
        # Start by creating vectors from the car pointing in the direction of the car's heading and +/- max bearing
        farthest_range_vectors = np.array([[np.cos(car_yaw),
                                            np.sin(car_yaw)],
                                           [np.cos(car_yaw + self.car_max_bearing),
                                            np.sin(car_yaw + self.car_max_bearing)],
                                           [np.cos(car_yaw + self.car_max_bearing/2),
                                            np.sin(car_yaw + self.car_max_bearing/2)],
                                           [np.cos(car_yaw - self.car_max_bearing),
                                            np.sin(car_yaw - self.car_max_bearing)],
                                           [np.cos(car_yaw - self.car_max_bearing/2),
                                            np.sin(car_yaw - self.car_max_bearing/2)]])
        
        # Normalize and multiply by the sensor range to get points at the farthest range
        norms = np.linalg.norm(farthest_range_vectors, axis=1).reshape(-1, 1)
        farthest_range_vectors = farthest_range_vectors / norms * self.car_max_range
        
        # Stack the car position with the farthest points to get the 4 corners of the sensor range
        outer_points = np.vstack((car_state[0:2], farthest_range_vectors + car_state[0:2]))
        x_min, x_max = np.min(outer_points[:,0]), np.max(outer_points[:,0])
        y_min, y_max = np.min(outer_points[:,1]), np.max(outer_points[:,1])
        
        # Check if the bounding box is outside the grid
        if x_min > self.grid_origin[0] + self.len_meters_xy[0] or x_max < self.grid_origin[0] or \
           y_min > self.grid_origin[1] + self.len_meters_xy[1] or y_max < self.grid_origin[1]:
            return new_grid, 0
        
        # Now we need to clip the bounding box to the grid
        x_min = np.clip(x_min, self.grid_origin[0], self.grid_origin[0] + self.len_meters_xy[0])
        x_max = np.clip(x_max, self.grid_origin[0], self.grid_origin[0] + self.len_meters_xy[0])
        y_min = np.clip(y_min, self.grid_origin[1], self.grid_origin[1] + self.len_meters_xy[1])
        y_max = np.clip(y_max, self.grid_origin[1], self.grid_origin[1] + self.len_meters_xy[1])
        
        # Convert the mins and maxes into pixel indices
        x_min_pixel = int((x_min - self.grid_origin[0]) / self.meters_per_pixel)
        x_max_pixel = int((x_max - self.grid_origin[0]) / self.meters_per_pixel)
        y_min_pixel = int((y_min - self.grid_origin[1]) / self.meters_per_pixel)
        y_max_pixel = int((y_max - self.grid_origin[1]) / self.meters_per_pixel)
        
        # Now we can work with the grid in the bounding box
        new_grid_bb = new_grid[x_min_pixel:x_max_pixel, y_min_pixel:y_max_pixel]
        
        # Get the pixel indices and values of the grid
        pixel_indices, values = get_pixels_and_values(new_grid_bb)
        
        # Convert into world coordinates 
        # (vector to bottom left of grid + bottom left of bounding box + (pixel indices min pixels of bounding box)
        world_coords = self.grid_origin + self.meters_per_pixel * (pixel_indices + np.array([x_min_pixel, y_min_pixel]))
        
        # Now find what points are within the car's max range
        in_range = np.linalg.norm(world_coords - car_pos, axis=1) < self.car_max_range
        abs_relative_bearings = np.abs(wrapped_angle_diff(np.arctan2(world_coords[:, 1] - car_pos[1], world_coords[:, 0] - car_pos[0]), car_yaw))
        in_fov = abs_relative_bearings < self.car_max_bearing
        
        # Find the indeces that are in range and in the field of view
        is_observable = in_range & in_fov
        
        # If object dataframe is given, account for occlusions when updating the grid
        occluded_bearings = np.empty((0,2)) # Maintained to add occluded bearing intervals for each occlusion
        if object_df is not None:
            # First get occlusions from objects
            occlusion_df = object_df[(object_df['object_type'] == 'occlusion') | (object_df['object_type'] == 'ooi')].copy()
            
            # Get all the object data we need
            object_means = np.vstack(occlusion_df['mean'])
            object_ranges = np.linalg.norm(object_means - car_pos, axis=1)
            object_radii = np.hstack(occlusion_df['radius'])
            
            # Remove objects that are not in the bounding box and sort by range
            in_bounds = (object_means[:,0] > x_min) & (object_means[:,0] < x_max) & (object_means[:,1] > y_min) & (object_means[:,1] < y_max)
            object_means = object_means[in_bounds]
            object_ranges = object_ranges[in_bounds]
            object_radii = object_radii[in_bounds]
            range_sort = np.argsort(object_ranges)
            object_means = object_means[range_sort]
            object_ranges = object_ranges[range_sort]
            object_radii = object_radii[range_sort]
            
            # If there are no objects after bounding box filter, continue without accounting for occlusions
            if not len(object_means) == 0:
                # Get world coordinates of observable cells sorted by range
                observable_coords = world_coords[is_observable]
                observable_ranges = np.linalg.norm(observable_coords - car_pos, axis=1)
                observable_idx = np.argsort(observable_ranges) # Index of observable points sub array sorted by range
                all_pt_idx = np.where(is_observable)[0][observable_idx] # Index of all points sorted by range
                
                # Iterate through cells by range and account for occlusions as they are within range
                # occluded_bearings = np.empty((0,2)) # Maintained to add occluded bearing intervals for each occlusion
                is_not_occluded = np.zeros_like(is_observable) # Maintained to save if a cell is not occluded
                object_index = 0 # Index of the current object we are checking for occlusions
                for obs_idx, all_idx in zip(observable_idx, all_pt_idx):
                    # Get range and bearing for this cell
                    range = observable_ranges[obs_idx]
                    bearing = np.arctan2(observable_coords[obs_idx][1] - car_pos[1], observable_coords[obs_idx][0] - car_pos[0]) - car_yaw
                    
                    # If we still have occlusions to account for check if we need to account for the next one
                    if object_index < len(object_means):
                        # If we are within range of the next occlusion
                        if range >= object_ranges[object_index]:
                            # Find the angle to the edge of the circle from the bearing and add to the occluded bearing intervals
                            circle_mean_bearing = np.arctan2(object_means[object_index][1] - car_pos[1], object_means[object_index][0] - car_pos[0]) - car_yaw
                            mean_to_edge_angle = np.arcsin(object_radii[object_index] / (object_ranges[object_index] + object_radii[object_index])) # Adding back radius which was removed before for sorting
                            occluded_bearings = np.vstack((occluded_bearings, np.array([circle_mean_bearing - mean_to_edge_angle, circle_mean_bearing + mean_to_edge_angle])))
                            
                            # Add one to the object index to check the next object
                            object_index += 1
                        
                    # Check if the cell is occluded
                    is_not_occluded[all_idx] = ~(np.any((occluded_bearings[:,0] < bearing) & (bearing < occluded_bearings[:,1])))
                    
                # Update the observable cells with occlusions accounted for
                is_observable = is_not_occluded
        
        # Find how many points were prviosuly unexplored and are now explored
        num_explored = np.sum(is_observable & (values == 1))
        
        # Update the grid points that are in range and in the field of view with the value of 0
        new_grid_bb[pixel_indices[is_observable, 0], pixel_indices[is_observable, 1]] = 0
        
        # Now insert the updated bounding box grid back into the original grid
        new_grid[x_min_pixel:x_max_pixel, y_min_pixel:y_max_pixel] = new_grid_bb
        
        if return_occlusions:
            return new_grid, num_explored, occluded_bearings
        
        return new_grid, num_explored
    
        