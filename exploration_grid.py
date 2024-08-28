import numpy as np
from copy import deepcopy as copy
from typing import List, Tuple
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom
from utils import wrapped_angle_diff, get_pixels_and_values

class ExplorationGrid:
    def __init__(self, bounds: np.ndarray, meters_per_pixel: float,
                 car_max_range: float, car_max_bearing: float, ui=None):
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
        pixel_HW = 1/meters_per_pixel * self.len_meters_xy

        # Create grid using pixel height and width
        self.grid = np.ones((int(pixel_HW[0]), int(pixel_HW[1])))
        self.initial_grid = copy(self.grid)
        
        # Get the car parameters we need
        self.car_max_range = car_max_range
        self.car_max_bearing = car_max_bearing
        
        # Store the ui
        self.ui = ui
        
    def get_state(self):
        """
        Get the state of the grid.
        :return: Tuple of the grid and the number of points in range and field of view.
        """
        return self.grid
        
    def draw_grid(self, grid: np.ndarray, color: str='g'):
        """
        Draw a state on the grid.
        :param state: Tuple (x, y) for the state.
        :param color: Color of the state.
        """
        # Pull out the elements of the state
        # car_state, corner_means, corner_covs, explore_grid, horizon = state
        
        # Plot the grid
        for (i, j), value in np.ndenumerate(grid):
            grid_xy = np.array([i, j]) * self.meters_per_pixel
            world_xy = grid_xy + self.grid_origin
            self.ui.patches.append(Rectangle(world_xy, self.meters_per_pixel, self.meters_per_pixel,
                                             linewidth=1, alpha=0.2, facecolor='g' if value == 1 else 'w'))
            
    def update(self, grid: np.ndarray, car_state: np.ndarray):
        """
        Update the grid with the state.
        :param state: Tuple (x, y) for the state.
        :return Tuple of updated grid and number of points which were explored.
        """
        # Pull out the elements of the car state
        car_pos, car_yaw = car_state[:2], car_state[3]
        
        # Get a copy before modifying the given grid
        new_grid = copy(grid)
        
        # Create a rectangular bounding box for the car sensor range to minimize number of cell checks
        # Start by creating vectors from the car pointing in the direction of the car's heading and +/- max bearing
        farthest_range_vectors = np.array([[np.cos(car_state[3]),
                                            np.sin(car_state[3])],
                                           [np.cos(car_state[3] + self.car_max_bearing),
                                            np.sin(car_state[3] + self.car_max_bearing)],
                                           [np.cos(car_state[3] + self.car_max_bearing/2),
                                            np.sin(car_state[3] + self.car_max_bearing/2)],
                                           [np.cos(car_state[3] - self.car_max_bearing),
                                            np.sin(car_state[3] - self.car_max_bearing)],
                                           [np.cos(car_state[3] - self.car_max_bearing/2),
                                            np.sin(car_state[3] - self.car_max_bearing/2)]])
        
        # Normalize and multiply by the sensor range to get points at the farthest range
        norms = np.linalg.norm(farthest_range_vectors, axis=1).reshape(-1, 1)
        farthest_range_vectors = farthest_range_vectors / norms * self.car_max_range
        
        # Stack the car position with the farthest points to get the 4 corners of the sensor range
        outer_points = np.vstack((car_state[0:2], farthest_range_vectors + car_state[0:2]))
        print(f'Outer points: {outer_points}')
        x_min, x_max = np.min(outer_points[:,0]), np.max(outer_points[:,0])
        y_min, y_max = np.min(outer_points[:,1]), np.max(outer_points[:,1])
        
        # Draw the bounding box as a polygon
        for point in outer_points:
            self.ui.draw_point(point, color='r')
        bb_points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
        self.ui.draw_polygon(bb_points, 'r')
        
        # Check if the bounding box is outside the grid
        if x_min > self.grid_origin[0] + self.len_meters_xy[0] or x_max < self.grid_origin[0] or \
           y_min > self.grid_origin[1] + self.len_meters_xy[1] or y_max < self.grid_origin[1]:
            return new_grid, 0
        
        # Now we need to clip the bounding box to the grid
        x_min = np.clip(x_min, self.grid_origin[0], self.grid_origin[0] + self.len_meters_xy[0])
        x_max = np.clip(x_max, self.grid_origin[0], self.grid_origin[0] + self.len_meters_xy[0])
        y_min = np.clip(y_min, self.grid_origin[1], self.grid_origin[1] + self.len_meters_xy[1])
        y_max = np.clip(y_max, self.grid_origin[1], self.grid_origin[1] + self.len_meters_xy[1])
        
        bb_clipped_points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
        self.ui.draw_polygon(bb_clipped_points, 'b')
        
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
        in_range_fov = in_range & in_fov
        
        # Find how many points were prviosuly unexplored and are now explored
        num_explored = np.sum(in_range_fov & (values == 1))
        
        # Update the grid points that are in range and in the field of view with the value of 0
        new_grid_bb[pixel_indices[in_range_fov, 0], pixel_indices[in_range_fov, 1]] = 0
        
        # Now insert the updated bounding box grid back into the original grid
        new_grid[x_min_pixel:x_max_pixel, y_min_pixel:y_max_pixel] = new_grid_bb
        
        return new_grid, num_explored
    
    def get_upscaled_pixels_and_values(self, grid: np.ndarray, meters_per_pixel: float=1) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Upscale if resolution is lower than input and return pixel indices corresponding values and new resolution.
        :param grid: Grid to convert.
        :return: Tuple of pixel indices and corresponding values.
        """
        meters_per_pixel = np.clip(meters_per_pixel, self.meters_per_pixel, None) # Clip to be at least the current resolution
        upscale_factor = meters_per_pixel / self.meters_per_pixel
        upscaled_grid = zoom(grid, upscale_factor, order=0) # Nearest neighbor interpolation
        
        pixel_indices, meters_per_pixel = get_pixels_and_values(upscaled_grid)
        
        return pixel_indices, meters_per_pixel, meters_per_pixel
        