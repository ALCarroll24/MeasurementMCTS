import numpy as np
from copy import deepcopy as copy
from typing import List, Tuple
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom
from utils import wrapped_angle_diff, get_pixels_and_values

class ExplorationGrid:
    def __init__(self, env, bounds: np.ndarray, padding: np.ndarray=np.array([15,15]),
                 meters_per_pixel: float=1):
        """
        Initialize the exploration grid.
        :param bounds: Numpy array with the bounds of the grid. [[x_min, x_max], [y_min, y_max]]
        :param padding: Numpy array with the padding to add to the grid for x and y.
        :param meters_per_pixel: Meters per pixel of the grid.
        """
        # Keep a copy of the environment
        self.env = env
        
        # Make a copy of the mean bounds and add padding
        bounds = copy(bounds)
        x_padding_m, y_padding_m = padding # meters of padding to add to the grid on each side
        bounds[:, 0] -= np.array([x_padding_m, y_padding_m])
        bounds[:, 1] += np.array([x_padding_m, y_padding_m])
        
        # Grid origin is the minimum x and y values
        self.grid_origin = bounds[:, 0]

        # Get conversion between pixels and meters
        self.meters_per_pixel = meters_per_pixel
        self.meter_HW = bounds[:, 1] - bounds[:, 0] # This gives height and width in meters
        pixel_HW = 1/meters_per_pixel * self.meter_HW

        # Create grid using pixel height and width
        self.grid = np.ones((int(pixel_HW[0]), int(pixel_HW[1])))
        self.initial_grid = copy(self.grid)
        
        # Get the car parameters we need
        self.car_max_range = self.env.car.max_range
        self.car_max_bearing = env.car.max_bearing
        self.car_length = env.car.length
        
        
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
            self.env.ui.patches.append(Rectangle(world_xy, self.meters_per_pixel, self.meters_per_pixel,
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
        
        # Center of grid in world coordinates
        grid_center = self.grid_origin + self.meters_per_pixel * np.array([new_grid.shape[0]//2, new_grid.shape[1]//2])
        
        # Do an initial check to see if there is no overlap between the car max range box and the grid
        grid_radius_x, grid_radius_y = self.meter_HW / 2
        if grid_center[0] - grid_radius_x > car_pos[0] + self.car_max_range or \
           grid_center[0] + grid_radius_x < car_pos[0] - self.car_max_range or \
           grid_center[1] - grid_radius_y > car_pos[1] + self.car_max_range or \
           grid_center[1] + grid_radius_y < car_pos[1] - self.car_max_range:
            # Since there is no overlap, return the grid as is
            return new_grid
        
        # Get the pixel indices and values of the grid
        pixel_indices, values = get_pixels_and_values(new_grid)
        
        # Convert into world coordinates (vector to bottom left of grid + pixel indices converted to meters)
        world_coords = self.grid_origin + self.meters_per_pixel * pixel_indices
        
        # Now find what points are within the car's max range
        in_range = np.linalg.norm(world_coords - car_pos, axis=1) < self.car_max_range
        abs_relative_bearings = np.abs(wrapped_angle_diff(np.arctan2(world_coords[:, 1] - car_pos[1], world_coords[:, 0] - car_pos[0]), car_yaw))
        in_fov = abs_relative_bearings < self.car_max_bearing
        
        # Find the indeces that are in range and in the field of view
        in_range_fov = in_range & in_fov
        
        # Find how many points were prviosuly unexplored and are now explored
        num_explored = np.sum(in_range_fov & (values == 1))
        
        # Update the grid points that are in range and in the field of view with the value of 0
        new_grid[pixel_indices[in_range_fov, 0], pixel_indices[in_range_fov, 1]] = 0
        
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
        