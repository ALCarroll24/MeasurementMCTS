import numpy as np
import pandas as pd
from typing import NamedTuple, List, Tuple
from measurement_mcts.utils.utils import get_ellipse_scaling, wrap_angle
from measurement_mcts.environment.static_kf_2d import measurement_model

# class ObjectTuple(NamedTuple):
#     """
#     This defines a single object which is a row of the object dataframe maintained in the ObjectManager class
#     """
#     object_type: str         # occlusion, obstacle, ooi
#     shape: str               # circle, 4polygon
#     mean: np.ndarray         # [x, y]
#     ooi_id: int=None
#     points: np.ndarray=None  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
#     covariances: List[np.ndarray]=None # [cov1, cov2, cov3, cov4]
#     radius: float=None
#     observed: np.ndarray=np.zeros(4, dtype=bool) # [obs1, obs2, obs3, obs4]
#     in_collision: bool=False
    
# def get_empty_df():
#     """
#     This function returns an empty dataframe with the columns of the ObjectTuple
#     """
#     return pd.DataFrame({'object_type' : pd.Series(dtype='str'),
#                          'shape' : pd.Series(dtype='str'),
#                          'mean' : pd.Series(dtype='object'),
#                          'ooi_id' : pd.Series(dtype='int'),
#                          'points' : pd.Series(dtype='object'),
#                          'covariances' : pd.Series(dtype='object'),
#                          'radius' : pd.Series(dtype='float'),
#                          'observed' : pd.Series(dtype='object'),
#                          'in_collision' : pd.Series(dtype='bool')})

class ObjectManager:
    def __init__(
        self,
        num_obstacles: int, 
        num_occlusion: int, 
        num_oois: int, 
        car_collision_radius: float,
        car_sensor_range: float, 
        car_max_bearing: float, 
        object_bounds: np.ndarray = np.array([15, 85]),
        size_bounds: np.ndarray   = np.array([1.0, 10.0]),
        ooi_size_bounds: np.ndarray = np.array([1.0, 10.0]),
        bounding_box_buffer: float = 5.0,
        object_min_spacing: float = 10.0,
        init_covariance_diag: float = 8,
        init_center_stddev: float = 5.0,
        init_width_guess: float = 5.0,
        range_stddev: float = 1.0,
        bearing_stddev: float = 0.1,
        final_corner_covariance: float = 0.4,
        ui=None
    ):
        # Random object generation parameters
        self.num_obstacles = num_obstacles
        self.num_occlusion = num_occlusion
        self.num_oois = num_oois
        self.object_bounds = object_bounds
        self.size_bounds = size_bounds
        self.ooi_size_bounds = ooi_size_bounds
        self.bounding_box_buffer = bounding_box_buffer  # Buffer in sensor area checks
        self.object_min_spacing = object_min_spacing   # Spacing to prevent object overlap
        
        # Noisy initialization parameters
        self.init_covariance_diag = init_covariance_diag
        self.init_center_stddev = init_center_stddev
        self.init_width_guess = init_width_guess
        
        # Noise model for generating noisy observations
        self.range_stddev = range_stddev
        self.bearing_stddev = bearing_stddev
        
        # Car parameters and a UI for rendering
        self.car_collision_radius = car_collision_radius
        self.car_sensor_range = car_sensor_range
        self.car_max_bearing = car_max_bearing
        self.final_corner_covariance = final_corner_covariance # Covariance for completed corner for drawing
        self.ui = ui
        
        # --- New data structures: arrays instead of a DataFrame ---
        # Obstacles: circle shapes
        self.obstacle_means = np.zeros((num_obstacles, 2))
        self.obstacle_radii = np.zeros(num_obstacles)
        # Occlusions: circle shapes
        self.occlusion_means = np.zeros((num_occlusion, 2))
        self.occlusion_radii = np.zeros(num_occlusion)
        # OOIs: each is a 4-pt polygon (rectangle or general quadrilateral)
        # Shape: (num_oois, 4, 2)
        self.oois = np.zeros((num_oois, 4, 2))
    
    # Need to remake for new data structure
    # def add_ooi(self, points: np.ndarray, df: pd.DataFrame=None):
    #     """
    #     Add an OOI to the dataframe maintained by the class or a passed dataframe
    #     """
    #     if df is None and self.df is None:
    #         raise ValueError('Objects have not been generated yet and no dataframe was passed')
        
    #     # Use the passed dataframe if it is not None
    #     if df is None:
    #         df = self.df  
        
    #     # Get the number of OOIs to find next id
    #     if len(df) == 0:
    #         ooi_id = 0
    #     else:
    #         ooi_ids = df[df['object_type'] == 'ooi']['ooi_id'].values
    #         max_ooi_id = np.max(ooi_ids)
    #         ooi_id = max_ooi_id + 1
        
    #     # Create initial covariances for the points
    #     covariances = [np.diag([self.init_covariance_diag, self.init_covariance_diag]) for _ in range(4)]
        
    #     # Find max radius of the points from the mean
    #     mean = np.mean(points, axis=0)
    #     max_radius = np.max(np.linalg.norm(points - mean, axis=1))
        
    #     # Create the object tuple within a dataframe
    #     object_row_df = pd.DataFrame([ObjectTuple(object_type='ooi',
    #                                               shape='4polygon',
    #                                               ooi_id=ooi_id,
    #                                               mean=mean,
    #                                               covariances=covariances,
    #                                               points=points,
    #                                               radius=max_radius)])
    #     # Add to the dataframe
    #     df = pd.concat([df, object_row_df], ignore_index=True)
    #     print(f'Adding OOI with id {ooi_id}')
        
    #     return df
    
    def reset(self, car_state):
        """
        Generates obstacles, occlusions, and OOIs with retries if placement fails.
        """
        max_restarts = 100  # Maximum restarts for the entire process
        max_attempts_per_object = 1000  # Maximum attempts per object placement

        def is_overlapping(mean, radius, existing_objs):
            """Check if (mean, radius) overlaps with any item in existing_objs."""
            for (obj_mean, obj_radius) in existing_objs:
                dist = np.linalg.norm(mean - obj_mean)
                if dist < (radius + obj_radius + self.object_min_spacing):
                    return True
            return False

        for _ in range(max_restarts):
            # Reset placed_objects on each restart
            placed_objects = [
                (np.array(car_state[0:2]), self.car_collision_radius)
            ]

            # --- Generate obstacles ---
            obstacle_success = True
            for i in range(self.num_obstacles):
                attempts = 0
                while True:
                    attempts += 1
                    if attempts > max_attempts_per_object:
                        obstacle_success = False
                        break  # Exit while loop
                    mean = np.random.uniform(self.object_bounds[0], self.object_bounds[1], size=2)
                    radius = np.random.uniform(self.size_bounds[0], self.size_bounds[1])
                    if not is_overlapping(mean, radius, placed_objects):
                        self.obstacle_means[i, :] = mean
                        self.obstacle_radii[i] = radius
                        placed_objects.append((mean, radius))
                        break
                if not obstacle_success:
                    break  # Exit obstacle loop to restart
            if not obstacle_success:
                continue  # Restart the entire process

            # --- Generate occlusions ---
            occlusion_success = True
            for i in range(self.num_occlusion):
                attempts = 0
                while True:
                    attempts += 1
                    if attempts > max_attempts_per_object:
                        occlusion_success = False
                        break
                    mean = np.random.uniform(self.object_bounds[0], self.object_bounds[1], size=2)
                    radius = np.random.uniform(self.size_bounds[0], self.size_bounds[1])
                    if not is_overlapping(mean, radius, placed_objects):
                        self.occlusion_means[i, :] = mean
                        self.occlusion_radii[i] = radius
                        placed_objects.append((mean, radius))
                        break
                if not occlusion_success:
                    break
            if not occlusion_success:
                continue

            # --- Generate OOIs ---
            ooi_success = True
            for i in range(self.num_oois):
                attempts = 0
                while True:
                    attempts += 1
                    if attempts > max_attempts_per_object:
                        ooi_success = False
                        break
                    mean = np.random.uniform(self.object_bounds[0], self.object_bounds[1], size=2)
                    length_width = np.random.uniform(self.ooi_size_bounds[0], self.ooi_size_bounds[1], size=2)
                    corners = np.array([
                        mean + np.array([-length_width[0]/2, -length_width[1]/2]),
                        mean + np.array([ length_width[0]/2, -length_width[1]/2]),
                        mean + np.array([ length_width[0]/2,  length_width[1]/2]),
                        mean + np.array([-length_width[0]/2,  length_width[1]/2])
                    ])
                    max_radius = 0.5 * np.linalg.norm(length_width)
                    if not is_overlapping(mean, max_radius, placed_objects):
                        self.oois[i, :, :] = corners
                        placed_objects.append((mean, max_radius))
                        break
                if not ooi_success:
                    break
            if not ooi_success:
                continue

            # If all objects placed successfully, exit
            return

        # If all restarts failed
        raise RuntimeError("Failed to place objects after maximum restarts")
        
    def get_noisy_initial_state(self):
        """
        Use maintained real object means to generate noisy initial states for MCTS.
        
        Class variables init_covariance_diag, init_center_stddev, init_width_guess are used to make the guess for the initial state.
        """
        
        # Create initial output data structures
        ooi_corner_means = np.zeros((self.num_oois, 4, 2))
        ooi_corner_covariances = np.zeros((self.num_oois, 4, 2, 2))
        
        # Fill the diagonals of the last two dimensions with the initial covariance diagonal value
        for ooi_idx in range(self.num_oois):
            for c_idx in range(4):
                np.fill_diagonal(ooi_corner_covariances[ooi_idx, c_idx], self.init_covariance_diag)
        
        # Generate noisy center point for each OOI
        noisy_ooi_means = self.oois.mean(axis=1) + \
                        np.random.normal(0, self.init_center_stddev, size=(self.num_oois, 2))
                        
        # Use the init_width_guess to generate the corner points of the object
        h_w = self.init_width_guess/2 # half width
        for ooi_idx in range(self.num_oois):
            ooi_corner_means[ooi_idx] = np.array([
                        noisy_ooi_means[ooi_idx] + np.array([-h_w, -h_w]),
                        noisy_ooi_means[ooi_idx] + np.array([ h_w, -h_w]),
                        noisy_ooi_means[ooi_idx] + np.array([ h_w,  h_w]),
                        noisy_ooi_means[ooi_idx] + np.array([-h_w,  h_w])
                    ])

        # Return the object means and covariances tuple
        return ooi_corner_means, ooi_corner_covariances

    def draw_objects(
        self, 
        car_state, 
        in_collision_obs, 
        in_collision_ocl, 
        in_collision_oois, 
        observation_indices=None,
        observation=None,
        ooi_means=None,
        ooi_covs=None
    ):
        """
        Draw all obstacles, occlusions, and OOIs. For objects in collision,
        draw a solid red circle on top. Also, display an observation marker
        for OOI corners that have been observed, draw arrows 
        from the vehicle to those corners and highlight with a circle.
        
        :param car_state: (x, y, theta) of the car
        :param in_collision_obs: 1D array of obstacle indices that are in collision
        :param in_collision_ocl: 1D array of occlusion indices that are in collision
        :param in_collision_oois: 1D array of OOI indices that are in collision
        :param observation_indices: Dictionary of observed OOI corners for displaying observation arrows
        :param observation: Dictionary of observed OOI corner positions
        :param ooi_means: Means of the OOIs for drawing current estimate
        :param ooi_covs: Covariances of the OOIs for drawing current estimated uncertainty
        """
        if self.ui is None:
            raise ValueError("No UI has been set for drawing.")
        
        # Car position and collision radius
        car_pos = np.array(car_state[0:2], dtype=float)
        car_radius = self.car_collision_radius
        
        # (Optional) Draw the car's collision circle
        self.ui.draw_circle(car_pos, car_radius, color='r', facecolor='none', alpha=0.1)
        
        # ------------------------------------------------------------------
        # 1) Draw Obstacles (stored in self.obstacle_means, self.obstacle_radii)
        # ------------------------------------------------------------------
        for i in range(self.num_obstacles):
            center = self.obstacle_means[i]
            radius = self.obstacle_radii[i]
            
            # Draw the obstacle circle in its normal style
            self.ui.draw_circle(center, radius, color='r', facecolor='r', alpha=0.2)
            
            # If this obstacle is in collision, overlay a solid red circle
            if i in in_collision_obs:
                self.ui.draw_circle(center, radius, color='red', facecolor='red', alpha=0.5)
        
        # ------------------------------------------------------------------
        # 2) Draw Occlusions (stored in self.occlusion_means, self.occlusion_radii)
        # ------------------------------------------------------------------
        for i in range(self.num_occlusion):
            center = self.occlusion_means[i]
            radius = self.occlusion_radii[i]
            
            # Draw the occlusion circle in yellow
            self.ui.draw_circle(center, radius, color='y', facecolor='y', alpha=0.3)
            
            # If in collision, overlay a solid red circle
            if i in in_collision_ocl:
                self.ui.draw_circle(center, radius, color='red', facecolor='red', alpha=0.5)
        
        # ------------------------------------------------------------------
        # 3) Draw OOIs (stored in self.oois as Nx4x2 corners)
        # ------------------------------------------------------------------
        for i in range(self.num_oois):
            corners = self.oois[i]  # shape = (4, 2)
            
            # Draw the polygon outline
            self.ui.draw_polygon(corners, color='b', facecolor='none', alpha=0.2)
            
            # for corner in corners:
            #     # A small cyan point for each corner
            #     self.ui.draw_point(corner, color='cyan')
            
            # Draw a red bounding circle for OOIs in collision
            if i in in_collision_oois:
                ooi_center = corners.mean(axis=0)
                # bounding radius is the max distance from center to any corner
                offset = corners - ooi_center
                max_radius = np.linalg.norm(offset, axis=1).max()
                self.ui.draw_circle(ooi_center, max_radius, color='red', facecolor='red', alpha=0.5)
            
        # ------------------------------------------------------------------
        # 4) Draw Observation index (if available)
        #    This marks the real observed corners with green circles and arrows
        # ------------------------------------------------------------------
        if observation_indices is not None:
            # Iterate through observed OOIs: observation_indices = {ooi_idx: [corner0, corner3, ...], ...}
            for ooi_idx, observed_corners in observation_indices.items():
                # Iterate through each corner of the OOI
                for corner_idx in observed_corners:
                    # Get the corner position
                    pt = self.oois[ooi_idx, corner_idx] 
                    
                    # A green circle to indicate an observed corner
                    self.ui.draw_circle(pt, 0.5, color='cyan', facecolor='none', alpha=0.2)
                    
                    # An arrow from the car to the observed corner
                    self.ui.draw_arrow(car_pos, pt, color='g', alpha=0.1)

        # ------------------------------------------------------------------
        # 5) Draw Noisy Observation (if available)
        # ------------------------------------------------------------------
        if observation is not None:
            # Iterate through observed OOIs: observation = {ooi_idx: [[x0, y0], [x3, y3], ...], ...}
            for ooi_idx, observed_corners in observation.items():
                # Iterate through each corner of the OOI
                for pt in observed_corners:
                    # A green circle to indicate an observed corner
                    self.ui.draw_circle(pt, 0.5, color='g', facecolor='none', alpha=1.0)
                    
                    # # An arrow from the car to the observed corner
                    # self.ui.draw_arrow(car_pos, pt, color='g', alpha=0.1)
                    
        # ------------------------------------------------------------------
        # 6) Draw Estimated OOIs (if available)
        # ------------------------------------------------------------------
        if (ooi_means is not None) and (ooi_covs is not None):
            for i in range(self.num_oois):
                # Draw the estimated means
                self.ui.draw_polygon(ooi_means[i], color='purple', facecolor='none', linestyle='--', alpha=1.0)
                
                for j in range(4):
                    # Change color to green if this corner is fully observed
                    if np.trace(ooi_covs[i][j]) < self.final_corner_covariance:
                        color = 'g'
                    else:
                        color = 'cyan'
                    
                    # A small cyan point for each corner
                    self.ui.draw_point(ooi_means[i][j], color=color, alpha=1.0)
                    
                    # Draw the covariance ellipse
                    scalings, angle = get_ellipse_scaling(ooi_covs[i][j])
                    self.ui.draw_ellipse(ooi_means[i][j], scalings[0], scalings[1], 
                                         angle=angle, color='purple', alpha=0.2)
                    
    def check_collision(self, car_state):
        """
        Checks which obstacles, occlusions, and OOIs are colliding with the car.
        Returns three 1D arrays of indices:
        - in_collision_obs   (for obstacles)
        - in_collision_ocl   (for occlusions)
        - in_collision_oois  (for OOIs)
        """
        car_pos = np.array(car_state[0:2], dtype=float)
        car_radius = self.car_collision_radius
        
        # -------------------------
        # 1) Check collision: Obstacles
        # -------------------------
        if self.num_obstacles > 0:
            obstacle_distances = np.linalg.norm(self.obstacle_means - car_pos, axis=1)
            # True if distance < (car + obstacle radius)
            collision_mask_obs = obstacle_distances < (car_radius + self.obstacle_radii)
            in_collision_obs = np.where(collision_mask_obs)[0]
        else:
            in_collision_obs = np.array([], dtype=int)

        # -------------------------
        # 2) Check collision: Occlusions
        # -------------------------
        if self.num_occlusion > 0:
            occlusion_distances = np.linalg.norm(self.occlusion_means - car_pos, axis=1)
            collision_mask_ocl = occlusion_distances < (car_radius + self.occlusion_radii)
            in_collision_ocl = np.where(collision_mask_ocl)[0]
        else:
            in_collision_ocl = np.array([], dtype=int)

        # -------------------------
        # 3) Check collision: OOIs
        # -------------------------
        # We'll treat each OOI as a circle with:
        #   center = mean of its four corners
        #   radius = max distance from that center to any corner
        if self.num_oois > 0:
            # Compute each OOI's center
            ooi_centers = self.oois.mean(axis=1)  # shape (num_oois, 2)
            # Compute bounding radius (max corner distance from center)
            corner_offsets = self.oois - ooi_centers[:, None, :]  # broadcast center to corners
            corner_dists = np.linalg.norm(corner_offsets, axis=2)  # shape (num_oois, 4)
            ooi_radii = corner_dists.max(axis=1)
            
            # Collision check with car
            ooi_distances = np.linalg.norm(ooi_centers - car_pos, axis=1)
            collision_mask_ooi = ooi_distances < (car_radius + ooi_radii)
            in_collision_oois = np.where(collision_mask_ooi)[0]
        else:
            in_collision_oois = np.array([], dtype=int)
        
        # Return the indices that are colliding
        return in_collision_obs, in_collision_ocl, in_collision_oois
    
    def get_observation_indices(self, car_state):
        """
        Determine which OOI corners are visible to the car, considering occlusions.
        Returns a dictionary `observation_indices`, e.g. {ooi_idx: [corner0, corner3, ...], ...}.
        """
        # Make sure we actually have oois to observe
        if (self.num_oois == 0):
            return {}  # No oois -> no observations
        
        # Car pose & sensor parameters
        car_x, car_y, car_heading = car_state[0], car_state[1], car_state[3]
        car_pos = np.array([car_x, car_y], dtype=float)
        sensor_range = self.car_sensor_range
        sensor_max_bearing = self.car_max_bearing
        
        # ----------------------------------------------------------------
        # 1) Build bounding box for the sensor range + buffer
        #    (Optional: you can omit if you want to skip bounding box filtering)
        # ----------------------------------------------------------------
        # Vectors from the car heading +/- max bearing
        farthest_range_vectors = np.array([
            [np.cos(car_heading),              np.sin(car_heading)],
            [np.cos(car_heading + sensor_max_bearing), np.sin(car_heading + sensor_max_bearing)],
            [np.cos(car_heading - sensor_max_bearing), np.sin(car_heading - sensor_max_bearing)]
        ])
        norms = np.linalg.norm(farthest_range_vectors, axis=1, keepdims=True)
        farthest_range_vectors = farthest_range_vectors / norms * sensor_range
        
        # Stack them with the car position
        outer_points = np.vstack((car_pos, farthest_range_vectors + car_pos))
        x_min, x_max = outer_points[:, 0].min() - self.bounding_box_buffer, outer_points[:, 0].max() + self.bounding_box_buffer
        y_min, y_max = outer_points[:, 1].min() - self.bounding_box_buffer, outer_points[:, 1].max() + self.bounding_box_buffer
        
        # ----------------------------------------------------------------
        # 2) Gather occlusions and OOIs in a single list for sorting
        # ----------------------------------------------------------------
        object_list = []  # each entry: dict with keys: "object_type", "index", "mean", "radius", "bearing_minmax", etc.
        
        # ---- A) Occlusions (assuming circles for simplicity) ----
        for i in range(self.num_occlusion):
            center = self.occlusion_means[i]
            radius = self.occlusion_radii[i]
            
            # bounding-box filter
            if (center[0] < x_min) or (center[0] > x_max) or (center[1] < y_min) or (center[1] > y_max):
                continue
            
            dist_car_to_center = np.linalg.norm(center - car_pos)
            
            # If you want to mimic the old "range -= radius" for sorting:
            sort_range = dist_car_to_center - radius
            
            object_list.append({
                "object_type": "occlusion",
                "shape":       "circle",
                "index":       i,  # occlusion index
                "center":      center,
                "radius":      radius,
                "range":       sort_range if sort_range > 0 else dist_car_to_center
            })
        
        # ---- B) OOIs (4-corner polygons) ----
        #     We'll treat each OOI with bounding circle for sorting & occlusion logic
        for i in range(self.num_oois):
            corners = self.oois[i]  # shape = (4,2)
            # OOI center is average of corners
            center = corners.mean(axis=0)
            
            # bounding-box filter
            if (center[0] < x_min) or (center[0] > x_max) or (center[1] < y_min) or (center[1] > y_max):
                continue
            
            dist_car_to_center = np.linalg.norm(center - car_pos)
            # bounding radius is the max corner distance from center
            corner_offsets = corners - center
            bounding_radii = np.linalg.norm(corner_offsets, axis=1)
            max_radius = bounding_radii.max()
            
            # If you want to mimic the old "range -= radius" for sorting:
            sort_range = dist_car_to_center - max_radius
            
            object_list.append({
                "object_type": "ooi",
                "index":       i,  # OOI index
                "corners":     corners,
                "center":      center,
                "radius":      max_radius,  # bounding circle for the OOI
                "range":       sort_range if sort_range > 0 else dist_car_to_center
            })
        
        # If no OOIs made it into bounding box, no sense doing more
        # But we do want to handle occlusions that might appear for future logic if you prefer.
        # For now, if there are 0 "ooi" in object_list after bounding box filtering, we can return {}.
        any_ooi_in_list = any(obj["object_type"] == "ooi" for obj in object_list)
        if not any_ooi_in_list:
            return {}
        
        # ----------------------------------------------------------------
        # 3) Sort objects by ascending "range"
        # ----------------------------------------------------------------
        object_list.sort(key=lambda obj: obj["range"])
        
        # ----------------------------------------------------------------
        # 4) Iterate through objects in ascending order, building up occluded bearing intervals
        # ----------------------------------------------------------------
        occluded_bearings = np.empty((0, 2))  # each row = [bearing_min, bearing_max]
        observation_indices = {}  # { ooi_index: [corner_idx0, corner_idx1, ...], ... }
        
        for obj in object_list:
            if obj["object_type"] == "ooi":
                # --- Corner-by-corner check ---
                corners = obj["corners"]  # shape(4,2)
                ooi_idx = obj["index"]
                
                # bearings of corners in car's local heading frame
                corner_bearings = np.zeros(4, dtype=float)
                corner_ranges   = np.zeros(4, dtype=float)
                
                for c_idx in range(4):
                    dx = corners[c_idx, 0] - car_pos[0]
                    dy = corners[c_idx, 1] - car_pos[1]
                    bearing = wrap_angle(np.arctan2(dy, dx) - car_heading)
                    corner_bearings[c_idx] = bearing
                    corner_ranges[c_idx]   = np.sqrt(dx*dx + dy*dy)
                
                # Sort corners by range to find the "closest" corner
                sorted_indices = np.argsort(corner_ranges)
                closest_idx = sorted_indices[0]
                lower_neighbor_idx = (closest_idx - 1) % 4
                higher_neighbor_idx = (closest_idx + 1) % 4
                
                # The OOI itself blocks a small bearing range around the closest corner
                # old code: "pull out the bearing intervals which are occluded by the object itself"
                # We'll do the same approach:
                lower_to_closest = np.sort([corner_bearings[lower_neighbor_idx], corner_bearings[closest_idx]])
                closest_to_higher = np.sort([corner_bearings[closest_idx], corner_bearings[higher_neighbor_idx]])
                ooi_self_occluded = np.vstack((lower_to_closest, closest_to_higher))  # shape (2,2)
                
                # Determine which corners are NOT occluded by the OOI itself
                # i.e. a corner bearing is "allowed" if it is not in either interval
                is_corner_visible_from_ooi = np.ones(4, dtype=bool)  # we set to false if occluded by OOI
                for c_idx in range(4):
                    b = corner_bearings[c_idx]
                    # if b is in (bmin, bmax) for either interval, it is blocked
                    blocked_by_self = np.any((ooi_self_occluded[:,0] < b) & (b < ooi_self_occluded[:,1]))
                    if blocked_by_self:
                        is_corner_visible_from_ooi[c_idx] = False
                
                # Among corners that pass OOI-self test, check for occlusion by other objects:
                # We'll also check sensor range & bearing limit
                final_visible_corners = []
                for c_idx in np.where(is_corner_visible_from_ooi)[0]:
                    b = corner_bearings[c_idx]
                    r = corner_ranges[c_idx]
                    
                    # Check if b is in any occluded interval
                    blocked_by_others = np.any((occluded_bearings[:,0] < b) & (b < occluded_bearings[:,1]))
                    
                    # Check sensor range & max bearing
                    within_sensor_range  = (r < sensor_range)
                    within_bearing_limit = (abs(b) < sensor_max_bearing)

                    if (not blocked_by_others) and within_sensor_range and within_bearing_limit:
                        final_visible_corners.append(c_idx)
                
                if len(final_visible_corners) > 0:
                    # Record them in the observation_indices dictionary
                    observation_indices[ooi_idx] = final_visible_corners
                
                # Finally, add this OOI’s own min/max bearing to the global occluded intervals
                min_bearing = corner_bearings.min()
                max_bearing = corner_bearings.max()
                occluded_bearings = np.vstack((occluded_bearings, [min_bearing, max_bearing]))
                
            elif obj["object_type"] == "occlusion":
                # We assume "circle" shape for simplicity
                center = obj["center"]
                radius = obj["radius"]
                
                # Find bearing from car to center
                dx = center[0] - car_pos[0]
                dy = center[1] - car_pos[1]
                bearing_center = wrap_angle(np.arctan2(dy, dx) - car_heading)
                
                # The angle from center to edge:
                dist_car_to_center = np.hypot(dx, dy)
                # Avoid zero-dist edge case
                if dist_car_to_center < radius:
                    # Car is inside or extremely close to occlusion => entire bearing range is blocked.
                    # For safety, block everything from -pi to pi, or do something else:
                    occluded_bearings = np.vstack((occluded_bearings, [-np.pi, np.pi]))
                else:
                    # small angle from center line to each edge
                    half_angle = np.arcsin(radius / dist_car_to_center)
                    bearing_min = bearing_center - half_angle
                    bearing_max = bearing_center + half_angle
                    # add to occluded bearings
                    occluded_bearings = np.vstack((occluded_bearings, [bearing_min, bearing_max]))
                
            else:
                # If you support polygon occlusions, do the corner-based min/max bearing logic
                pass
        
        # ----------------------------------------------------------------
        # Return only the final observation_indices dictionary
        # ----------------------------------------------------------------
        return observation_indices

    def get_noisy_observation(self, car_state):
        """
        Get a noisy observation containing the observable ooi corners.
        Noise is modelled by range-bearing sensor model with Gaussian noise.
        
        returns:
        - observation_indices: Dictionary of observed OOI corners for displaying observation arrows
        - noisy_observation: Dictionary of noisy OOI corner positions
        """
        
        # Get the observation indices
        observation_indices = self.get_observation_indices(car_state)
        
        # Create the noisy observation
        noisy_observation = {}
        for ooi_idx, observed_corners in observation_indices.items():
            # Create a noisy corner for each observed corner
            noisy_corners = np.zeros((len(observed_corners), 2))
            
            # Loop through each observed corner and add noise using the measurement model
            for j, corner_idx in enumerate(observed_corners):
                # Get the corner of the OOI
                corner = self.oois[ooi_idx][corner_idx]
                
                # Use the measurement model to get the observation matrix
                observation_matrix = measurement_model(corner, car_state[0:2], car_state[3], 
                                                       range_dev=self.range_stddev, bearing_dev=self.bearing_stddev)
                
                # Add noise to the real corner using the observation matrix
                noisy_corners[j] = np.random.multivariate_normal(corner, observation_matrix)
        
            # Add the noisy corners to the noisy observation
            noisy_observation[ooi_idx] = noisy_corners
            
        return observation_indices, noisy_observation