import numpy as np
from shapely.geometry import Polygon, LineString
from typing import Tuple

def wrap_angle(angle):
    # Wrap an angle to the range [-pi, pi]
    return (angle + np.pi) % (2 * np.pi) - np.pi

def sat_value(value, max_value):
    # Saturate a value to the range [-max_value, max_value]
    return np.clip(value, -max_value, max_value)

# Old non-vectorized version
# def wrapped_angle_diff(angle1, angle2):
#     # Wrap both angles just to be safe
#     angle1 = wrap_angle(angle1)
#     angle2 = wrap_angle(angle2)
    
#     # Now find the closest angle between the two
#     diff = angle1 - angle2
#     if diff > np.pi:
#         diff -= 2*np.pi
#     elif diff < -np.pi:
#         diff += 2*np.pi
        
#     return diff

def wrapped_angle_diff(angle1, angle2):
    # Convert inputs to numpy arrays
    angle1 = np.asarray(angle1)
    angle2 = np.asarray(angle2)
    
    # Wrap both angles
    angle1 = wrap_angle(angle1)
    angle2 = wrap_angle(angle2)
    
    # Find the closest angle between the two
    diff = angle1 - angle2
    diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
    diff = np.where(diff < -np.pi, diff + 2 * np.pi, diff)
    
    return diff

def angle_in_interval(angle, low_angle, high_angle):
    # Wrap the angles just to be safe
    angle = wrap_angle(angle)
    low_angle = wrap_angle(low_angle)
    high_angle = wrap_angle(high_angle)
    
    # print("Angle:", np.degrees(angle), "Low angle:", np.degrees(low_angle), "High angle:", np.degrees(high_angle))

    # Adjust for intervals that cross the -π/π boundary
    if low_angle <= high_angle:
        return low_angle <= angle <= high_angle
    else:
        return angle >= low_angle or angle <= high_angle
    
def rotate(points: np.ndarray, angle):
    # Create rotation matrix and output array
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_points = np.zeros(points.shape)
    
    # If the points are 1D, we can just rotate them directly
    if points.ndim == 1:
        # Rotate a single point around the origin by a given angle
        return rot_matrix @ points
    else:
        # Iterate through the points
        for i, point in enumerate(points):
            # Rotate a point around the origin by a given angle
            rotated_points[i] = rot_matrix @ point
        
    return rotated_points


# def rotate(points: np.ndarray, angle: float) -> np.ndarray:
#     # Create rotation matrix
#     rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
#                            [np.sin(angle), np.cos(angle)]])
    
#     # Ensure points is at least 2D (nx2 or 1x2)
#     points = np.atleast_2d(points)
    
#     # Perform the rotation using matrix multiplication
#     rotated_points = points @ rot_matrix.T
    
#     # Return rotated points, ensuring original dimensionality is preserved
#     return rotated_points.squeeze()


def rotate_about_point(points: np.ndarray, angle, center):
    # Subtract the center from the points, rotate them, and add the center back
    return rotate(points - center, angle) + center

def min_max_normalize(value, min_value, max_value):
    # Normalize a value to the range [0, 1]
    return (value - min_value) / (max_value - min_value)

def angle_difference(angle1, angle2):
    # Find the difference between two angles
    return np.abs(np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2)))

def get_interpolated_polygon_points(polygon: Polygon, num_points: int = 200):
    line = LineString(polygon.exterior.coords)
    densified_coords = [line.interpolate(float(i) / num_points, normalized=True).coords[0] for i in range(num_points + 1)]
    return np.array(densified_coords)

def get_pixels_and_values(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a grid to pixel indices and corresponding values.
    :param grid: Grid to convert.
    :return: Tuple of pixel indices and corresponding values.
    """
    # Convert grid into W*Wx2 array where each row is a pixel and the columns are [x_idx, y_idx] combinations
    X_idxs, Y_idxs = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]))
    X_idxs_flat, Y_idxs_flat = X_idxs.ravel(), Y_idxs.ravel()
    pixel_indices = np.vstack([X_idxs_flat, Y_idxs_flat]).T # rows are now [x_idx, y_idx] combinations
    
    # Get the values of the cells in the explore grid that match the pixel indices indices
    values = grid[pixel_indices[:, 0], pixel_indices[:, 1]] # Values of cells matching the pixel indices
    
    return pixel_indices, values

def get_ellipse_scaling(cov):
    eigvals, eigvecs = np.linalg.eig(cov)

    # Angle of first eigen column vector
    eigvec1 = eigvecs[:,0] # First column
    eigvec1_angle = np.arctan2(eigvec1[1], eigvec1[0])
    
    # Return eigenvalues [width, height] and angle of first eigenvector (rotation)
    return eigvals, eigvec1_angle

# def find_least_rectangular_point(points, return_debug=False):
#     """
#     Finds the point in a set of four 2D points that forms the internal angle
#     farthest from 90 degrees.

#     Parameters:
#     - points: A (4, 2) NumPy array where each row represents a point (x, y).
#     - return_debug: If True, return the index of the least rectangular point, the angle at that point, and all internal angles.

#     Returns:
#     - index: The index of the least rectangular point.
#     """
#     if points.shape != (4, 2):
#         raise ValueError("Input must be a 4x2 NumPy array.")

#     angles = []
    
#     for i in range(4):
#         # Current point
#         Pi = points[i]
        
#         # Previous and next points (with wrap-around)
#         P_prev = points[i - 1]
#         P_next = points[(i + 1) % 4]
        
#         # Vectors
#         v1 = P_prev - Pi
#         v2 = P_next - Pi
        
#         # Compute the angle between v1 and v2
#         dot_prod = np.dot(v1, v2)
#         norm_v1 = np.linalg.norm(v1)
#         norm_v2 = np.linalg.norm(v2)
        
#         # Avoid division by zero
#         if norm_v1 == 0 or norm_v2 == 0:
#             angle = 0
#         else:
#             # Clamp the cosine value to the valid range [-1, 1] to avoid numerical issues
#             cos_theta = np.clip(dot_prod / (norm_v1 * norm_v2), -1.0, 1.0)
#             angle_rad = np.arccos(cos_theta)
#             angle = np.degrees(angle_rad)
        
#         angles.append(angle)
#         # print(f"Point {i}: Angle = {angle:.2f} degrees")

#     angles = np.array(angles)
    
#     # Compute absolute difference from 90 degrees
#     diff = np.abs(angles - 90)
    
#     # Find the index with the maximum difference
#     least_rect_idx = np.argmax(diff)
#     least_rect_angle = angles[least_rect_idx]
    
#     # print(f"\nLeast rectangular point is at index {least_rect_idx} with an angle of {least_rect_angle:.2f} degrees.\n")
    
#     if return_debug:
#         return least_rect_idx, least_rect_angle, angles
    
#     return least_rect_idx

def calculate_internal_angles(points):
    """
    Calculates the internal angles at each point of a quadrilateral.

    Parameters:
    - points: A (4, 2) NumPy array where each row represents a point (x, y).

    Returns:
    - angles: A NumPy array containing internal angles at each point in degrees.
    """
    if points.shape != (4, 2):
        raise ValueError("Input must be a 4x2 NumPy array.")

    angles = []

    for i in range(4):
        # Current point
        Pi = points[i]

        # Previous and next points (with wrap-around)
        P_prev = points[i - 1]
        P_next = points[(i + 1) % 4]

        # Vectors
        v1 = P_prev - Pi
        v2 = P_next - Pi

        # Compute the angle between v1 and v2
        dot_prod = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # Avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            angle = 0
        else:
            # Clamp the cosine value to the valid range [-1, 1] to avoid numerical issues
            cos_theta = np.clip(dot_prod / (norm_v1 * norm_v2), -1.0, 1.0)
            angle_rad = np.arccos(cos_theta)
            angle = np.degrees(angle_rad)

        angles.append(angle)
        # print(f"Point {i}: Angle = {angle:.2f} degrees")

    return np.array(angles)

def find_least_rectangular_point(points, return_debug=False):
    """
    Identifies the point in a set of four 2D points that makes the shape more like a triangle.

    Parameters:
    - points: A (4, 2) NumPy array where each row represents a point (x, y).
    - return_debug: If True, return the index of the least rectangular point, the angle at that point, and all internal angles.

    Returns:
    - least_rect_idx: The index of the least rectangular point.
    """
    angles = calculate_internal_angles(points)

    # Compute absolute difference from 180 degrees
    diff_from_180 = np.abs(angles - 180)

    # Find the index with the smallest difference
    least_rect_idx = np.argmin(diff_from_180)
    least_rect_angle = angles[least_rect_idx]

    # print(f"\nLeast rectangular point is at index {least_rect_idx} with an angle of {least_rect_angle:.2f} degrees.\n")

    if return_debug:
        return least_rect_idx, least_rect_angle, angles
    
    return least_rect_idx

def find_farthest_point(points, position):
    """
    Finds the point in a set of 2D points that is farthest from a given position.
    
    Parameters:
    - points: A (N, 2) NumPy array where each row represents a point (x, y).
    - position: A 2-element NumPy array representing the position (x, y).
    
    Returns:
    - farthest_idx: The index of the farthest point.
    """
    distances = np.linalg.norm(points - position, axis=1)
    farthest_idx = np.argmax(distances)
    
    return farthest_idx
    
