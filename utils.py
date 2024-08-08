import numpy as np
from shapely.geometry import Polygon, LineString

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