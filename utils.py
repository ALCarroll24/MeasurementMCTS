import numpy as np

def wrap_angle(angle):
    # Wrap an angle to the range [-pi, pi]
    return (angle + np.pi) % (2 * np.pi) - np.pi

def sat_value(value, max_value):
    # Saturate a value to the range [-max_value, max_value]
    return np.clip(value, -max_value, max_value)

def wrapped_angle_diff(angle1, angle2):
    # Wrap both angles just to be safe
    angle1 = wrap_angle(angle1)
    angle2 = wrap_angle(angle2)
    
    # Now find the closest angle between the two
    diff = angle1 - angle2
    if diff > np.pi:
        diff -= 2*np.pi
    elif diff < -np.pi:
        diff += 2*np.pi
        
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
    
    # Iterate through the points
    for i, point in enumerate(points):
        # Rotate a point around the origin by a given angle
        rotated_points[i] = rot_matrix @ point
        
    return rotated_points