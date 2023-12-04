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