import numpy as np
from utils import wrap_angle, wrapped_angle_diff

class OOI:
    def __init__(self, ui, position=(50,50), length=4, width=8):
        self.position = position
        
        # TODO: support yaw
        # self.yaw = yaw
        
        self.ui = ui
        self.length = length
        self.width = width
        
        # Calculate corner positions
        self.corners = self.get_corners()
        
    def get_corners(self):
        # Calculate corner positions
        corners = np.zeros((4, 2))
        corners[0, :] = self.position + np.array([self.width / 2, self.length / 2])
        corners[1, :] = self.position + np.array([self.width / 2, -self.length / 2])
        corners[2, :] = self.position + np.array([-self.width / 2, -self.length / 2])
        corners[3, :] = self.position + np.array([-self.width / 2, self.length / 2])
        
        # Rotate corners TODO: This rotates around (0,0) and needs to be done some other way
        # rotation_matrix = np.array([[np.cos(self.yaw), -np.sin(self.yaw)], [np.sin(self.yaw), np.cos(self.yaw)]])
        # corners = np.matmul(rotation_matrix, corners.T).T
        
        return corners
        
    def draw(self):
        # Draw main rectangle
        self.ui.draw_rectangle(self.position, self.width, self.length, 0)
        
        # Draw corners
        # for corner in self.corners:
        #     self.ui.draw_circle(corner, 0.6)
        
    def get_observation(self, car, draw=True):
        # Get needed parameters from car
        car_position = car.position
        car_yaw = car.yaw
        car_range = car.max_range
        car_max_bearing = np.radians(car.max_bearing)
        
        # Find which corners are observable
        observable_corners = []
        for corner in self.corners:
            # Calculate distance and bearing to corner
            distance = np.linalg.norm(corner - car_position)
            bearing = wrapped_angle_diff(np.arctan2(corner[1] - car_position[1], corner[0] - car_position[0]), car_yaw)
            # print("Car yaw:", np.degrees(car_yaw), "Corner bearing:", np.degrees(np.arctan2(corner[1] - car_position[1], corner[0] - car_position[0])))
            
            # Check if corner is observable
            if distance >= car_range:
                # print("Corner is ", distance - car_range, "m too far away")
                pass
                
            elif np.abs(bearing) >= car_max_bearing:
                print("Bearing is", np.degrees(np.abs(bearing) - car_max_bearing), "degrees too large")
            else:
                observable_corners.append(corner)   
                
        if draw:
            # Draw observable corners
            for corner in observable_corners:
                self.ui.draw_circle(corner, 0.6)
                
        return observable_corners
        
        
if __name__ == '__main__':
    ooi = OOI(None)
    
    print(ooi.corners)
    
    for corner in ooi.corners:
        print(corner)