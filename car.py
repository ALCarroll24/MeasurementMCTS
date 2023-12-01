import numpy as np
import threading
from plotter_ui import MatPlotLibUI
import time

class Car:
    def __init__(self, ui, length=4.0, width=2.0):
        self.ui = ui
        self.length = length
        self.width = width
        self.position = np.array([10.0, 10.0])  # x, y
        self.yaw = 0.0  # Orientation
        self.velocity = 0.0
        self.steering_angle = 0.0

    def update(self, dt, draw=True):
        # Update car state using the bicycle model
        self.position[0] += self.velocity * np.cos(self.yaw) * dt
        self.position[1] += self.velocity * np.sin(self.yaw) * dt
        self.yaw += (self.velocity / self.length) * np.tan(self.steering_angle) * dt
        
        print("hi")
        if draw:
            self.draw()

    def calculate_steering(self, target):
        # Pure pursuit control logic to follow the target point
        # This is a placeholder, the actual implementation will depend on your specific requirements
        pass

    def draw(self):
        # Draw the car as a rectangle in the UI
        self.ui.draw_rectangle(self.position, self.width, self.length, np.degrees(self.yaw))
        print("we drawin")
        
    def test_actions(self):
        # Test drive the car around using a for loop
        self.velocity = 5.0
        self.steering_angle = np.radians(5.0)
        for i in range(100):
            print("we movin")
            self.update(0.1)
            time.sleep(0.1)
            

if __name__ == '__main__':
    # Create a plotter object
    ui = MatPlotLibUI()
    
    # Create a car object
    car = Car(ui)
    
    # Start the car action thread
    car_action_thread = threading.Thread(target=car.test_actions)
    car_action_thread.start()
    
    # Start the ui plotting loop
    ui.start()
    