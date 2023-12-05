import numpy as np
from ui import MatPlotLibUI
from car import Car
from ooi import OOI
from vkf import VectorizedStaticKalmanFilter

if __name__ == '__main__':
    # Determine update rate
    hz = 20.0
    period = 1.0 / hz
    
    # Create a plotter object
    ui = MatPlotLibUI(update_rate=hz)
    
    # Create a car object
    car = Car(ui, np.array([15.0, 15.0]), 90, hz)
    
    # Create an OOI object
    ooi = OOI(ui)
    
    # Create a Static Vectorized Kalman Filter object
    vkf = VectorizedStaticKalmanFilter(np.array([50]*8), np.diag([8]*8), 4.0)
    
    # Loop until matplotlib window is closed (handled by the UI class)
    while(True):
    
        # Get the observation from the OOI, pass it to the KF for update
        observable_corners, indeces = ooi.get_observation(car)
        vkf.update(observable_corners, indeces)
        
        # Get the control inputs from the arrow keys, pass them to the car for update
        car.get_arrow_key_control()
        car.update(period)
    
        # Update the displays, and pause for the period
        car.draw()
        ooi.draw()
        vkf.draw(ui)
        ui.update()
    