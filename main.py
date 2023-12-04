import numpy as np
from plotter_ui import MatPlotLibUI
from car import Car
from ooi import OOI

if __name__ == '__main__':
    # Determine update rate
    hz = 20.0
    period = 1.0 / hz
    
    # Create a plotter object
    ui = MatPlotLibUI(update_rate=hz)
    
    # Create a car object
    car = Car(ui, np.array([50.0, 40.0]), 90, hz)
    
    # Create an OOI object
    ooi = OOI(ui)
    
    # Loop until matplotlib window is closed (handled by the UI class)
    while(True):
    
        # Get the observation from the OOI and draw it
        observable_corners = ooi.get_observation(car)
        ooi.draw()
    
        # Update the car and ui
        car.get_arrow_key_control()
        car.update(period)
        ui.update()
    