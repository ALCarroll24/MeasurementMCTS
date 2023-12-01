import matplotlib.pyplot as plt
import matplotlib.patches as patches
import threading

class MatPlotLibUI:
    def __init__(self, update_rate=10, figsize=(8, 8), start=False):
        # Initialize the Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)

        # Store the rectangles (car and tires) for later update
        self.rectangles = []
        
        # Do initial update and show the plot
        self.update_display()
        plt.ion() # Turn on interactive mode for asynchronous drawing
        
        # Set the update rate and period
        self.rate = update_rate
        self.period = 1.0 / update_rate
        
        # THIS DOESN'T WORK (matplotlib doesn't like multithreading)
        # Spawn a thread to update the plot asynchronously
        # self.thread = threading.Thread(target=self.async_loop)
        # self.thread.start()
        
        # Flag for stopping the plotting loop
        self.run_loop = True
        
        # If start is true, start the plotting loop
        if start:
            self.start()

    # Runs a loop to update the plot asynchronously
    def start(self):
        while self.run_loop:
            # Update the plot every 1/rate seconds
            self.update_display()
            plt.draw()
            plt.pause(self.period)
            
    def update(self):
        # Update the plot
        self.update_display()
        plt.draw()
        plt.pause(self.period)

    def handle_close(self, event):
        # Handle what happens when the window is closed
        print("Matplotlib window closed.")
        self.run_loop = False  # Set a global flag to stop the main loop

    def draw_rectangle(self, center, width, length, angle=0):
        """
        Draw a rectangle on the plot.
        :param center: Tuple (x, y) for the center of the rectangle.
        :param width: Width of the rectangle.
        :param length: Length of the rectangle.
        :param angle: Rotation angle of the rectangle in degrees.
        """
        # Calculate bottom left corner from center
        bottom_left = (center[0] - width / 2, center[1] - length / 2)

        # Create a rectangle and add it to the plot
        rect = patches.Rectangle(bottom_left, width, length, linewidth=1, edgecolor='r', facecolor='none')
        rect.set_angle(angle)
        self.ax.add_patch(rect)
        self.rectangles.append(rect)
        
        # self.update_display()


    def update_display(self):
        """
        Refresh the display with new positions and orientations.
        """
        # Clear the current axes
        self.ax.clear()
        
        # Set the aspect of the plot to be equal
        self.ax.set_aspect('equal', adjustable='box')
        
        # Set plot limits and labels
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')

        # Redraw the rectangles
        for rect in self.rectangles:
            self.ax.add_patch(rect)
            
        # Remove rectangles from the list
        self.rectangles = []

        # Redraw the plot
        # plt.draw()

    def on_click(self, event):
        """
        Handle mouse click events to move the car.
        """
        print(f"Mouse clicked at ({event.xdata}, {event.ydata})")
        
        self.draw_rectangle((event.xdata, event.ydata), 4, 8, 45)

    def create_dpad(self):
        """
        Create a simple directional pad for basic movement.
        """
        # This method will be implemented later for interactive control
        pass

if __name__ == '__main__':
    # Create an instance of the MatPlotLibUI class
    ui = MatPlotLibUI()
    ui.start()

