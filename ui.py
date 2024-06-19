import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import threading
import webbrowser

class MatPlotLibUI:
    def __init__(self, update_rate=10, figsize=(8, 8), async_loop=False, title=None):
        # Initialize the Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=figsize)
        # self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # If title is not none, set the title of the plot
        if title is not None:
            self.title = title
            self.ax.set_title(title)
        
        # Create Buttons
        play_button_ax = self.fig.add_axes([0.3, 0.94, 0.2, 0.03])  # Adjust as necessary
        self.play_button = Button(play_button_ax, 'Play/Pause', color='lightgoldenrodyellow', hovercolor='0.975')
        self.play_button.on_clicked(self.on_play_button_click)
        self.paused = False
        
        viz_button_ax = self.fig.add_axes([0.52, 0.94, 0.2, 0.03])  # Adjust as necessary
        self.viz_button = Button(viz_button_ax, 'Visualize', color='lightgoldenrodyellow', hovercolor='0.975')
        self.viz_button.on_clicked(self.on_viz_button_click)
        
        # Store the keys being pressed for later retrieval
        self.keys = []

        # Store the rectangles (car and tires) for later update
        self.patches = []
        
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
        self.shutdown = False
        
        # If async_loop is True, start the asynchronous plotting loop
        if async_loop:
            self.start_async_loop()

    # Runs a loop to update the plot asynchronously
    def start_async_loop(self):
        while not self.shutdown:
            # Update the plot every 1/rate seconds
            # self.draw_rectangle((50, 50), 20, 20, np.radians(45))
            self.draw_ellipse((50, 50), 20, 10, np.radians(45))
            self.update_display()
            plt.draw()
            plt.pause(self.period)
            
    def update(self):
        # Update the plot
        self.keys = []          # Reset the keys, for incoming pause command
        self.update_display()
        plt.draw()
        plt.pause(self.period)

    def handle_close(self, event):
        # Handle what happens when the window is closed
        print("Matplotlib window closed.")
        self.shutdown = True  # Set a global flag to stop the main loop

    def draw_rectangle(self, center, width, length, angle=0, color='r'):
        """
        Draw a rectangle on the plot.
        :param center: Tuple (x, y) for the center of the rectangle.
        :param width: Width of the rectangle.
        :param length: Length of the rectangle.
        :param angle: Rotation angle of the rectangle in degrees.
        """
        # Calculate bottom left corner from center (rotation is compensated for by the rotation_point argument)
        bottom_left = center[0] - width/2, center[1] - length/2
        
        # Create a rectangle and add it to the plot
        rect = patches.Rectangle(bottom_left, width, length, linewidth=1, edgecolor=color, facecolor='none', rotation_point='center')
        rect.set_angle(np.degrees(angle))
        
        self.ax.add_patch(rect)
        self.patches.append(rect)
        
    def draw_circle(self, center, radius, color='r'):
        """
        Draw a circle on the plot.
        :param center: Tuple (x, y) for the center of the circle.
        :param radius: Radius of the circle.
        """
        # Create a circle and add it to the plot
        circle = patches.Circle(center, radius, linewidth=1, edgecolor=color, facecolor='none')
        self.ax.add_patch(circle)
        self.patches.append(circle)
        
    def draw_ellipse(self, center, width, length, angle=0, color='r'):
        """
        Draw an ellipse on the plot.
        :param center: Tuple (x, y) for the center of the ellipse.
        :param width: Width of the ellipse.
        :param length: Length of the ellipse.
        :param angle: Rotation angle of the ellipse in degrees.
        """
        # Create an ellipse and add it to the plot
        ellipse = patches.Ellipse(center, width, length, linewidth=1, edgecolor=color, facecolor='none')
        ellipse.set_angle(np.degrees(angle))
        
        self.ax.add_patch(ellipse)
        self.patches.append(ellipse)
        
    def draw_point(self, point, color='r'):
        """
        Draw a point on the plot.
        :param point: Tuple (x, y) for the point.
        :param color: Color of the point.
        """
        # Create a point and add it to the plot
        point = patches.Circle(point, 0.5, linewidth=1, edgecolor=color, facecolor=color)
        self.ax.add_patch(point)
        self.patches.append(point)
        
    def draw_arrow(self, start, end, color='b'):
        """
        Draw an arrow on the plot.
        :param start: Tuple (x, y) for the start of the arrow.
        :param end: Tuple (x, y) for the end of the arrow.
        """
        # Create an arrow and add it to the plot
        arrow = patches.Arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], width=1, color=color)
        self.ax.add_patch(arrow)
        self.patches.append(arrow)

    def update_display(self):
        """
        Refresh the display with new positions and orientations.
        """
        # Clear the current axes
        self.ax.clear()
        
        # Set title
        self.ax.set_title(self.title)
        
        # Set the aspect of the plot to be equal
        self.ax.set_aspect('equal', adjustable='box')
        
        # Set plot limits and labels
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')

        # Redraw the rectangles
        for rect in self.patches:
            self.ax.add_patch(rect)
                    
        # Remove rectangles from the list
        self.patches = []

    def on_play_button_click(self, event):
        print("play/pause button clicked")
        self.paused = not self.paused
        
    def on_viz_button_click(self, event):
        print("Visualize button clicked")
        webbrowser.open('http://127.0.0.1:5000')

    def on_click(self, event):
        """
        Handle mouse click events to move the car.
        """
        # Handle mouse being clicked not on plot
        if event.xdata is None or event.ydata is None:
            return
        
        print(f"Mouse clicked at ({event.xdata}, {event.ydata})")
        
        self.draw_rectangle((event.xdata, event.ydata), 4, 8, np.radians(45))

    def on_key_press(self, event):
        keys = []
        if event.key == 'up':
            keys.append('up')
        if event.key == 'down':
            keys.append('down')
        if event.key == 'left':
            keys.append('left')
        if event.key == 'right':
            keys.append('right')
            
        # print(f"Key pressed: {keys}")
        self.keys = keys

if __name__ == '__main__':
    # Create an instance of the MatPlotLibUI class
    ui = MatPlotLibUI()
    ui.start_async_loop()

