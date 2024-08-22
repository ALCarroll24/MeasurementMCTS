import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import threading
import webbrowser

class MatPlotLibUI:
    def __init__(self, notebook=False, update_rate=10, figsize=(8, 8), single_plot=False, title=None):
        self.notebook = notebook
        self.figsize = figsize
        if not self.notebook:
            # Initialize the Matplotlib figure and axes
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            # self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.fig.canvas.mpl_connect('close_event', self.handle_close)
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Store the keys being pressed for later retrieval
        self.keys = []

        # Store the rectangles (car and tires) for later update
        self.patches = []
        
        # Text is an artist which we can also save for later update
        self.artists = []
        
        # Flag for stopping the plotting loop and title
        self.shutdown = False
        self.title = title
        
        # If single_plot is True, don't create buttons or setup for looping
        if not single_plot and not self.notebook:
            # Create Buttons
            play_button_ax = self.fig.add_axes([0.3, 0.94, 0.2, 0.03])  # Adjust as necessary
            self.play_button = Button(play_button_ax, 'Play/Pause', color='lightgoldenrodyellow', hovercolor='0.975')
            self.play_button.on_clicked(self.on_play_button_click)
            self.paused = False
            
            viz_button_ax = self.fig.add_axes([0.52, 0.94, 0.2, 0.03])  # Adjust as necessary
            self.viz_button = Button(viz_button_ax, 'Visualize', color='lightgoldenrodyellow', hovercolor='0.975')
            self.viz_button.on_clicked(self.on_viz_button_click)
        
        
            # Do initial update and show the plot
            self.update_display()
            plt.ion() # Turn on interactive mode for asynchronous drawing
            
            # Set the update rate and period
            self.rate = update_rate
            self.period = 1.0 / update_rate
        
    def single_plot(self):
        # Update the plot
        self.update_display()
        
        # Create a single plot
        plt.show(block=True)
            
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
        
        self.patches.append(rect)
        
    def draw_circle(self, center, radius, color='r'):
        """
        Draw a circle on the plot.
        :param center: Tuple (x, y) for the center of the circle.
        :param radius: Radius of the circle.
        """
        # Create a circle and add it to the plot
        circle = patches.Circle(center, radius, linewidth=1, edgecolor=color, facecolor='none')

        self.patches.append(circle)
        
    def draw_ellipse(self, center, width, length, angle=0, color='r', alpha=1.0, linestyle='-', linewidth=1):
        """
        Draw an ellipse on the plot.
        :param center: Tuple (x, y) for the center of the ellipse.
        :param width: Width of the ellipse.
        :param length: Length of the ellipse.
        :param angle: Rotation angle of the ellipse in degrees.
        """
        # Create an ellipse and add it to the plot
        ellipse = patches.Ellipse(center, width, length, edgecolor=color, facecolor='none', linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        ellipse.set_angle(np.degrees(angle))
        
        self.patches.append(ellipse)
        
    def draw_point(self, point, color='r', radius=0.5):
        """
        Draw a point on the plot.
        :param point: Tuple (x, y) for the point.
        :param color: Color of the point.
        """
        # Create a point and add it to the plot
        point = patches.Circle(point, radius, linewidth=1, edgecolor=color, facecolor=color)

        self.patches.append(point)
        
    def draw_arrow(self, start, end, color='b', width=1):
        """
        Draw an arrow on the plot.
        :param start: Tuple (x, y) for the start of the arrow.
        :param end: Tuple (x, y) for the end of the arrow.
        """
        # Create an arrow and add it to the plot
        arrow = patches.Arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], width=width, color=color)

        self.patches.append(arrow)
        
    def draw_polygon(self, points, color='b', linestyle='--', linewidth=1, closed=True, alpha=1.0):
        """
        Draw a path on the plot.
        :param path: List of tuples [(x1, y1), (x2, y2), ...] for the path.
        """
        # Create a scatter plot and add it to the plot with line markers
        x, y = zip(*points)
        poly_patch = patches.Polygon(np.column_stack((x, y)), closed=closed, edgecolor=color, facecolor='none', 
                                     linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        self.patches.append(poly_patch)
        
    def draw_text(self, text, position, color='b', fontsize=12):
        """
        Draw text on the plot.
        :param text: The text to display.
        :param position: Tuple (x, y) for the position of the text.
        """
        # Create text and add it to the plot
        text = plt.text(position[0], position[1], text, color=color, fontsize=fontsize)
        self.artists.append(text)

    def update_display(self):
        """
        Refresh the display with new positions and orientations.
        """
        if self.notebook:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
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
            
        # Redraw the text (or other artist objects)
        for artist in self.artists:
            self.ax.add_artist(artist)
                    
        # Remove rectangles from the list
        self.patches = []
        
    def get_artists(self, clear=True):
        """
        Get the artists from the plot.
        :param clear: Whether to clear the artists after getting them.
        :return: List of artists.
        """
        artists = self.patches + self.artists
        if clear:
            self.patches = []
            self.artists = []
        return artists
    
    def get_figure(self) -> tuple[plt.figure, plt.axes]:
        """
        Get the figure and axes of the plot.
        :return: Tuple of the figure and axes.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Set title
        ax.set_title(self.title)
        
        # Set the aspect of the plot to be equal
        ax.set_aspect('equal', adjustable='box')
        
        # Set plot limits and labels
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        return fig, ax

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

