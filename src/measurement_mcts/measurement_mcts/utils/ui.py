import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import threading
import webbrowser

class MatPlotLibUI:
    def __init__(self, title=None, interactive=False):
        # Store the rectangles (car and tires) for later update
        self.patches = []
        
        # Text is an artist which we can also save for later update
        self.artists = []
        
        # Store the background image (if any)
        self.background_image = None
        
        # Flag for stopping the plotting loop and title
        self.shutdown = False
        self.title = title
        
        self.interactive = interactive
        if interactive:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(32, 32))
            
            # Handle close event
            self.fig.canvas.mpl_connect('close_event', self.handle_close)
            
            # Create Buttons
            play_button_ax = self.fig.add_axes([0.4, 0.97, 0.2, 0.03])  # Adjust as necessary
            self.play_button = Button(play_button_ax, 'Play/Pause', color='lightgoldenrodyellow', hovercolor='0.975')
            self.play_button.on_clicked(self.on_play_button_click)
            self.paused = False
            
        self.reset_future_state_data()

    def reset_future_state_data(self):
        self.ucb_states_pos = ([], [])
        self.ucb_rewards_pos = 20
        self.ucb_states_neg = ([], [])
        self.ucb_rewards_neg = 20
        self.rollout_states_pos = ([], [])
        self.rollout_rewards_pos = 20
        self.rollout_states_neg = ([], [])
        self.rollout_rewards_neg = 20
        self.hertg_states_pos = ([], [])
        self.hertg_rewards_pos = 20

    def update_future_state_data(self, ucb_states_pos, ucb_rewards_pos, ucb_states_neg, ucb_rewards_neg,
                                 rollout_states_pos, rollout_rewards_pos, rollout_states_neg, rollout_rewards_neg,
                                 hertg_states_pos, hertg_rewards_pos):
        """
        Update the future state data for the scatter plots.
        """
        self.ucb_states_pos = ucb_states_pos
        self.ucb_rewards_pos = ucb_rewards_pos
        self.ucb_states_neg = ucb_states_neg
        self.ucb_rewards_neg = ucb_rewards_neg
        self.rollout_states_pos = rollout_states_pos
        self.rollout_rewards_pos = rollout_rewards_pos
        self.rollout_states_neg = rollout_states_neg
        self.rollout_rewards_neg = rollout_rewards_neg
        self.hertg_states_pos = hertg_states_pos
        self.hertg_rewards_pos = hertg_rewards_pos

    def handle_close(self, event):
        # Handle what happens when the window is closed
        print("Matplotlib window closed.")
        self.shutdown = True  # Set a global flag to stop the main loop        
        
    def on_play_button_click(self, event):
        print("play/pause button clicked")
        self.paused = not self.paused

    def plot(self, get_fig_ax: bool=False, title:str=None, figsize:tuple=(32, 32)):
        """
        Refresh the display with new positions and orientations.
        """
        
        if not self.interactive:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = self.fig
            ax = self.ax
            
        ax.scatter(*self.ucb_states_pos, marker='o', color='green', s=self.ucb_rewards_pos, alpha=1.0)
        ax.scatter(*self.ucb_states_neg, marker='o', color='red', s=self.ucb_rewards_neg, alpha=1.0)
        ax.scatter(*self.rollout_states_pos, marker='D', color='purple', s=self.rollout_rewards_pos, alpha=0.3)
        ax.scatter(*self.rollout_states_neg, marker='D', color='red', s=self.rollout_rewards_neg, alpha=0.3)
        ax.scatter(*self.hertg_states_pos, marker='*', color='orange', s=self.hertg_rewards_pos, alpha=0.5)
        
        self.reset_future_state_data()
        
        # Set title
        if title is not None:
            ax.set_title(title)
        
        # Set the aspect of the plot to be equal
        ax.set_aspect('equal', adjustable='box')
        
        # Set plot limits and labels
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Draw the background image
        if self.background_image is not None:
            image, extent, alpha = self.background_image
            ax.imshow(image, extent=extent, alpha=alpha)

        # Redraw the rectangles
        for rect in self.patches:
            ax.add_patch(rect)
            
        # Redraw the text (or other artist objects)
        for artist in self.artists:
            ax.add_artist(artist)
                    
        # Remove patches, artists and background image
        self.patches = []
        self.artists = []
        self.background_image = None
        
        
        if get_fig_ax:
            return fig, ax
        
        if self.interactive:
            plt.draw()
            fig.canvas.flush_events()
            ax.clear()

    def draw_background_image(self, image, extent, alpha=1.0):
        """
        Draw a background image on the plot.
        :param image: Image to draw.
        :param extent: Extent of the image.
        """
        self.background_image = (image, extent, alpha)

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
        
    def draw_circle(self, center, radius, alpha=1.0, color='r', facecolor='none'):
        """
        Draw a circle on the plot.
        :param center: Tuple (x, y) for the center of the circle.
        :param radius: Radius of the circle.
        """
        # Create a circle and add it to the plot
        circle = patches.Circle(center, radius, linewidth=1, alpha=alpha, edgecolor=color, facecolor=facecolor)

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
        
    def draw_point(self, point, color='r', radius=0.5, alpha=1.0):
        """
        Draw a point on the plot.
        :param point: Tuple (x, y) for the point.
        :param color: Color of the point.
        """
        # Create a point and add it to the plot
        point = patches.Circle(point, radius, linewidth=1, edgecolor=color, facecolor=color, alpha=alpha)

        self.patches.append(point)
        
    def draw_arrow(self, start, end, color='b', width=1, alpha=1.0):
        """
        Draw an arrow on the plot.
        :param start: Tuple (x, y) for the start of the arrow.
        :param end: Tuple (x, y) for the end of the arrow.
        """
        # Create an arrow and add it to the plot
        arrow = patches.Arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], width=width, color=color, alpha=alpha)

        self.patches.append(arrow)
        
    def draw_polygon(self, points, color='b', linestyle='--', linewidth=1, closed=True, alpha=1.0, facecolor='none'):
        """
        Draw a path on the plot.
        :param path: List of tuples [(x1, y1), (x2, y2), ...] for the path.
        """
        # Create a scatter plot and add it to the plot with line markers
        x, y = zip(*points)
        poly_patch = patches.Polygon(np.column_stack((x, y)), closed=closed, edgecolor=color, facecolor=facecolor, 
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

    # Old not used    
    # def get_figure(self) -> tuple[plt.figure, plt.axes]:
    #     """
    #     Get the figure and axes of the plot.
    #     :return: Tuple of the figure and axes.
    #     """
    #     fig, ax = plt.subplots(figsize=self.figsize)
        
    #     # Set title
    #     ax.set_title(self.title)
        
    #     # Set the aspect of the plot to be equal
    #     ax.set_aspect('equal', adjustable='box')
        
    #     # Set plot limits and labels
    #     ax.set_xlim(0, 100)
    #     ax.set_ylim(0, 100)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
        
    #     return fig, ax

