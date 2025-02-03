import numpy as np
import pandas as pd
import timeit
import sys
import matplotlib.pyplot as plt
from time import sleep
import pygame
# Add measurement mcts python package to path
sys.path.append('./src/measurement_mcts')
from measurement_mcts.mcts.mcts import get_best_trajectory, MCTSNode, DummyNode
from measurement_mcts.state_evaluation.hertg import get_hertg_target_point, get_target_point_follow_action
from measurement_mcts.mcts.tree_viz import render_pyvis
from measurement_mcts.state_evaluation.reinforcement_learning import MCTSRLWrapper, plot_state_image
from measurement_mcts.environment.measurement_control_env import MeasurementControlEnvironment

# Initialize the environment
env = MeasurementControlEnvironment(init_reset=False, interactive=True)
state = env.reset()
env.draw_state(state)

# Initialize Pygame and the joystick module
pygame.init()
pygame.joystick.init()

# Check for connected controllers
if pygame.joystick.get_count() == 0:
    print("No controller detected.")
    exit()

# Initialize the first joystick/controller
joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Controller detected: {joystick.get_name()}")
dt = 0.1  # Time step in seconds
dt_ms = int(dt * 1000)

try:
    while True:
        # Track loop start time
        loop_start_ticks = pygame.time.get_ticks()

        # --- Controller Input & Processing ---
        pygame.event.pump()
        
        # Get left stick axes (X=0, Y=1)
        x_axis = joystick.get_axis(0)
        y_axis = joystick.get_axis(1)

        # Apply dead zone
        dead_zone = 0.25
        x_axis = 0.0 if abs(x_axis) < dead_zone else x_axis
        y_axis = 0.0 if abs(y_axis) < dead_zone else y_axis

        # Compute desired action values (pre-snap)
        target_long = -y_axis  # Invert y-axis for intuitive forward=positive acceleration
        target_steer = -x_axis

        # Define available options (replace with your actual arrays)
        long_acc_options = np.array([-1., -0.5, 0., 0.5, 1.])
        steering_acc_options = np.array([-1., -0.25, 0., 0.25, 1.])

        # Snap to closest valid action
        snapped_long = long_acc_options[np.argmin(np.abs(long_acc_options - target_long))]
        snapped_steer = steering_acc_options[np.argmin(np.abs(steering_acc_options - target_steer))]

        # Final snapped action
        action = [snapped_long, snapped_steer]

        # # Format action (swap axes for intuitive control)
        # action = [-y_axis, -x_axis]
        # print(f'Action: {action}')

        print(f'Action: {action}')

        # Plot the hertg target point for debugging
        get_hertg_target_point(state, env, ui=env.ui)
        
        # --- Environment Update ---
        state, reward, done = env.step(state, action, dt=dt)
        env.draw_state(state)
        # print(f'Speed: {state[0][2]*2.23694} mph')

        # --- Dynamic Sleep Adjustment ---
        elapsed_time = pygame.time.get_ticks() - loop_start_ticks
        remaining_sleep = dt_ms - elapsed_time
        
        if remaining_sleep > 0:
            pygame.time.delay(remaining_sleep)  # Sleep only if there's time left

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    pygame.quit()
