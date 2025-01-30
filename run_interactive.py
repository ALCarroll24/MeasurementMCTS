import numpy as np
import pandas as pd
import timeit
import sys
import matplotlib.pyplot as plt
from time import sleep
sys.path.append('src/measurement_mcts')
from measurement_mcts.mcts.mcts import mcts_with_rollout, get_action_subtree
from measurement_mcts.mcts.tree_viz import render_pyvis
from measurement_mcts.environment.measurement_control_env import MeasurementControlEnvironment

# Initialize the environment
env = MeasurementControlEnvironment(interactive=True)
state = env.get_state()
env.draw_state(state)

# Parameters
learning_iterations = 200
explore_factor = 0.1
discount_factor = 1.0
rollout_method = 'random_same'

# Permanent part of title for the window
title_perm = f'Interactive MCTS searches\n\
LI={learning_iterations}, EF={explore_factor}, DF={discount_factor}, HL={env.horizon_length} RL={rollout_method}'

# Goal is to match simulation dt
dt = env.simulation_dt
search_count = 1
cumulative_reward = 0
root = None
title = title_perm
try:
    while True:
        # Plot and check leftover time
        start_loop_time = timeit.default_timer()
        env.draw_state(state)
        leftover_time = dt - (timeit.default_timer() - start_loop_time)
        print(f"Leftover time: {leftover_time}")
        
        # Run MCTS, get best action, and update state
        root, LI_comp = mcts_with_rollout(env, state, learning_iterations, explore_factor, discount_factor, 
                                          rollout_method, start_with_root=root, max_time=leftover_time)
        best_child = root.children[root.best_child()]
        best_action = best_child.action
        state, reward, done = best_child.state, best_child.reward, best_child.done
        
        # Update cumulative reward, root for next iteration, and title
        cumulative_reward += reward
        root = get_action_subtree(root, best_action)
        title = f'{title_perm}\nSearch: {search_count}, LI: {LI_comp}, {round(state[0][2]*2.23694),2} mph, CR: {round(cumulative_reward,2)}'
        search_count += 1
        print(f'Total Time: {timeit.default_timer() - start_loop_time}')
        
        if done:
            print(f"Done! Cumulative reward: {cumulative_reward}")
            break

except KeyboardInterrupt:
    print("\nExiting...")
