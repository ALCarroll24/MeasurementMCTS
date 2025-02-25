import numpy as np
import pandas as pd
import timeit
import sys
import matplotlib.pyplot as plt
from time import sleep
sys.path.append('src/measurement_mcts')
from measurement_mcts.mcts.mcts import mcts_with_rollout, get_action_subtree
from measurement_mcts.state_evaluation.hertg import HERTG
from measurement_mcts.mcts.tree_viz import render_pyvis
from measurement_mcts.environment.measurement_control_env import MeasurementControlEnvironment

# Initialize the environment
env = MeasurementControlEnvironment(init_reset=False, interactive=True)
env.reset()

# Save an interesting state configuration to a file
# env.save_state('state_configurations', 'hertg_test1')

# If desired load a state from a file
# env.load_state('state_configurations', 'hertg_test1')

# Set and draw the initial state
state = env.get_state()
env.draw_state(env.get_state())

# Parameters
learning_iterations = 500
explore_factor = 0.1
discount_factor = 1.0
rollout_method = 'accelerate'
hertg_method = 'static' # 'static' or 'dynamic'
env.horizon_length = 3
env.obstacle_discount_factor = 0.8
env.car_collision_radius = env.object_manager.car_collision_radius = 5
env.fully_observed_corner_reward = 0.02 #0.1
env.obstacle_punishment = -0.003
hertg_reward_scale = 0.1
rollout_pre_collision_stop = False # When true decellerates car when collision is predicted
dynamic_learning_iterations = False # When true varies LI to match the timestep

# Pause initially if wanted
# env.ui.paused = True

# Create hertg object
hertg = HERTG(state, env, hertg_method, reward_scale=hertg_reward_scale)

# Permanent part of title for the window
title_perm = f'Interactive MCTS searches\n\
LI={learning_iterations}, EF={explore_factor}, DF={discount_factor}, HL={env.horizon_length} RL={rollout_method}'

# If dynamic learning iterations enabled goal is to match simulation dt
dt = env.simulation_dt
search_count = 1
cumulative_reward = 0
root = subroot = None
observation = None
title = title_perm
first_pause=True
if dynamic_learning_iterations is False:
    LI_comp = learning_iterations

try:
    while not env.ui.shutdown:
        if env.ui.paused:
            if root is None:
                env.draw_state(state, title=title, hertg=hertg)
            else:
                env.draw_state(state, title=title, root_node=root, scaling=1000, bias=10, max=100000.,
                               rew=True, hertg=hertg, hertg_scale=50)
                if first_pause: # Only render the first time
                    render_pyvis(root, env.action_space, show_unsimulated=False)
                    env.save_state('state_configurations', 'paused_state')
                first_pause = False
            sleep(0.1)
            continue
        first_pause = True
        
        # Plot and check leftover time
        start_loop_time = timeit.default_timer()
        env.draw_state(state, title=title, observation=observation, hertg=hertg)#, root_node=root, scaling=4, bias=0.1, max=1., rew=True)
        leftover_time = dt - (timeit.default_timer() - start_loop_time)
        print(f"Leftover time: {leftover_time}")
        
        # Run MCTS, get best action, and update state
        if dynamic_learning_iterations:
            root, LI_comp = mcts_with_rollout(env, state, learning_iterations, explore_factor, discount_factor, 
                                              rollout_method, rollout_pre_collision_stop=rollout_pre_collision_stop,
                                              start_with_root=None, max_time=leftover_time, hertg=hertg)
        else:
            root = mcts_with_rollout(env, state, learning_iterations, explore_factor, discount_factor, 
                                     rollout_method, rollout_pre_collision_stop=rollout_pre_collision_stop,
                                     start_with_root=None, max_time=None, hertg=hertg)

        best_action_idx = np.argmax(root.child_Q())
        state, reward, done, observation, failure = env.step(state, env.action_space[best_action_idx], 
                                                    return_observation=True, check_failure=True)
        
        if failure:
            print(f"Failure! Cumulative reward: {cumulative_reward}")
            env.ui.paused = True
            continue
        
        # Reset the depth of the state to 0
        state_list = list(state)
        state_list[3] = 0
        state = tuple(state_list)
        
        # Update cumulative reward, root for next iteration, and title
        cumulative_reward += reward
        # subroot = get_action_subtree(root, best_action_idx)
        best_action = env.action_space[best_action_idx]
        title = f'{title_perm}\nAction: {best_action}, Search: {search_count}, LI: {LI_comp}, {round(state[0][2]*2.23694,2)} mph, CR: {round(cumulative_reward,2)}'
        print(f'Total Time: {timeit.default_timer() - start_loop_time}')
        search_count += 1
        
        if done:
            print(f"Done! Cumulative reward: {cumulative_reward}")
            env.ui.paused = True
        

except KeyboardInterrupt:
    print("\nExiting...")
