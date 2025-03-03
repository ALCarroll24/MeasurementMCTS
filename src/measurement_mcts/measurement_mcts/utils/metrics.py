import numpy as np
import timeit
import os
import pprint
import sys
from copy import deepcopy
# Add measurement mcts python package to path
sys.path.append('../src/measurement_mcts')
from measurement_mcts.mcts.mcts import mcts_with_rollout
from measurement_mcts.state_evaluation.hertg import HERTG
from measurement_mcts.environment.measurement_control_env import MeasurementControlEnvironment

def get_percent_done(state, env):
    # First calculate total trace at start
    total_trace = env.init_covariance_trace
    final_coner_trace = env.final_corner_cov_trace
    
    # Then calculate trace of ooi cov
    ooi_covs = state[2]
    per_ooi_traces = np.trace(ooi_covs, axis1=2, axis2=3)
    all_traces = per_ooi_traces.flatten()
    
    # Make any traces below the final corner trace 0
    all_traces[all_traces <= final_coner_trace] = 0
    
    # Sum the traces and calculate the percentage
    sum_traces = np.sum(all_traces)
    return (1 - sum_traces / total_trace) * 100


def save_environment_config(env, folder_path):
    """
    Save the configuration parameters of the environment to a text file in the specified folder.
    
    This function records the instance variables (and optionally, class variables)
    of the environment instance. It can be useful to refer back to these parameters later,
    especially after changes in the code.
    
    Parameters:
        env: An instance of your MeasurementControlEnvironment (or similar) class.
        folder_path (str): The path to the folder where the configuration file should be saved.
    """
    # Ensure the folder exists.
    os.makedirs(folder_path, exist_ok=True)
    
    config_file = os.path.join(folder_path, "env_config.txt")
    
    # Gather instance variables.
    config_data = {
        "instance_vars": env.__dict__
    }
    
    # Optionally, include non-callable class variables (those not starting with '__')
    class_vars = {
        k: v for k, v in env.__class__.__dict__.items()
        if not k.startswith("__") and not callable(v)
    }
    config_data["class_vars"] = class_vars
    
    # Write the configuration to the file in a nicely formatted way.
    with open(config_file, "w") as f:
        f.write(pprint.pformat(config_data, indent=4))
    
    print(f"Environment configuration saved to {config_file}")
    
    
def get_mcts_metrics(env, state, max_actions=200, LI=100,
                     EF=0.1, DF=1.0, HL=6, rollout_method='same',
                     hertg_method='static', no_hertg=False,
                     skip_rollout=False, rollout_pre_collision_stop=False) -> dict:
    """
    Run MCTS and return the metrics.
    params:
        env: the environment object
        state: the initial state of the environment
        max_actions: the maximum number of actions to take
        LI: the length of the interval for the HERTG method
        EF: the exploration factor for the HERTG method
        DF: the discount factor for the HERTG method
        rollout_method: the rollout method to use
        hertg_method: the HERTG method to use
        
    returns:
        metrics: a dictionary of metrics
    """
    # Take a copy of the state
    state = deepcopy(state)
    
    # Change environment parameters and create HERTG object
    env.horizon_length = HL
    hertg = HERTG(state, env, method=hertg_method)
    if no_hertg:
        hertg = None
    
    # Create metric trackers
    cumulative_reward = 0.
    num_actions = 0
    done = False
    failure=False
    start_time = timeit.default_timer()
    for i in range(max_actions):
        # Run MCTS and take the best action
        root = mcts_with_rollout(env, state, LI, EF, DF, rollout_method,
                                 rollout_pre_collision_stop, hertg=hertg, skip_rollout=skip_rollout)
        best_action_idx = np.argmax(root.child_Q())
        state, reward, done, failure = env.step(state, env.action_space[best_action_idx], check_failure=True)
        
        # Reset the horizon to 0
        state_list = list(state)
        state_list[3] = 0
        state = tuple(state_list)
        
        # Increment the cumulative reward and number of actions
        cumulative_reward += reward
        num_actions = i + 1
        if done:
            break
        
        if failure:
            print(f"Collision failure! Cumulative reward: {cumulative_reward}")
            break
        
    comp_time = timeit.default_timer() - start_time
    percent_done = get_percent_done(state, env)
    
    metrics = {
        'LI': LI,
        'EF': EF,
        'DF': DF,
        'HL': HL,
        'rollout_method': rollout_method,
        'hertg_method': hertg_method,
        'done': done,
        'percent_done': percent_done,
        'cumulative_reward': cumulative_reward,
        'num_actions': num_actions,
        'computation_time': comp_time,
        'computation_per_action': comp_time / num_actions,
        'rollout_pre_collision_stop': rollout_pre_collision_stop,
        'collision_failure': failure,
        'skip_rollout': skip_rollout,
        'no_hertg': no_hertg
    }
    
    return metrics

def worker_wrapper(trial_number, trial_config_name, rollout_method,
                   trial_config_path,
                   max_actions=200,
                   LI=100,
                   EF=0.1,
                   DF=1.0,
                   HL=6,
                   hertg_method='static',
                   no_hertg=False,
                   skip_rollout=False,
                   rollout_pre_collision_stop=False):
    """
    Worker function that:
      - Instantiates a fresh environment.
      - Loads a pre-saved trial configuration from file.
      - Retrieves the state and object true state.
      - Runs get_mcts_metrics using the given rollout method and other parameters.
    
    Parameters:
        trial_config_name (str): Name of the trial config file (without the .pkl extension)
        rollout_method (str): Rollout method to use.
        trial_config_path (str): Directory where trial configuration files are stored.
        max_actions, LI, EF, DF, hertg_method, rollout_pre_collision_stop:
            Additional parameters passed to get_mcts_metrics.
    
    Returns:
        dict: Metrics dictionary from get_mcts_metrics.
    """
    # Create a unique seed using process ID and trial number.
    seed = os.getpid() + trial_number
    np.random.seed(seed)  # Re-seed NumPy's RNG in this process.
    
    # Create a new environment instance.
    env = MeasurementControlEnvironment(init_reset=False)
    
    # Load the saved state and object configuration.
    # This call uses your custom load_state method.
    env.load_state(trial_config_path, trial_config_name)
    
    # Retrieve the state and true object state.
    state = env.get_state()
    
    # Run the MCTS metrics collection using the loaded configuration.
    metrics = get_mcts_metrics(
        env,
        state,
        max_actions=max_actions,
        LI=LI,
        EF=EF,
        DF=DF,
        HL=HL,
        rollout_method=rollout_method,
        hertg_method=hertg_method,
        no_hertg=no_hertg,
        skip_rollout=skip_rollout,
        rollout_pre_collision_stop=rollout_pre_collision_stop
    )
    
    # Optionally, record which trial configuration was used.
    metrics['trial_config'] = trial_config_name
    return metrics