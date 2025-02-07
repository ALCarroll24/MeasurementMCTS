import numpy as np
import os
import pprint

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
    print(all_traces)
    
    # Sum the traces and calculate the percentage
    sum_traces = np.sum(all_traces)
    print(f'sum_traces: {sum_traces}')
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