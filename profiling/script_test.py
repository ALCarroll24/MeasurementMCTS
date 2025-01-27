import cProfile
import sys
# MCTS code imports
sys.path.append("../src/measurement_mcts")  # Adds higher directory to python modules path.
from measurement_mcts.environment.measurement_control_env import MeasurementControlEnvironment
from measurement_mcts.mcts.mcts import mcts_search, get_best_trajectory

env = MeasurementControlEnvironment(enable_explore_grid=False)
env.reset()
state = env.get_state()

state, reward, done = env.step(state, [1,.5])