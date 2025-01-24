import numpy as np
import pandas as pd
import timeit
import sys
import matplotlib.pyplot as plt
from time import sleep
# Add measurement mcts python package to path
sys.path.append('../src/measurement_mcts')
from measurement_mcts.mcts.mcts import mcts_search, get_best_trajectory, MCTSNode, DummyNode
from measurement_mcts.mcts.tree_viz import render_pyvis
from measurement_mcts.state_evaluation.reinforcement_learning import MCTSRLWrapper, plot_state_image
from measurement_mcts.environment.measurement_control_env import MeasurementControlEnvironment


env = MeasurementControlEnvironment(init_reset=False, interactive=True)
env.load_state('../state_configurations', 'evaluation1')
state = env.get_state()
# state_list = list(state)
state[0][2] = 4
dt = 0.6
time = 5

for i in range(int(5/dt)):
    env.draw_state(state)
    state, reward, done = env.step(state, [0., 1.], dt=dt)
    sleep(dt)

