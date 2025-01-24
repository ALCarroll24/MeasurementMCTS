
import numpy as np
import pandas as pd
import timeit
import sys
import cProfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
# Add measurement mcts python package to path
sys.path.append('../../src/measurement_mcts')
from measurement_mcts.mcts.mcts import mcts_search, get_best_trajectory, MCTSNode, DummyNode
from measurement_mcts.mcts.tree_viz import render_pyvis
from measurement_mcts.environment.measurement_control_env import MeasurementControlEnvironment

env = MeasurementControlEnvironment(init_reset=False)
env.load_state('../../state_configurations', 'evaluation2')
starting_state = env.get_state()


learning_iterations = 25
explore_factor = 100
discount_factor = 1.0

def one_action_mcts(action, learning_iterations=30):
    root = MCTSNode(env, starting_state, action=None, explore_factor=explore_factor, 
                    discount_factor=discount_factor, parent=DummyNode())
    root.maybe_add_child(action)
    action_root = root.children[action]
    
    for i in range(learning_iterations):
        leaf = action_root.select_leaf() # Select with UCB up to the leaf node and do one environment step
        
        # Add the transition to the replay buffer for training (except for the root node)
        # child_priors, value_estimate = eval.inference(leaf.state) # Inference the model to get the probability of each action and the value estimate
        child_priors, value_estimate = np.ones([env.N]) / env.N, 0. # Even probability for each action for testing and no expected reward to go
        
        leaf.expand(child_priors) # Expand the leaf node with the child priors
        leaf.backup(value_estimate) # Backup the value estimate up the tree to the root node
        
    return action_root.Q + action_root.U, action_root

def parallel_mcts(action_space):
    """
    Parallelizes the MCTS using multiprocessing.Pool.
    """
    pool = mp.Pool(processes=len(action_space))
    
    start_time = timeit.default_timer()
    
    # Distribute the actions to processes
    results = pool.map(one_action_mcts, np.arange(len(action_space)))
    
    pool.close()
    pool.join()
    
    print(timeit.default_timer() - start_time)
    
    # Find the action with the best value
    # best_action, best_value = max(results, key=lambda x: x[1])
    
    return results
    # return best_action, best_value


print(parallel_mcts(env.action_space))


# import multiprocessing as mp
# import numpy as np
# import timeit

# print("Number of processors: ", mp.cpu_count())

# # Prepare data
# np.random.RandomState(100)
# arr = np.random.randint(0, 10, size=[900000, 5])
# data = arr.tolist()
# data[:5]
# len(data)


# # Solution Without Paralleization

# def howmany_within_range(row, minimum, maximum):
#     """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
#     count = 0
#     for n in row:
#         if minimum <= n <= maximum:
#             count = count + 1
#     return count

# results = []

# start_time = timeit.default_timer()
# for row in data:
#     results.append(howmany_within_range(row, minimum=4, maximum=8))
# comp_time = timeit.default_timer() - start_time

# print(comp_time)
# print(results[:10])



# # # Parallelizing using Pool.apply()

# # # Step 1: Init multiprocessing.Pool()
# # pool = mp.Pool(mp.cpu_count())

# # # Step 2: `pool.apply` the `howmany_within_range()`
# # # start_time = timeit.default_timer()
# # results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]
# # # end_time = timeit.default_timer() - start_time

# # # Step 3: Don't forget to close
# # pool.close()

# # # print(end_time)
# # print(results[:10])



# # Parallelizing using Pool.map()
# import multiprocessing as mp

# # Redefine, with only 1 mandatory argument.
# def howmany_within_range_rowonly(row, minimum=4, maximum=8):
#     count = 0
#     for n in row:
#         if minimum <= n <= maximum:
#             count = count + 1
#     return count

# pool = mp.Pool(mp.cpu_count())
# start_time = timeit.default_timer()

# results = pool.map(howmany_within_range_rowonly, [row for row in data])

# pool.close()
# comp_time = timeit.default_timer() - start_time

# print(comp_time)
# print(results[:10])


# # Parallel processing with Pool.apply_async()

# pool = mp.Pool(mp.cpu_count())

# results = []

# # Step 1: Redefine, to accept `i`, the iteration number
# def howmany_within_range2(i, row, minimum, maximum):
#     """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
#     count = 0
#     for n in row:
#         if minimum <= n <= maximum:
#             count = count + 1
#     return (i, count)


# # Step 2: Define callback function to collect the output in `results`
# def collect_result(result):
#     global results
#     results.append(result)


# # Step 3: Use loop to parallelize
# for i, row in enumerate(data):
#     pool.apply_async(howmany_within_range2, args=(i, row, 4, 8), callback=collect_result)

# # Step 4: Close Pool and let all the processes complete    
# pool.close()
# pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

# # Step 5: Sort results [OPTIONAL]
# results.sort(key=lambda x: x[0])
# results_final = [r for i, r in results]

# print(results_final[:10])
# #> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]
