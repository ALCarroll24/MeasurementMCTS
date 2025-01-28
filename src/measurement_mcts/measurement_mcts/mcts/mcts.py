import collections
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Tuple, Any
import multiprocessing as mp
from multiprocessing.sharedctypes import Array as mpArray
from functools import partial
import ctypes
import timeit
import time
import pickle
import sys

# Measurement MCTS python package imports
sys.path.append("..")  # Adds higher directory to python modules path.
# from state_evaluation.reinforcement_learning import MCTSRLWrapper

class Environment(ABC):
  """
  Abstract class for an environment that MCTS can run on
  Must implement step, evaluate, and N

  required methods:
    step: take a step in the environment (state, action -> state, reward, done)
    evaluate: evaluate the current state of the environment (state -> (child_priors, value_estimate))
    N: get the size of the action space (int)
  """

  @abstractmethod
  def step(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool]:
    """
    Take a step in the environment

    params:
      state: the current state of the environment (np.ndarray(shape=(1, N)))
      action: the action from in the action space to take (int)

    returns:
      new_state: the new state of the environment (np.ndarray(shape=(1, N)))
      reward: the reward of the action (float)
      done: whether the episode is done (bool)
    """
    pass

#   @abstractmethod
#   def evaluate(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
#     """
#     Evaluate the current state of the environment

#     params:
#       state: the current state of the environment (np.ndarray(shape=(1, N)))

#     returns:
#       child_priors: the prior probabilities of each action, probabilities of best action (np.ndarray(shape=(1, N)))
#       value_estimate: the value estimate of the current state (float)
#     """
#     pass

  @property
  def N(self):
    """Get the size of the action space"""
    pass

class MCTSNode:
    """
    MCTS Node class that represents a node in the MCTS tree
    init params:
        env: the environment to run MCTS on, class that inherits from Environment class
        state: the state of the node (np.ndarray(shape=(1, N)))
        action: the action that led to this state (int)
        explore_factor: the exploration factor for UCB in selection (float)
        discount_factor: the discount factor for the value estimate (float)
        parent: the parent node of this node (MCTSNode)
        done: whether the episode is done (bool)
        parallel: whether to run MCTS in parallel (bool)

    properties:
        number_visits: get/set the number of visits to this node
        total_value: get/set the total value of this node

    methods:
        child_Q: get child Q values based on the stored total value and number of visits
        child_U: get child U values (upper confidence bounds)
        best_child: get the best child based on the upper confidence bound
        select_leaf: from current node, select highest upper confidence bound node until at next leaf node
        maybe_add_child: add a child if it does not exist
        expand: expand the current node with the given child_priors
        backup: backpropogate the value estimate up the tree to the root node
    """
    def __init__(self, env: Environment, state: np.ndarray, action: np.ndarray,
                 explore_factor: float=1., discount_factor: float=0.9, reward: float=0.,
                 parent: 'MCTSNode'=None, done: bool=False, parallel: bool=False):
        # Initialize parameters
        self.env = env
        self.state = state
        self.action = action
        self.explore_factor = explore_factor
        self.discount_factor = discount_factor
        self.reward = reward
        self.parent = parent  # Optional[MCTSNode]
        self.done = done
        self.parallel = parallel
        self.is_expanded = False
        self.children = {}  # Dict[action, MCTSNode]
        self.child_priors = np.zeros([self.env.N], dtype=np.float32)

        if self.parallel:
            # Initialize shared arrays if parallel
            self.child_number_visits, self.shared_number_visits_base = create_shared_array((self.env.N,), ctypes.c_int)
            self.child_total_value, self.shared_total_value_base = create_shared_array((self.env.N,), ctypes.c_float)
            
            # Initialize lock for making sure shared variables are updated correctly
            self.lock = mp.Lock() # Does nothing if not parallel
        else:
            # Initialize numpy arrays if not parallel
            self.child_number_visits = np.zeros([self.env.N], dtype=np.int32)
            self.child_total_value = np.zeros([self.env.N], dtype=np.float32)
            self.lock = None # No lock if not parallel


    # These properties are designed to allow you to use values of this node despite the vectorized structure
    @property
    def prior(self):
        """Get the prior probability of the action that led to this node"""
        return self.parent.child_priors[self.action]
    
    @prior.setter
    def prior(self, value):
        """Set the prior probability of the action that led to this node"""
        self.parent.child_priors[self.action] = value

    @property
    def number_visits(self):
        """Get the number of visits to this node."""
        if self.lock:
            with self.lock:
                return self.parent.child_number_visits[self.action]
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        """Set the number of visits to this node."""
        if self.lock:
            with self.lock:
                self.parent.child_number_visits[self.action] = value
        else:
            self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        """Get the total value of this node."""
        if self.lock:
            with self.lock:
                return self.parent.child_total_value[self.action]
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        """Set the total value of this node."""
        if self.lock:
            with self.lock:
                self.parent.child_total_value[self.action] = value
        else:
            self.parent.child_total_value[self.action] = value
            
    @property
    def Q(self):
        """Get the Q value of this node"""
        return self.parent.child_Q()[self.action]
    
    @property
    def U(self):
        """Get the U value of this node"""
        return self.parent.child_U()[self.action]

    def child_Q(self):
        """
        Returns the Q-values for each child.
        For children with zero visits, set Q to 0.
        """
        # Prepare an array of infinities
        q = np.full_like(self.child_number_visits, 0, dtype=float)
        
        visited_mask = (self.child_number_visits > 0)
        q[visited_mask] = (self.child_total_value[visited_mask] 
                        / self.child_number_visits[visited_mask])
        
        return q

    # UCB1 style of UCB
    def child_U(self):
        """
        Returns the upper confidence bound (U) array for each child.
        For children with zero visits, set U to +∞ and skip the division.
        """
        # Prepare an array of infinities
        u = np.full_like(self.child_number_visits, np.inf, dtype=float)
        
        # Only compute for children that have > 0 visits
        visited_mask = (self.child_number_visits > 0)
        u[visited_mask] = (self.explore_factor 
                        * np.sqrt(np.log(self.number_visits) 
                                    / self.child_number_visits[visited_mask]))
        
        return u

    # Alpha go zero style of UCB
    # def child_U(self):
    #     """Get child U values (upper confidence bounds)"""
    #     return self.explore_factor * math.sqrt(self.number_visits) * (
    #         self.child_priors / (1 + self.child_number_visits))

    def best_child(self):
        """
        Returns the index of the best child.
        If any children have U = +∞ (e.g., unvisited), pick randomly among them.
        Otherwise, pick the child with the max Q + U.
        """
        # Get the array of U-values for each child
        u = self.child_U()  # [child_U1, child_U2, ...]
        
        # Check if any of them are infinity
        inf_mask = np.isinf(u)
        if np.any(inf_mask):
            # Pick a random index among the infinite-U children
            inf_indices = np.where(inf_mask)[0]
            return np.random.choice(inf_indices)
        else:
            # No infinite values; select based on Q + U
            q = self.child_Q()
            return np.argmax(q + u)


    def select_leaf(self, return_path=False):
        """From current node, select highest upper confidence bound node until at next leaf node"""
        current = self
        path = []
        # While we aren't at a leaf node
        while current.is_expanded and not current.done:
            # Since the thread is passing through this node remove one from the total value to encourage other threads to explore other nodes
            # current.total_value -= 1 # This has no change with one thread because it is replaced in backup
            
            # Pick the best child and move to that node
            best_action = current.best_child()
            current = current.maybe_add_child(best_action) # Child is only added if we reach the unsimulated leaf node
            path.append(best_action) # Add the action to the path

        # Return the leaf node we ended on
        if return_path:
            return current, path
        return current

    def maybe_add_child(self, action, insert_leaf=None):
        """Add a child if it does not exist"""
        # Check if the action has already been simulated
        if action not in self.children:
            # If we have a leaf we want to insert since it was already simulated
            if insert_leaf is not None:
                # If the leaf node has already been created, add it to the children
                self.children[action] = insert_leaf
                self.children[action].parent = self
                # NOTE: Total value is shared and has already been added to the parent so it is not updated here
             
            # If not, Run the simulation and update the child
            else:
                new_state, reward, done = self.env.step(self.state, self.env.action_space[action])
                self.children[action] = MCTSNode(self.env, new_state, action, explore_factor=self.explore_factor,
                                                 discount_factor=self.discount_factor, reward=reward, parent=self, 
                                                 done=done, parallel=self.parallel)

        return self.children[action]
            
    def one_action_rollout(self, action, all_same=False, zero_after_first=False, accelerate_after_first=False, 
                           accelerate=False, hertg=False, keep_nodes=False, keep_data=False):
        """ Simulate the same action until reaching a terminal state 
            params:
                action: the action to simulate (int)
                all_same: whether to simulate the same action for all steps (bool)
                zero_after_first: whether to simulate the zero action after the first action (bool)
                accelerate_after_first: accelerate in velocity direction after the first action (bool)
                hertg: whether to use the HERTG heuristic at the end of the rollout (bool)
                keep_nodes: whether to keep the nodes of the rollout (bool)
                keep_data: whether to keep the data of the rollout (bool)
        """
        if all_same and zero_after_first and accelerate_after_first and accelerate:
            raise ValueError('Cannot have all of 4 first options be True')
        elif not all_same and not zero_after_first and not accelerate_after_first and not accelerate:
            raise ValueError('Must have one of 4 first options be True')
        
        # Get the cumulative reward of the same action
        done = False
        state = self.state
        cumulative_reward = 0
        leaf = self
        states, rewards, dones = [], [], []
        first_action = True
        while not done:
            if not first_action and zero_after_first:
                action = 0
            elif (not first_action and accelerate_after_first) or accelerate:
                if state[0][2] < 0:
                    action = 15
                else:
                    action = 10
            else:
                action = action
                
            if not keep_nodes:
                state, reward, done = self.env.step(state, self.env.action_space[action])
                states.append(state)
                rewards.append(reward)
                dones.append(done)
            else:
                leaf = leaf.maybe_add_child(action)
                state, reward, done = leaf.state, leaf.reward, leaf.done
                
            cumulative_reward += reward
            first_action = False
        if hertg:
            # Tack on expected cost to go to final state
            pass
        
        if keep_data:
            return cumulative_reward, states, rewards, dones
        
        return cumulative_reward
    
    def rollout_children(self, all_same=False, zero_after_first=False, accelerate_after_first=False,
                         hertg=False, keep_nodes=True):
        """ rollout children to get the expected reward
            params:
                all_same: whether to simulate the same action for all steps (bool)
                zero_after_first: whether to simulate the zero action after the first action (bool)
                accelerate_after_first: accelerate in velocity direction after the first action (bool)
                hertg: whether to use the HERTG heuristic at the end of the rollout (bool)
                keep_nodes: whether to keep the nodes of the rollout (bool)
                keep_data: whether to keep the data of the rollout (bool)
        """
        if all_same and zero_after_first and accelerate_after_first:
            raise ValueError('Cannot have same_action and zero_action be True')
        elif not all_same and not zero_after_first and not accelerate_after_first:
            raise ValueError('Must have all_same or zero_after_first or accelerate_after_first be True')
        
        rollout_rewards = np.zeros([self.env.N], dtype=np.float32)
        for action in range(self.env.N):
            rollout_rewards[action] = self.one_action_rollout(action, all_same=all_same, zero_after_first=zero_after_first,
                                                              accelerate_after_first=True, hertg=hertg, keep_nodes=keep_nodes)
        return rollout_rewards
    
    def parallel_rollout_children(self, all_same=False, zero_after_first=False, accelerate_after_first=False,
                                  hertg=False, keep_nodes=True):
        """ rollout children in parallel to get the expected reward 
            params:
                all_same: whether to simulate the same action for all steps (bool)
                zero_after_first: whether to simulate the zero action after the first action (bool)
                accelerate_after_first: accelerate in velocity direction after the first action (bool)
                hertg: whether to use the HERTG heuristic at the end of the rollout (bool)
                keep_nodes: whether to keep the nodes of the rollout (bool)
                keep_data: whether to keep the data of the rollout (bool)
        """
        if all_same and zero_after_first and accelerate_after_first:
            raise ValueError('Cannot have same_action and zero_action be True')
        elif not all_same and not zero_after_first and not accelerate_after_first:
            raise ValueError('Must have all_same or zero_after_first or accelerate_after_first be True')
        
        pool = mp.Pool(processes=self.env.N)
        
        if keep_nodes:
            partial_one_action_rollout = partial(self.one_action_rollout, all_same=all_same, zero_after_first=zero_after_first,
                                                 accelerate_after_first=accelerate_after_first, hertg=hertg, keep_data=True)
            rollout_results = pool.map(partial_one_action_rollout, np.arange(self.env.N))
            
            rollout_rewards, states, rewards, dones  = zip(*rollout_results)
            for action, (state, reward, done) in enumerate(zip(states, rewards, dones)):
                leaf = self
                for i, (s, r, d) in enumerate(zip(state, reward, done)):
                    leaf.children[action] = MCTSNode(self.env, s, action, explore_factor=self.explore_factor,
                                                    discount_factor=self.discount_factor, reward=r, parent=leaf, 
                                                    done=d, parallel=self.parallel)
                    leaf = leaf.children[action]
                    if i == 0:
                        leaf.is_expanded = True
                        leaf.number_visits += 1
        else:
            partial_one_action_rollout = partial(self.one_action_rollout, hertg=hertg, keep_data=False)
            rollout_rewards= pool.map(partial_one_action_rollout, np.arange(self.env.N))
        
        pool.close()
        pool.join()
        return rollout_rewards

    def expand(self, child_priors):
        """Expand the current node with the given child_priors"""
        self.is_expanded = True
        # self.child_priors = child_priors

    def backup(self, value_estimate: float):
        """Backpropogate the value estimate up the tree to the root node"""
        current = self

        # Cumulate the rewards in each node and tack on the value estimate
        # This is done because we want to incorporate the rewards we have simulated so far plus the expected reward to go
        backup_cumulative_rewards = value_estimate
        
        # While we aren't at the root node which has no parent
        while current.parent is not None:
            # Add a visit to since we are adding a new value estimate to this node (N is the number of estimates for average)
            current.number_visits += 1
            
            # Add this nodes reward to the backup cumulative rewards
            backup_cumulative_rewards += self.discount_factor * current.reward
            
            # Add the value estimate to the total value of the node (Reward + expected reward to go)
            current.total_value += backup_cumulative_rewards # Add the 1 value back we subtracted in select_leaf
            # current.total_value += backup_cumulative_rewards + 1 # Add the 1 value back we subtracted in select_leaf
            current = current.parent # Move to the parent node
            
def get_best_trajectory(root: MCTSNode, highest_Q=False, return_rewards=False):
    """
    Get the best action trajectory from the root node
    :param root: the root node of the MCTS tree
    :param highest_Q: whether to take the highest Q value action or the highest UCB action
    :returns action_trajectory: the best action trajectory from the root node
    :returns state_trajectory: the state trajectory of the best action trajectory
    """
    current = root
    action_trajectory = []
    state_trajectory = []
    reward_trajectory = []
    
    # Iterate through the best actions until the best action node does not exist
    while True:
        state_trajectory.append(current.state)
        reward_trajectory.append(current.reward)
        if highest_Q:
            best_action = np.argmax(current.child_Q())
        else:
            best_action = current.best_child() # this accounts for upper confidence bound
        action_trajectory.append(current.env.action_space[best_action])
        
        # Break if the best child node does not exist (hasn't been expanded yet)
        if best_action not in current.children:
            break
        
        # Continue traversal
        current = current.children[best_action]
    
    if return_rewards:
        return action_trajectory, state_trajectory, reward_trajectory
    
    return action_trajectory, state_trajectory

class DummyNode(object):
    """
    Dummy node class that simplifies implimentation when it is the root of the MCTS tree
    """
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def mcts_search(env: Environment, eval, starting_state: np.ndarray, learning_iterations: int=1000, explore_factor: float=1., discount_factor: float=0.9):
    """
    Run many iterations of MCTS to build up a tree and get the best action to take
    params:
    env: the environment to run MCTS on, class that inherits from Environment class
    starting_state: the starting state of the search (Any)
    learning_iterations: the number of iterations to run MCTS (int)

    returns:
    the index of the best action to take (int)
    the root node of the MCTS tree (MCTSNode)
    """
    root = MCTSNode(env, starting_state, action=None, parent=DummyNode(), explore_factor=explore_factor, discount_factor=discount_factor)
    for _ in range(learning_iterations):
        leaf = root.select_leaf() # Select with UCB up to the leaf node and do one environment step
        
        # Add the transition to the replay buffer for training (except for the root node)
        # child_priors, value_estimate = eval.inference(leaf.state) # Inference the model to get the probability of each action and the value estimate
        child_priors, value_estimate = np.ones([env.N]) / env.N, 0. # Even probability for each action for testing and no expected reward to go
        
        leaf.expand(child_priors) # Expand the leaf node with the child priors
        leaf.backup(value_estimate) # Backup the value estimate up the tree to the root node
        
        # eval.optimize_model() # Optimize the model using replay memory which we just added one transition to

    # Return the action with the most visits and the root node
    return env.action_space[np.argmax(root.child_number_visits)], root

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

def init_worker(global_env, global_starting_state):
    global env
    global starting_state
    env = global_env
    starting_state = global_starting_state

def parallel_mcts(action_space, env, starting_state):
    """
    Parallelizes the MCTS using multiprocessing.Pool.
    """
    pool = mp.Pool(processes=len(action_space), initializer=init_worker, initargs=(env, starting_state,))
    
    start_time = timeit.default_timer()
    
    # Distribute the actions to processes
    results = pool.map(one_action_mcts, np.arange(len(action_space)), chunksize=1)
    
    pool.close()
    pool.join()
    
    print(timeit.default_timer() - start_time)
    
    # Find the action with the best value
    best_action, best_value = max(results, key=lambda x: x[1])
    
    return best_action, best_value

def create_shared_array(shape, dtype=ctypes.c_float):
    """
    Create a shared array with the given shape and dtype for parallel MCTS
    """
    shared_array_base = mpArray(dtype, int(np.prod(shape)))
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    return shared_array.reshape(shape), shared_array_base

def mcts_worker(env: Environment, root_mcts_node: MCTSNode, output_queue: mp.Queue):
    """
    Worker function for running MCTS in parallel.
    
    params:
    env: the environment to run MCTS on, class that inherits from Environment class
    root_mcts_node: the root node of the MCTS tree, which we are starting the search from
    output_queue: the queue to put the tree additions into
    """
    
    # Run stages of MCTS until we create a new leaf node
    leaf, path = root_mcts_node.select_leaf(return_path=True)
    child_priors, value_estimate = env.evaluate(leaf.state)
    
    # Place the node parameters into the queue
    leaf_parameters = (path, leaf.state, leaf.action, child_priors, value_estimate)
    output_queue.put(leaf_parameters)

def parallel_mcts_search(env: Environment, starting_state: np.ndarray, learning_iterations: int, num_processes: int):
    """
    Run many iterations of MCTS to build up a tree and get the best action to take in parallel
    params:
    env: the environment to run MCTS on, class that inherits from Environment class
    starting_state: the starting state of the search (Any)
    learning_iterations: the number of iterations to run MCTS (int)
    
    returns:
    the index of the best action to take (int)
    the root node of the MCTS tree (MCTSNode)
    """
    root = MCTSNode(env, starting_state, action=None, parent=DummyNode(), parallel=True)
    output_queue = mp.Queue()
    
    for _ in range(learning_iterations // num_processes):
        processes = []
        for _ in range(num_processes):
            p = mp.Process(target=mcts_worker, args=(env, root, output_queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
            
            # Get the parameters to create a new leaf node
            path, state, action, child_priors, value_estimate = output_queue.get()

            # Recreate the leaf node
            leaf = MCTSNode(env, state, action, parallel=True)
            
            # Add the leaf node using the path of actions (If path is empty, leaf is the root node, no need to add)
            current = root
            for action in path:
                current = current.maybe_add_child(action, insert_leaf=leaf)
                
            # Expand node with priors and backup value estimate
            current.expand(child_priors)
            current.backup(value_estimate)

    return np.argmax(root.child_number_visits), root

def test_picklable(obj):
    try:
        pickle.dumps(obj)
        print('Picklable')
    except pickle.PicklingError as e:
        print(f'PicklingError: {e}')
    except TypeError as e:
        print(f'TypeError: {e}')