import collections
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Tuple, Any
import multiprocessing as mp
import ctypes

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

  @abstractmethod
  def evaluate(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Evaluate the current state of the environment

    params:
      state: the current state of the environment (np.ndarray(shape=(1, N)))

    returns:
      child_priors: the prior probabilities of each action, probabilities of best action (np.ndarray(shape=(1, N)))
      value_estimate: the value estimate of the current state (float)
    """
    pass

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
        parent: the parent node of this node (MCTSNode)

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
    def __init__(self, env: Environment, state: np.ndarray, action: np.ndarray, parent: 'MCTSNode'=None, shared_num_threads=None, lock=None):
        self.env = env
        self.state = state
        self.action = action
        self.is_expanded = False
        self.parent = parent  # Optional[MCTSNode]
        self.children = {}  # Dict[action, MCTSNode]
        self.child_priors = np.zeros([self.env.N], dtype=np.float32)
        self.child_total_value = np.zeros([self.env.N], dtype=np.float32)
        self.child_number_visits = np.zeros([self.env.N], dtype=np.float32)

        # Initialize shared_num_threads and lock if they are not provided
        if shared_num_threads is None:
            self.shared_num_threads = mp.Array(ctypes.c_int, [0] * self.env.N)
        else:
            self.shared_num_threads = shared_num_threads

        if lock is None:
        self.lock = mp.Lock()
        else:
        self.lock = lock

    @property
    def number_visits(self):
        """Get number of visits to this node"""
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        """Set the number of visits to this node"""
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        """Get the total value of this node"""
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        """Set the total value of this node"""
        self.parent.child_total_value[self.action] = value

    def child_Q(self):
        """Get child Q values based on the stored total value and number of visits"""
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        """Get child U values (upper confidence bounds)"""
        exploration_factor = math.sqrt(self.number_visits) / (1 + self.shared_num_threads[self.action])
        return exploration_factor * (self.child_priors / (1 + self.child_number_visits))

    def best_child(self):
        """Get the best child based on the upper confidence bound"""
        return np.argmax(self.child_Q() + self.child_U())

    def select_leaf(self):
        """From current node, select highest upper confidence bound node until at next leaf node"""
        current = self
        # While we aren't at a leaf node
        while current.is_expanded:
            # Grab the lock to increment the thread shared variable num_threads when selecting a node
            with self.lock:
                # Pick the best child and move to that node
                best_move = current.best_child()
                current.shared_num_threads[best_move] += 1  # Increment the num_threads when selecting a node
                current = current.maybe_add_child(best_move) # Child is only added if we reach the unsimulated leaf node
                
        # Return the leaf node we ended on
        return current

    def maybe_add_child(self, action):
        """Add a child if it does not exist"""
        # Check if the action has already been simulated
        if action not in self.children:
            # If not, Run the simulation and update the child
            new_state, reward, done = self.env.step(self.state, action)
            self.children[action] = MCTSNode(self.env, new_state, action, parent=self)
            self.child_total_value[action] = reward # Replace the value estimate with the environment reward

        return self.children[action]

    def expand(self, child_priors):
        """Expand the current node with the given child_priors"""
        self.is_expanded = True
        self.child_priors = child_priors

    def backup(self, value_estimate: float):
        """Backpropogate the value estimate up the tree to the root node"""
        current = self

        # While we aren't at the root node which has no parent
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value_estimate
            current = current.parent

class DummyNode(object):
    """
    Dummy node class that simplifies implimentation when it is the root of the MCTS tree
    """
    def __init__(self):
    self.parent = None
    self.child_total_value = collections.defaultdict(float)
    self.child_number_visits = collections.defaultdict(float)


def mcts_search(env: Environment, starting_state: np.ndarray, learning_iterations: int):
    """
    Run many iterations of MCTS to build up a tree and get the best action to take
    params:
    env: the environment to run MCTS on, class that inherits from Environment class
    starting_state: the starting state of the search (np.ndarray(shape=(1, N)))
    learning_iterations: the number of iterations to run MCTS (int)

    returns:
    the index of the best action to take (int)
    the root node of the MCTS tree (MCTSNode)
    """
    root = MCTSNode(env, starting_state, action=None, parent=DummyNode())
    for _ in range(learning_iterations):
        leaf = root.select_leaf()
        child_priors, value_estimate = env.evaluate(leaf.state)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
        
    return np.argmax(root.child_number_visits), root

def mcts_worker(env: Environment, state: np.ndarray, output_queue: mp.Queue):
    """Worker function for running MCTS in parallel."""
    leaf = state.select_leaf()
    child_priors, value_estimate = env.evaluate(leaf.state)
    leaf.expand(child_priors)
    leaf.backup(value_estimate)
    output_queue.put((leaf.action, leaf.child_number_visits[leaf.action], leaf.child_total_value[leaf.action]))

def parallel_mcts_search(env: Environment, starting_state: np.ndarray, learning_iterations: int, num_processes: int):
    root = MCTSNode(env, starting_state, action=None, parent=DummyNode())
    output_queue = mp.Queue()

    for _ in range(learning_iterations // num_processes):
        processes = []
        for _ in range(num_processes):
            p = mp.Process(target=mcts_worker, args=(env, root, output_queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
            action, visits, total_value = output_queue.get()
            root.child_number_visits[action] += visits
            root.child_total_value[action] += total_value

    return np.argmax(root.child_number_visits), root
