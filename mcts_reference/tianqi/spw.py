from typing import List, Any, Callable, Tuple
import numpy as np
from nav2_air_active_track_planner.mcts.mcts import MCTS
from nav2_air_active_track_planner.mcts.nodes import DecisionNode

class SPW(MCTS):
    """
    Simple Progressive Widening trees based on Monte Carlo Tree Search for Continuous and
    Stochastic Sequential Decision Making Problems, Courtoux

    :param alpha: (float) the number of children of a decision node are always greater that v**alpha,
        where v is the number of visits to the current decision node
    :param initial_obs: (int or tuple) initial state of the tree. Returned by env.reset().
    :param env: (gym env) game environment
    :param K: exploration parameter of UCB
    """

    def __init__(
        self, 
        initial_obs: Any, 
        env, 
        K: float,
        _hash_action: Callable[[Any], Tuple],
        _hash_state: Callable[[Any], Tuple],
        alpha: float, 
    ):
        # super(MCTS, self).__init__(initial_obs, env, K,
        #     _hash_action, _hash_state)
        MCTS.__init__(self, initial_obs, env, K,
            _hash_action, _hash_state)

        self.alpha = alpha

    def select(self, x: DecisionNode) -> Any:
        """
        Selects the action to play from the current decision node. The number of children of a DecisionNode is
        kept finite at all times and monotonic to the number of visits of the DecisionNode.

        :param x: (DecisionNode) current decision node
        :return: (Any) action to play
        """
        # If we have already visited the node enough times, we select a random action
        if x.visits**self.alpha >= len(x.children):
            a = self.env.action_space_sample(x.state)

        # Otherwise, pick the best action according to the UCB
        else:

            def scoring(k):
                if x.children[k].visits > 0:
                    return x.children[k].cumulative_reward/x.children[k].visits + \
                        self.K*np.sqrt(np.log(x.visits)/x.children[k].visits)
                else:
                    return np.inf

            a = max(x.children, key=scoring)

        return a
    
    def _collect_data(self, action_vector: Any = None):
        """
        Collects the data and parameters to save.
        """
        data = {
            "K": self.K,
            "nodes": [],
            "decision": action_vector,
            "alpha": self.alpha,
        }

        self._traverse_tree(self.root, data)
        # do a dfs for all nodes

        return data