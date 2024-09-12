'''
modified from
https://github.com/martinobdl/MCTS
'''

import cloudpickle
import numpy as np
# import gym
from nav2_air_active_track_planner.mcts.nodes import DecisionNode, RandomNode
from typing import Callable, List, Tuple, Any, Optional


class MCTS:
    """
    Base class for MCTS based on Monte Carlo Tree Search for Continuous and Stochastic Sequential
    Decision Making Problems, Courtoux

    :param initial_obs: (int or tuple) initial state of the tree. Returned by env.reset().
    :param env: (gym env) game environment
    :param K: (float) exporation parameter of UCB
    """

    def __init__(
        self, 
        initial_obs, 
        env, 
        K: float,
        _hash_action: Callable[[Any], Tuple],
        _hash_state: Callable[[Any], Tuple],
    ):
        
        self.env = env
        self.K = K
        self.root = DecisionNode(state=initial_obs, is_root=True)
        self._initialize_hash(_hash_action, _hash_state)

    def _initialize_hash(
        self, 
        _hash_action: Callable[[Any], Tuple],
        _hash_state: Callable[[Any], Tuple],
    ):
        """
        Set the hash preprocessors of the state and the action, 
        in order to make them hashable.

        Need to be customized based on the definition of state and action
        """

        self._hash_action = _hash_action
        self._hash_state = _hash_state

    def _collect_data(self, action_vector: Any = None):
        """
        Collects the data and parameters to save.
        """
        data = {
            "K": self.K,
            "nodes": [],
            "decision": action_vector,
        }

        self._traverse_tree(self.root, data)
        # do a dfs for all nodes

        return data
    
    def _traverse_tree(self, node: DecisionNode, storage_dict: dict):

        if node.is_final:   return

        position, _, _ = node.state

        for _, random_node in node.children.items():
            for _, next_decision_node in random_node.children.items():
                next_position, _, time_step = next_decision_node.state
                node_dict = {"position": next_position,
                        "parent_position": position,
                        "time_step": time_step,
                        "visits": next_decision_node.visits,
                        }
                storage_dict["nodes"].append(node_dict)
                self._traverse_tree(next_decision_node, storage_dict=storage_dict)


    def update_decision_node(
        self, 
        decision_node: DecisionNode, 
        random_node: RandomNode, 
        hash_preprocess: Callable,
    ):
        """
        Return the decision node of drawn by the select outcome function.
        If it's a new node, it gets appended to the random node.
        Else returns the decsion node already stored in the random node.

        :param decision_node (DecisionNode): the decision node to update
        :param random_node (RandomNode): the random node of which the decison node is a child
        :param hash_preprocess (fun: gym.state -> hashable) function that sepcifies the preprocessing in order to make
        the state hashable.
        """

        if hash_preprocess(decision_node.state) not in random_node.children.keys():
            decision_node.parent = random_node
            random_node.add_children(decision_node, hash_preprocess)
        else:
            decision_node = random_node.children[hash_preprocess(decision_node.state)]

        return decision_node

    def grow_tree(self):
        """
        Explores the current tree with the UCB principle until we reach an unvisited node
        where the reward is obtained with random rollouts.
        """

        decision_node = self.root
        internal_env = self.env

        while (not decision_node.is_final): # and decision_node.visits > 1:

            a = self.select(decision_node)

            new_random_node = decision_node.next_random_node(a, self._hash_action)

            (new_decision_node, r) = self.select_outcome(internal_env, new_random_node)

            new_decision_node = self.update_decision_node(new_decision_node, new_random_node, self._hash_state)

            new_decision_node.reward = r
            new_random_node.reward = r

            decision_node = new_decision_node

        decision_node.visits += 1
        # TODO: update this evaluation function based on env
        # cumulative_reward = self.evaluate(internal_env)
        cumulative_reward = self.env.evaluate(decision_node.state)
        # back propagate
        while not decision_node.is_root:
            random_node = decision_node.parent
            cumulative_reward += random_node.reward
            random_node.cumulative_reward += cumulative_reward
            random_node.visits += 1
            decision_node = random_node.parent
            decision_node.visits += 1

    def evaluate(self, env, state) -> float:
        """
        a customized function, don't have to be

        Evaluates a DecionNode playing until an terminal node using the rollotPolicy,
        

        :param env: (gym.env) gym environemt that describes the state at the node to evaulate.
        :return: (float) the cumulative reward observed during the tree traversing.
        """
        max_iter = 100
        R = 0
        done = False
        iter = 0
        while ((not done) and (iter < max_iter)):
            iter += 1
            a = env.action_space.sample(state)
            s, r, done = env.step(a)
            R += r

        return R

    def select_outcome(
        self, 
        env, 
        random_node: RandomNode,
    ) -> DecisionNode:
        """
        Given a RandomNode returns a DecisionNode

        :param: env: (gym env) the env that describes the state in which to select the outcome
        :param: random_node: (RandomNode) the random node from which selects the next state.
        :return: (DecisionNode) the selected Decision Node
        """
        new_state, r, done = env.step(random_node.parent.state, random_node.action)
        return DecisionNode(state=new_state, parent=random_node, is_final=done), r

    def select(
        self, 
        x: DecisionNode,
    ) -> Any:
        """
        Selects the action to play from the current decision node

        :param x: (DecisionNode) current decision node
        :return: action to play
        """
        if x.visits <= 2:
            x.children = {a: RandomNode(a, parent=x) for a in range(self.env.action_space.n)}

        def scoring(k):
            if x.children[k].visits > 0:
                return x.children[k].cumulative_reward/x.children[k].visits + \
                    self.K*np.sqrt(np.log(x.visits)/x.children[k].visits)
            else:
                return np.inf

        a = max(x.children, key=scoring)

        return a

    def best_action(self) -> Any:
        """
        At the end of the simulations returns the most visited action

        :return: (float) the best action according to the number of visits
        """

        action_vector = list()

        decision_node = self.root
        # depth = 0
        while not decision_node.is_final:
            # depth += 1
            number_of_visits_children = [node.visits for node in decision_node.children.values()]
            # avg_reward_children = [node.cumulative_reward/node.visits for node in decision_node.children.values()]
            # print(f'layer {depth}: {number_of_visits_children}')
            indices_most_visit = np.argwhere(number_of_visits_children == np.amax(number_of_visits_children)).flatten().tolist()
            # this may contain more than 1 children
            if len(indices_most_visit) == 1:
                index_best_action = indices_most_visit[0]
            else:
                avg_reward_list = []
                for index in indices_most_visit:
                    node = list(decision_node.children.values())[index]
                    element = (index, node.cumulative_reward/node.visits)
                    avg_reward_list.append(element)
                index_best_action = max(avg_reward_list, key = lambda x: x[1])[0]

            # index_best_action = np.argmax(number_of_visits_children)
            random_node = list(decision_node.children.values())[index_best_action]
            a = random_node.action
            action_vector.append(a)
            # find next decision state, only for determinisitic case
            # TODO need to consider the stochastic case
            assert len(random_node.children) == 1, print(random_node.children)
            decision_node = list(random_node.children.values())[0]

        # print("action output is {}".format(a))
        return action_vector

    def learn(
        self, 
        Nsim: int, 
        progress_bar=False,
    ):
        """
        Expand the tree and return the best action

        :param: Nsim: (int) number of tree traversals to do
        :param: progress_bar: (bool) wether to show a progress bar (tqdm)
        """

        if progress_bar:
            # iterations = tqdm(range(Nsim))
            iterations = range(Nsim)
        else:
            iterations = range(Nsim)
        for _ in iterations:
            
            self.grow_tree()

    # TODO: visualize the MCTS process
    def save(self, path=None, action_vector: Any = None):
        """
        Saves the tree structure as a pkl.

        :param path: (str) path in which to save the tree
        """
        data = self._collect_data(action_vector=action_vector)

        # name = np.random.choice(['a', 'b', 'c', 'd', 'e', 'f']+list(map(str, range(0, 10))), size=8)
        # if path is None:
        #     path = './logs/'+"".join(name)+'_'
        # if os.path.exists(path):
        with open(path, "wb") as f:
            cloudpickle.dump(data, f)
        # print("Saved at {}".format(path))


if __name__ == "__main__":
    pass
