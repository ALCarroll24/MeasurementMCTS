'''
modified from
https://github.com/martinobdl/MCTS
'''

import cloudpickle
import numpy as np
# import gym
from .nodes import DecisionNode, RandomNode
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
        expansion_branch_factor: int = 2,
        deterministic: bool = True,
    ):
        
        self.env = env
        self.K = K
        self.root = DecisionNode(state=initial_obs, is_root=True)
        self._initialize_hash(_hash_action, _hash_state)
        self.expansion_branch_factor = expansion_branch_factor
        self.deterministic = deterministic

    def get_node(self, node_hash: int) -> Optional[DecisionNode]:
        """
        Returns the node from the tree given its hash

        :param node_hash: (int) hash of the node to retrieve
        :return: (DecisionNode) the node corresponding to the hash
        """
        def _get_node(node: DecisionNode, node_hash: int) -> Optional[DecisionNode]:
            if node.__hash__() == node_hash:
                return node
            for _, random_node in node.children.items():
                for _, next_decision_node in random_node.children.items():
                    found_node = _get_node(next_decision_node, node_hash)
                    if found_node is not None:
                        return found_node
            return None

        return _get_node(self.root, node_hash)

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

        ## SELECTION PHASE
        # While goal has not been reached and we are not at a leaf node (has been visited)``
        while (not decision_node.is_final) and decision_node.visits > 0:
            # print("Starting Decision node: ", decision_node)
            
            # Get action from this decision node using UCB
            a = self.select(decision_node)
            # print(f"Action selected: {a}")

            # Get the existing random node from the action and decision node
            new_random_node = decision_node.next_random_node(a, self._hash_action)
            # print(f"New Random node: {new_random_node}")

            # Create new decision node using environment step function, 
            # if stochastic or if this node has not been visited (not simulated) use environment to get the next state and reward
            if not self.deterministic or new_random_node.visits == 0:
                (new_decision_node, r) = self.select_outcome(internal_env, new_random_node)
            
            # If deterministic, we have already simulated this node, so just get the child decision node
            else:
                new_decision_node = list(new_random_node.children.values())[0]
                r = new_decision_node.reward
            
            # print(f"New Decision node: {new_decision_node}")

            # Ensure that the decision node is connected to its parent random node (not sure why it wouldn't be though...?)
            new_decision_node = self.update_decision_node(new_decision_node, new_random_node, self._hash_state)
            # print(f"Updated Decision node: {new_decision_node}")

            # Set the reward of the new nodes
            new_decision_node.reward = r
            new_random_node.reward = r

            # Continue the tree traversal
            decision_node = new_decision_node


        ### EXPANSION PHASE
        # If we have not yet visited (expanded) this decision node, expand it (add children nodes)
        # No need to do this if we are done (at the horizon)
        if decision_node.visits == 0 and not decision_node.is_final:
            # If we are expanding with all actions
            if self.expansion_branch_factor == -1:
                for a in self.env.all_actions:
                    # Action -> random node -> attach to decision node
                    new_random_node = RandomNode(a, parent=decision_node)
                    decision_node.add_children(new_random_node, self._hash_action)
            
            # Otherwise if we are sampling a number of times from the action space
            else:
                for i in range(self.expansion_branch_factor):
                    # Random action -> random node -> attach to decision node
                    new_random_node = RandomNode(a, parent=decision_node)
                    decision_node.add_children(new_random_node, self._hash_action)

        # Add a visit since we ended traversal on this decision node
        decision_node.visits += 1
        
        
        ### SIMULATION PHASE
        # Custom evaluation function in the environment class
        cumulative_reward = self.env.evaluate(decision_node.state)
        decision_node.evaluation_reward = cumulative_reward
        
        # print("Evaluation reward: ", cumulative_reward)
        
        # Add the reward step reward to the cumulative reward
        cumulative_reward += decision_node.reward
        
        ### BACKPROPAGATION PHASE
        # Back propagate the reward back to the root
        while not decision_node.is_root:
            random_node = decision_node.parent
            # cumulative_reward += random_node.reward
            random_node.cumulative_reward += cumulative_reward
            random_node.visits += 1
            decision_node = random_node.parent
            decision_node.visits += 1

    def expand(self, decision_node: DecisionNode, action: np.ndarray):
        # Random action -> random node -> environment step -> decision node
        new_random_node = decision_node.next_random_node(action, self._hash_action)
        (new_decision_node, r) = self.select_outcome(self.env, new_random_node)
        new_decision_node = self.update_decision_node(new_decision_node, new_random_node, self._hash_state)
        new_decision_node.reward = r
        new_random_node.reward = r

    # Random action evaluation, doesn't really make sense for this problem
    # def evaluate(self, env, state) -> float:
    #     """
    #     a customized function, don't have to be

    #     Evaluates a DecionNode playing until an terminal node using the rollotPolicy,
        

    #     :param env: (gym.env) gym environemt that describes the state at the node to evaulate.
    #     :return: (float) the cumulative reward observed during the tree traversing.
    #     """
    #     R = 0
    #     done = False
    #     iter = 0
    #     while ((not done) and (iter < self.random_iterations)):
    #         iter += 1
    #         a = env.action_space_sample()
    #         s, r, done = env.step(state, a)
    #         R += r

    #     return R

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
        def scoring(k):
            if x.children[k].visits > 0:
                return x.children[k].cumulative_reward/x.children[k].visits + \
                    self.K*np.sqrt(np.log(x.visits)/x.children[k].visits)
            else:
                return np.inf

        a = max(x.children, key=scoring)

        return a

    def best_action(self):
        """
        At the end of the simulations returns the highest mean reward action
        
        : return: (tuple) the best action according to the mean reward
        """
        
        # Create a list of the mean rewards of the children
        children_key = list(self.root.children.keys())
        children_values = list(self.root.children.values())
        
        # Initialize mean reward list
        children_mean_rew = [0.0] * len(children_key)
        
        # Calculate the mean reward for each child
        for i in range(len(children_key)):
            children_mean_rew[i] = children_values[i].cumulative_reward / children_values[i].visits
            
        # Get the index of the highest mean reward
        index_best_action = np.argmax(children_mean_rew)
        
        # Return the action corresponding to the highest mean reward
        a = children_key[index_best_action]
        
        return a

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
            
            # print("Next Learning Iteration")
            # print("Node hash: ", self.root.__hash__())
            # print("Root node: ", self.root)
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
