from typing import List, Dict, Any, Tuple, Optional
from threading import Lock

class DecisionNode:
    """
    The Decision Node class

    :param state: (tuple) defining the state
    :param parent: (RandomNode) The parent of the Decision Node, None if root node
    :param is_root: (bool)
    :param is_final: (bool)
    """

    def __init__(
        self, 
        state: Any = None, 
        parent = None, 
        is_root: bool = False, 
        is_final: bool = False,
        avg_eval_reward: float = 0.0
        
    ):
        self.state = state
        self.parent: RandomNode = parent
        self.children: Dict[Any, RandomNode] = {}
        self.is_root = is_root
        self.is_final = is_final
        self.avg_eval_reward = avg_eval_reward
        self.visits: int = 0
        self.reward: float = 0.0
        self.lock = Lock()

    def add_children(
        self, 
        random_node, 
        hash_preprocess=None,
    ):
        """
        Adds a RandomNode object to the dictionary of children (key is the action)

        :param random_node: (RandomNode) add a random node to the set of children visited
        """
        if hash_preprocess is None:
            def hash_preprocess(x):
                return x

        with self.lock:
            self.children[hash_preprocess(random_node.action)] = random_node

    def next_random_node(
        self, 
        action: Any, 
        hash_preprocess=None
    ) -> 'RandomNode':
        """
        Add the random node to the children of the decision node if note present. Otherwise it resturns the existing one

        :param action: (float) the action taken at the current node
        :return: (RandomNode) the resulting random node
        """

        if hash_preprocess is None:
            def hash_preprocess(x):
                return x

        if hash_preprocess(action) not in self.children.keys():
            new_random_node = RandomNode(action, parent=self)
            self.add_children(new_random_node, hash_preprocess)
        else:
            new_random_node = self.children[hash_preprocess(action)]

        return new_random_node
    
    def get_depth(self) -> int:
        """
        Returns the depth of the node in the tree

        :return: (int) the depth of the node
        """
        return self.state[3]

    def __repr__(self):
        s = ""
        for k, v in self.__dict__.items():
            if k == "children":
                pass
            elif k == "parent":
                pass
            else:
                s += str(k)+": "+str(v)+"\n"
        return s


class RandomNode:
    """
    The RandomNode class defined by the state and the action, it's a random node since the next state is not yet defined

    :param action: (action) taken in the decision node
    :param parent: (DecisionNode)
    """

    def __init__(
        self, 
        action: Any, 
        parent: DecisionNode =None,
        cumulative_reward: float = 0.0,
        eval_reward: float = 0.0,
        visits: int = 0
    ):
        self.action = action
        self.children: Dict[Any, DecisionNode] = {}
        self.cumulative_reward: float = cumulative_reward
        self.parent: DecisionNode = parent
        self.lock = Lock()
        self.eval_reward = eval_reward
        self.visits: int = visits

    def add_children(
        self, 
        x: DecisionNode, 
        hash_preprocess
    ):
        """
        adds a DecisionNode object to the dictionary of children (key is the state)

        :param x: (DecisinNode) the decision node to add to the children dict
        """
        with self.lock:
            self.children[hash_preprocess(x.state)] = x

    def __repr__(self):
        mean_rew = round(self.cumulative_reward/(self.visits+1), 2)
        s = "action: {}\ncumulative reward: {}\nvisits: {}".format(self.action, mean_rew, self.visits)
        return s
