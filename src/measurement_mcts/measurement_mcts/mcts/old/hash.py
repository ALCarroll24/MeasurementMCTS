from typing import Tuple, List
import numpy as np


def hash_state(state: Tuple[np.ndarray, np.ndarray, np.ndarray, int]) -> Tuple[Tuple, Tuple, Tuple, int]:
    '''
    Take in the state Tuple(car_state, corner_means, corner_cov, horizon) and convert inner np.ndarrays to tuples
    This makes the state hashable
    '''
    # First, convert the np.ndarrays to tuples
    state_list = list()
    for s in range(len(state)-1):
        state_list.append(tuple(state[s].flatten()))
        
    # Then, append the horizon (int)
    state_list.append(state[-1])
    
    return tuple(state_list)

def hash_action(
    action: np.ndarray
    ) -> Tuple[float, float]:  
    '''
    '''
    
    return tuple(action)

# def hash_state(
#     state: Tuple[Tuple[float, float, float], List[np.ndarray], int],
#     # ego_pos: Tuple[float, float, float], 
#     # P_list: List[np.ndarray],
#     # horizon: int,
# ):  
#     '''
#     A trick we used for hashing np.matrix type for covariances
#     https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
#     '''
#     info_list = list()
#     ego_pos, P_list, horizon = state

#     for state in ego_pos:
#         info_list.append(state)
    
#     for P in P_list:
#         # info_list.append(np.array2string(P))   # test time 0.938 s

#         # data.tobytes()  hash time: 0.074
#         P.flags.writeable = False
#         info_list.append(P.data.tobytes())
#         P.flags.writeable = True
    
#     info_list.append(horizon)
    
#     return tuple(info_list)
