from typing import Tuple, List
import numpy as np

def hash_state(
    state: Tuple[Tuple[float, float, float], List[np.ndarray], int],
    # ego_pos: Tuple[float, float, float], 
    # P_list: List[np.ndarray],
    # horizon: int,
):  
    '''
    A trick we used for hashing np.matrix type for covariances
    https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
    '''
    info_list = list()
    ego_pos, P_list, horizon = state

    for state in ego_pos:
        info_list.append(state)
    
    for P in P_list:
        # info_list.append(np.array2string(P))   # test time 0.938 s

        # data.tobytes()  hash time: 0.074
        P.flags.writeable = False
        info_list.append(P.data.tobytes())
        P.flags.writeable = True
    
    info_list.append(horizon)
    
    return tuple(info_list)

def hash_action(
    action: Tuple[float, float], 
) -> Tuple[float, float]:  
    '''
    '''
    
    return tuple(action)