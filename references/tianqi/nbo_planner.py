import numpy as np
from nav2_air_active_track_planner.util.planner import planner, worst_traj_follow_planner
from nav2_air_active_track_planner.util.util import pdf_construct, euclidean_distance, stamp_add_duration
from typing import Any, Callable, Dict, List, Optional, Tuple
from active_track_msgs.msg import SensorPlanArray
from geometry_msgs.msg import Pose
from nav2_air_active_track_planner.mcts.spw import SPW
from nav2_air_active_track_planner.mcts.hash import hash_action, hash_state
from active_track_msgs.msg import SensorPlan, SensorPlanArray, Fov
from nav2_air_active_track_planner.util.util import locate_time_index
from rclpy.node import Node


class nbo_planner(worst_traj_follow_planner):

    def __init__(self, 
        F: np.ndarray, 
        Q: np.ndarray, 
        H: np.ndarray, 
        sensor_model: Callable[[Any], np.ndarray], 
        v_max: float, 
        horizon: float, 
        dt: float, 
        cdt: float, 
        label_weight: Dict[str, float],
        team_size: int,
        agent_id: int,
        precision_matrix: int = 2,
        action_sample_heuristic: str = 'uniform',
        mcts_max_iter: int = 1000,
        precision_state: int = 3,
        gamma: float = 1.0,
        wtp: bool = True,
        rosnode: Node = Optional[Node],
    ):
        super().__init__(F, Q, H, sensor_model, v_max, horizon, dt, cdt, label_weight)
        
        self.gamma = gamma      # discount factor
        # policy over time
        self.u: Dict[int, List[List[float]]] = dict()
        self.team_size = team_size
        self.id = agent_id
        self.precision_matrix = precision_matrix
        self.precision_state = precision_state
        self.mcts_max_iter = mcts_max_iter
        self.action_sample_heuristic = action_sample_heuristic
        self.wtp = wtp
        self._init_other_agents_future_pose()
        self.rosnode = rosnode

    def _init_other_agents_future_pose(self):
        self.other_agents_future_pose: Dict[int, List[List[float]]] = dict()
        
        for agent_id in range(1, self.team_size + 1):
            
            self.other_agents_future_pose[agent_id] = list()
    
    def _reset_other_agents_future_pose(self):

        for agent_future_pose in self.other_agents_future_pose.values():
            agent_future_pose.clear()
        

    def CheckFovVisibility(
        self, 
        z: List[float],
        agent_id: int, 
        agent_pose: List[float]
    ) -> bool:
        
        if agent_id == 0:
            raise ValueError('agent_id should not be 0')
        fov: Fov = self.group_policy.plans[agent_id - 1].fov

        if fov.geometry == 'circle':
            radius = fov.configs[0]
            dist = euclidean_distance(z, agent_pose)
            return dist <= radius
        else:
            raise ValueError(f'fov geometry {fov.geometry} not supported')

    def process_policy(
        self,
        group_policy: SensorPlanArray,
        time_sec: float,
    ):
        '''
        process the policy from other agents
        '''
        self.group_policy = group_policy
        for sensorplan in group_policy.plans:

            # skip ego intention
            if sensorplan.robot_id == self.id:
                continue
            
            agent_j_future_pose: List[List[float]] = list()

            if len(sensorplan.timestep) != 0:

                for i in range(self.horizon):

                    
                    t = time_sec + (i + 1) * self.cdt
                    time_index = locate_time_index(time_list=sensorplan.timestep,
                                        time_sec=t, rosnode=self.rosnode)
                    # linear interpolation of the position
                    # if time_index == len(sensorplan.timestep) - 1:
                    dt = t - sensorplan.timestep[time_index]
                    wp = sensorplan.waypoints[time_index]
                    x = wp.pose.position.x + wp.twist.linear.x * dt
                    y = wp.pose.position.y + wp.twist.linear.y * dt
                    new_yaw = np.arctan2(wp.twist.linear.y, wp.twist.linear.x)
                    agent_j_future_pose.append([x, y, new_yaw])
            
            self.other_agents_future_pose[sensorplan.robot_id] = agent_j_future_pose
            # self.rosnode.get_logger().info(f'!!! {sensorplan.robot_id} intention length {len(agent_j_future_pose)}')
                

    def plan(
        self, 
        single_sesor_state: List[float], 
        traj_estimation: Dict[str, dict],
        group_policy: SensorPlanArray,
        time_sec: float,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float]], \
               List[Tuple[float, float]], List[Tuple[float, float, float]]]:
        '''
        run the MCTS online optimization over horizon
        '''
        
        P_list = []
        self.trajInTime = []
    
        self._reset_other_agents_future_pose()
        # fill in the other_agents_future_pose
        self.process_policy(group_policy=group_policy, time_sec=time_sec)

        # nominal trajectory
        for j in range(self.horizon):
            self.trajInTime.append([])

        for label, trajInTime in traj_estimation.items():

            for P in trajInTime['P_list']:

                P_list.append((P * self.label_weight[label]).round(self.precision_matrix))

            for traj in trajInTime['traj']:
                for j in range(self.horizon):
                    self.trajInTime[j].append(traj[j])
        
        x, y = single_sesor_state
        agent_pos = (x, y)
        horizon = 0
        init_state = (agent_pos, P_list, horizon)

        # prof = cProfile.Profile()
        # prof.enable()
        mcts_model = SPW(
            initial_obs = init_state, 
            env = self, 
            K=0.3**5,
            _hash_action = hash_action, 
            _hash_state = hash_state, 
            alpha=0.5)
        mcts_model.learn(self.mcts_max_iter, progress_bar=True)
        action_vector = mcts_model.best_action()

        # generate waypoints  
        wp_command, vel_command = [], []
        wp_intention, vel_intention = [], []

        for action in action_vector:
            vx, vy = action
            velocity = np.sqrt(vx**2 + vy**2)
            wp_intention.append((x + vx * self.cdt, y + vy * self.cdt))
            vel_intention.append((vx, vy, velocity))

            for _ in range(int(self.cdt / self.dt)):
                x += vx * self.dt
                y += vy * self.dt
                wp_command.append((x, y))
                vel_command.append((vx, vy, velocity))
        
        return wp_command, vel_command, wp_intention, vel_intention
    
    def step(self, 
        state: Tuple[Tuple[float, float, float], List[np.ndarray], int], 
        action: Tuple[float, float]
    ) -> Tuple[Tuple[Tuple[float, float, float], List[np.ndarray], int], float, bool]:
        '''
        transition function
        '''
        objValue = 0.0
        agent_pos, P_list, horizon = state
        z_k = self.trajInTime[horizon]

        is_final = (horizon == self.horizon - 1)

        # agent new position
        new_x = self.cdt * action[0] + agent_pos[0]
        new_y = self.cdt * action[1] + agent_pos[1]
        new_yaw = np.arctan2(action[1], action[0])
        agent_new_pos = (round(new_x, self.precision_state), 
                         round(new_y, self.precision_state), 
                         round(new_yaw, self.precision_state))
        new_P_list = []
        
        # start using a sequential way to estimate KF
        for j in range(len(z_k)):
            z = z_k[j]
            
            P_k_k_min = self.KF_predict(P_list[j])
            isObserved = False
            R = np.matrix(np.zeros((2, 2)))
            # if not self.isInsideOcclusion_planning(z[0:2]):  # TODO when sml added
            
            # find any observing agents
            for agent_id in range(1, self.team_size + 1):
                # edge case: first iteration no other agents intention
                if len(self.other_agents_future_pose[agent_id]) == 0:
                    # self.rosnode.get_logger().info(f'agent {agent_id} at {horizon} intention length {len(self.other_agents_future_pose[agent_id])}')
                    continue
                if agent_id == self.id:
                    agent_pose = agent_new_pos
                else:
                    agent_pose = self.other_agents_future_pose[agent_id][horizon]

                if self.CheckFovVisibility(z, agent_id, agent_pose):

                    # Woodbury matrix identity for KF
                    # https://en.wikipedia.org/wiki/Woodbury_matrix_identity#Special_cases
                    R_i = self.cal_R_cen(z, agent_pose, agent_id) 
                    R += np.linalg.inv(R_i)
                    isObserved = True

            # update all agents track
            if isObserved:
                info_sum = np.dot(np.dot(self.H.T, R), self.H)
                P_fused = np.linalg.inv(np.linalg.inv(P_k_k_min) + info_sum)
                P_k_k = P_fused
                                
            else:
                P_k_k = P_k_k_min
            
            new_P_list.append(P_k_k.round(self.precision_matrix))
            trace = np.trace(P_k_k[0:2, 0:2])                            
            objValue += trace # (self.gamma ** i) # * weight
        
        # prepare output

        new_state = (agent_new_pos, new_P_list, horizon + 1)
        reward = - objValue
        return new_state, reward, is_final
    
    def KF_predict(self, P: np.ndarray) -> np.ndarray:
        P_k_k_min = np.dot(self.F, np.dot(P,self.F.T)) + self.Q
        return P_k_k_min

    def cal_R_cen(
        self, 
        z: List[float], 
        xs: List[float], 
        agent_id: int,
    ) -> np.ndarray:
        # xs: sensor pos, = [x, y, theta]
        # z: measurement, [x, y] 
        dx = z[0] - xs[0]
        dy = z[1] - xs[1]
        r0 = 0.5
        alpha = 1.0
        theta = np.arctan2(dy, dx) - xs[2]
        r = max(r0, np.sqrt(dx**2 + dy**2))
        G = np.matrix([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]])
        M = np.diag([0.1 * r, 0.1 * np.pi * r])
        R = alpha * np.dot(np.dot(G, M), G.T)
        
        return R


    def evaluate(
        self, 
        state: Tuple[Tuple[float, float, float], List[np.ndarray], int]
    ) -> float:
        '''
        evaluation of the terminal node in MCTS
        '''
        
        if self.wtp:
            agent_pos, P_list, horizon = state

            agent_pos_dict = dict()
            neighbor_agent_list = list()
            for agent_id in range(1, self.team_size + 1):
                
                if agent_id == self.id:
                    agent_pose = [agent_pos[0], agent_pos[1], agent_pos[2]]
                else:
                    if len(self.other_agents_future_pose[agent_id]) == 0:
                        continue
                    agent_pose = [self.other_agents_future_pose[agent_id][horizon - 1][0],
                                self.other_agents_future_pose[agent_id][horizon - 1][1],
                                self.other_agents_future_pose[agent_id][horizon - 1][2]]
                agent_pos_dict[agent_id] = agent_pose
                neighbor_agent_list.append(agent_id)
                        
            wtp = 0.0
            Dj = {}
            for agent_id in neighbor_agent_list:
                Dj[agent_id] = 0.0

            trace_list_ordered = []
            
            for j in range(len(self.trajInTime[-1])):
                z = self.trajInTime[-1][j]            
                
                # if self.isInsideOcclusion_planning(z[0:2]):
                #     continue
                isoutside = True
                for k in neighbor_agent_list:
                    # isoutside = (not agent_list[k].isInFoV(z)) and isoutside
                    if self.CheckFovVisibility(z, k, agent_pos_dict[k]):
                        isoutside = False
                        break
                if isoutside:
                    P_trace = np.trace(P_list[j][0:2, 0:2])
                    trace_list_ordered.append([P_trace, z])
            trace_list_ordered.sort(key=lambda r: r[0], reverse=True)
            # go with the order
            for j in range(len(trace_list_ordered)):
                z = trace_list_ordered[j][1]
                # find the minimium distance from sensor to target
                distance = 10000
                new_dist = 100000
                agent_id = -1
                for k in neighbor_agent_list:
                    d_k = euclidean_distance(z, agent_pos_dict[k])
                    # print("d_k %s, distance %s, Dj %s, k %s" % (d_k, distance, Dj, k))
                    if d_k + Dj[k] < new_dist:
                        agent_id = k
                        distance = d_k
                        new_dist = d_k + Dj[k]
                if agent_id > 0 and Dj[k] == 0:
                    wtp += self.gamma * distance * np.trace(P_list[j][0:2, 0:2])
                Dj[agent_id] += distance
                # change sensor position to the place to cover its fov
                r = self.find_min_fov_dimension(fov=self.group_policy.plans[agent_id - 1].fov)
                angle = np.arctan2(agent_pos_dict[k][1] - z[1], agent_pos_dict[k][0] - z[0])
                agent_pos_dict[k][0] += agent_pos_dict[k][0] - z[0] - r * np.cos(angle)
                agent_pos_dict[k][1] += agent_pos_dict[k][1] - z[1] - r * np.sin(angle)
             
            return - wtp
        else:
            return 0.0
    
    def find_min_fov_dimension(self, fov: Fov) -> float:
        '''
        find the minimum dimension of the fov
        '''
        if fov.geometry == 'circle':
            return fov.configs[0]
        elif fov.geometry == 'rectangle':
            return min(fov.configs[0], fov.configs[1])
        else:
            raise ValueError(f'fov geometry {fov.geometry} not supported')
        

    def action_space_sample(
        self, 
        state: Tuple[Tuple[float, float, float], List[np.ndarray], int], 
    ) -> Tuple[float, float]:

        if self.action_sample_heuristic == 'uniform':

            return self.action_space_sample_uniform()

        elif self.action_sample_heuristic == 'target_dist_heuristic':
            
            return self.action_space_sample_target_dist_heurisitic(state=state)
        
        else:   
            raise ValueError(f'action sample heuristic {self.action_sample_heuristic} not supported')

    def action_space_sample_uniform(self,) -> Tuple[float, float]:

        velocity = np.random.uniform(low=0.0, high=self.v_max,)
        theta = np.random.uniform(low = 0.0, high=2 * np.pi)
        vx = velocity * np.cos(theta)
        vy = velocity * np.sin(theta)
        return (vx, vy)

    def action_space_sample_target_dist_heurisitic(
        self, 
        state: Tuple[Tuple[float, float, float], List[np.ndarray], int], 
        std_theta: float = 0.05,
    ) -> Tuple[float, float]:
        '''
        this sample is based on the selection of uncovered targets around
        agent
        '''
        
        agent_pos, _, horizon = state
        z_k = self.trajInTime[horizon]

        angle_list = []
        selection_rate = []
        # check uncovered targets
        for j in range(len(z_k)):
            z = z_k[j]
            isObserved = False
            
            # find any observing agents
            for agent_id in range(1, self.team_size + 1):
                if len(self.other_agents_future_pose[agent_id]) == 0:
                    continue

                if agent_id == self.id:
                    continue
                else:
                    # self.rosnode.get_logger().info(f'agent {agent_id} at {horizon} intention length {len(self.other_agents_future_pose[agent_id])}')

                    agent_pose = self.other_agents_future_pose[agent_id][horizon]
                if self.CheckFovVisibility(z, agent_id, agent_pose):
                    isObserved = True
                    break
            dx = z[0] - agent_pos[0]
            dy = z[1] - agent_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            if not isObserved and distance < 2 * self.horizon * self.v_max:
                angle_list.append(np.arctan2(dy, dx))
                selection_rate.append(1/distance)
                # selection_rate.append(distance * np.trace(P_list[j]))

        if not angle_list:
            return self.action_space_sample_uniform()
        
        # select one target to move forward
        weights = np.array(selection_rate)
        weights /= np.sum(weights)
        theta = np.random.choice(angle_list, p=weights)
        # add some Gaussian noise
        theta = np.random.normal(loc=theta, scale=std_theta)
        velocity = np.random.uniform(low=0.0, high=self.v_max,)
        vx = velocity * np.cos(theta)
        vy = velocity * np.sin(theta)
        return (vx, vy)