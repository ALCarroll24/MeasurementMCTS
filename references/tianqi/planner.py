import numpy as np
from typing import List, Any, Callable, Dict, Tuple

class planner:

    def __init__(
        self,
        F: np.ndarray,
        Q: np.ndarray,
        H: np.ndarray,
        sensor_model: Callable[[Any], np.ndarray],
        v_max: float,
        horizon: float,
        dt: float,
        cdt: float,
        label_weight: Dict[str, float],
    ) -> None:
        self.F = F
        self.Q = Q
        self.H = H
        self.sensor_model = sensor_model
        self.v_max = v_max
        self.horizon = horizon
        self.dt = dt
        self.cdt = cdt
        self.label_weight = label_weight


    def plan(
        self, 
        sesor_state: List[List[float]], 
        traj_estimation: Dict[str, dict],
    ) -> Tuple[Tuple[List[float]], List[Tuple[float]]]:
        '''
        taking sensor state and traj_estimation, generate the waypoints
        for UAV
        '''
        pass

class worst_traj_follow_planner(planner):

    def __init__(self, 
        F: np.ndarray, 
        Q: np.ndarray, 
        H: np.ndarray, 
        sensor_model: Callable[[Any], np.ndarray],
        v_max: float, 
        horizon: float, 
        dt: float, 
        cdt: float,
        label_weight: Dict[str, float]
    ):
        super().__init__(F, Q, H, sensor_model, v_max, horizon, dt, cdt, label_weight)

    def cal_velocity(self, dx: float, dy: float) -> List[float]:
        dist_ = np.sqrt(dx**2 + dy**2)
        
        velocity = min(self.v_max, dist_ / self.cdt)
        vx = round(dx * velocity / dist_, 1)
        vy = round(dy * velocity / dist_, 1)
        return [vx, vy, velocity]

    def plan(
        self, 
        single_sesor_state: List[float], 
        traj_estimation: Dict[str, dict],
    ) -> Tuple[Tuple[List[float]], List[Tuple[float]]]:
        '''
        Pick the trajectory with largest covarance to chase
        '''
        x, y = single_sesor_state
        
        worst_label, worst_index, trace_max = None, -1, -np.inf
        for label, trajInTime in traj_estimation.items():

            for index, P in enumerate(trajInTime['P_list']):
                trace_val = np.trace(P) * self.label_weight[label]
                if trace_val > trace_max:
                    worst_label = label
                    worst_index = index
                    trace_max = trace_val
        
        # determine the worst one, just follow it
        if not worst_label:
            return [(single_sesor_state)] * self.horizon * int(self.cdt/self.dt), \
                [(0.0, 0.0, 0.0)] * self.horizon * int(self.cdt/self.dt)

        traj = traj_estimation[worst_label]['traj'][worst_index]

        return self.generate_wpts_w_ref_traj(traj, x, y)


    def generate_wpts_w_ref_traj(
        self, 
        traj: List[List[float]], 
        x: float, 
        y: float
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        '''
        given waypoints, generate waypoints based on referenced traj
        '''
        wp_command = []
        vel_command = []
        
        for wp in traj:
            x_ref, y_ref = wp
            dx, dy = x_ref - x, y_ref - y
            vx, vy, velocity = self.cal_velocity(dx, dy)

            for _ in range(int(self.cdt / self.dt)):
                x += vx * self.dt
                y += vy * self.dt
                wp_command.append((x, y))
                vel_command.append((vx, vy, velocity))
        return wp_command, vel_command


