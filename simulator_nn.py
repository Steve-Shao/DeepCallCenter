import torch
import pandas as pd
from typing import Dict, Any, Optional

from simulator import CallCenterSimulator


class CallCenterNNSimulator(CallCenterSimulator):
    def __init__(
        self,
        bsde_solver,
        config: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
        accuracy: int = 32,
        num_paths: int = 10,
        seed: Optional[int] = None
    ):
        super().__init__(config, device, accuracy, num_paths, seed)
        
        # Initialize model
        self.model = bsde_solver.model
        self.logger = bsde_solver.logger
        # Scaling factor
        self.N = torch.tensor(400, device=self.device, dtype=self.dtype_float)
        self.lambda_trans_hourly_by_interval = torch.tensor(config["lambd_trans_hourly"], device=self.device, dtype=self.dtype_float)
        # Get number of nn steps
        self.num_nn_steps = self.model.bsde.num_time_interval
        # Check if num_nn_steps is a multiplier of num_interval
        if self.num_nn_steps % (self.max_interval+1) != 0:
            raise ValueError(f"num_nn_steps ({self.num_nn_steps}) must be a multiple of num_interval ({self.max_interval+1})")
        self.nn_step_multiplier = self.num_nn_steps // (self.max_interval+1)
        self.hour_to_step_scaler = self.hour_to_interval_scaler * self.nn_step_multiplier
    
    def _update_control_rules(self):
        """
        Update the control rules for the call center system.
        In the current implementation, we use constant, pre-emptive, priority-based control rules.
        Each sample path has its own priority order based on the policy rule.
        """
        # Calculate mean field value by combining arrival rate, population size and service rate
        Q = self.current_states.t().to(dtype=self.model.dtype)
        X = (Q - self.current_mean_field) / torch.sqrt(self.N)
        gradient = self.current_nn(X, training=False)
        self.kappa = (self.mu_hourly - self.theta_hourly)[None, :] * gradient
        self.kappa += self.cost_total_hourly[None, :]
        self.kappa *= -1.0
        
        # Sort priority order by policy rule
        self.priority_order = torch.argsort(self.kappa, dim=1)

    def _update_actions(self):
        """
        Compute or retrieve the current action for each sample path, given the current state.
        This implements a priority-based queueing discipline where:
        1. Customer classes are ordered by priority based on the policy rule (cost, c_mu_theta, etc.)
        2. Available agents are assigned to customers in order of class priority
        3. Within each class, agents serve as many customers as possible up to:
           - The number of waiting customers in that class
           - The number of remaining available agents
        4. Any remaining agents move on to serve the next priority class
        Updates self.next_actions with shape (A, num_paths). 
        Updates self.waiting_customers with shape (S, num_paths). 
        """
        # Update control rules
        self._update_control_rules()

        # Reset the last actions to 0
        self.next_actions.zero_()
        
        # Keep track of remaining agents for each path
        remaining_agents = self.num_server[self.current_intervals].clone()
        
        # Iterate over priority levels
        for priority in range(self.S):
            # Get class indices for current priority level across all paths
            class_idxs = self.priority_order[:, priority]
            
            # Gather the number of waiting customers for the selected classes
            waiting = self.current_states[class_idxs, torch.arange(self.num_paths, device=self.device)]
            
            # Determine how many callers can be served
            callers_served = torch.minimum(waiting, remaining_agents)
            
            # Update actions
            self.next_actions[class_idxs, torch.arange(self.num_paths, device=self.device)] = callers_served
            
            # Update remaining agents
            remaining_agents -= callers_served
        
        # Update the number of waiting customers
        self.waiting_customers = self.current_states - self.next_actions 
    
    def simulate_nn_policy(self):
        self.logger.info("========== Neural Network Policy Simulation ==========")
        self.logger.info(f"Number of paths: {self.num_paths}")
        self.logger.info(f"Number of steps: {self.num_nn_steps}")
        self.logger.info(f"Step multiplier: {self.nn_step_multiplier}")

        self.logger.info("========== Simulation Started ==========")
        for i in range(self.num_nn_steps):
            current_interval = int(i * 1.0 / self.nn_step_multiplier)
            if i == 0 or current_interval != int((i-1) * 1.0 / self.nn_step_multiplier):
                self.logger.info(f"Simulating interval {current_interval+1} / {self.max_interval+1}")

            self.current_intervals = torch.full((self.num_paths,), current_interval, dtype=self.dtype_int, device=self.device)
            self.current_mean_field = (self.N * 
                                       self.lambda_trans_hourly_by_interval[:,current_interval] /
                                       self.mu_hourly)
            self.current_nn = self.model.subnet[i]
            self.run_until_time(target_time=(i+1) / self.hour_to_step_scaler)
            
        self.logger.info("========== Simulation Ended ==========")
        self.logger.info(f"Terminal State: {[f'{x:.2f}' for x in self.current_states.float().mean(dim=1).tolist()]}")
        self.logger.info(f"Terminal Time: {self.current_times.mean().item():.2f}")
        self.logger.info(f"Total Loss: {self.total_reward.mean().item():.2f}")
        self.logger.info(f"Total Loss by Class: {[f'{x:.2f}' for x in self.total_reward_by_class.mean(dim=1).tolist()]}")



if __name__ == "__main__":

    ########################################################
    # Example Usage
    ########################################################
    # Load system data from CSV files
    dim = 17
    data_dir = f"config-simulator/config_{dim}dim/"

    lambd_5min = pd.read_csv(data_dir + f"main_test_total_arrivals_partial_5min.csv", header=None)[0].to_numpy()  # Arrival rates
    mu_hourly = pd.read_csv(data_dir + f"mu_hourly_{dim}dim.csv", header=None)[0].to_numpy()  # Service rates
    theta_hourly = pd.read_csv(data_dir + f"theta_hourly_{dim}dim.csv", header=None)[0].to_numpy()  # Abandonment rates 
    arr_cdf = pd.read_csv(data_dir + f"cdf_{dim}dim.csv", header=None, delimiter=",").to_numpy() 
    cost_holding_hourly = pd.read_csv(data_dir + f"hourly_holding_cost_{dim}dim.csv", header=None)[0].to_numpy() 
    cost_abandonment = pd.read_csv(data_dir + f"abandonment_cost_{dim}dim.csv", header=None)[0].to_numpy() 
    cost_total_hourly = pd.read_csv(data_dir + f"hourly_total_cost_{dim}dim.csv", header=None)[0].to_numpy() 
    num_server = pd.read_csv(data_dir + f"main_test_agents.csv", header=None)[0].to_numpy()  # Arrival rates
    num_server_init = pd.read_csv(data_dir + f"initialization_{dim}dim.csv", header=None)[0].to_numpy() 

    # Model configuration dictionary
    config = {
        "_comment": "Dynamic scheduling config for call center system",
        "num_state_variables": dim,
        "policy": "c_mu_theta",
        "num_interval": 204,
        "lambda_5min": lambd_5min,
        "mu_hourly": mu_hourly, 
        "theta_hourly": theta_hourly,
        "arr_cdf": arr_cdf,
        "cost_holding_hourly": cost_holding_hourly,
        "cost_abandonment": cost_abandonment,
        "cost_total_hourly": cost_total_hourly,
        "num_server": num_server,
        "num_server_init": num_server_init
    }

    # Set device
    # device = "cpu"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    import time
    from datetime import datetime

    print(f"Starting simulation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize the simulator
    simulator = CallCenterNNSimulator(
        config=config,
        num_paths=10000,
        device=device,
        seed=None
    )

    simulator.run_until_time(target_time=0.1)

    # # Print all history records
    # print("=== Full History ===")
    # for i, record in enumerate(simulator.history):
    #     print(f"\nRecord {i}:")
    #     print("Time  :", record["time"].tolist())
    #     print("State :", record["state"].t().tolist())
    #     print("Reward:", record["reward"].tolist())

    # Print final state and time
    print("\n=== Final Results ===")
    print("Terminal State:", [f"{x:.2f}" for x in simulator.current_states.float().mean(dim=1).tolist()])
    print("Terminal Time:", f"{simulator.current_times.mean().item():.2f}")
    
    print("Total Loss:", f"{simulator.total_reward.mean().item():.2f}")  # Negative since rewards are costs
    print("Total Loss by Class:", [f"{x:.2f}" for x in simulator.total_reward_by_class.mean(dim=1).tolist()])
    
    print(f"Ending simulation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")