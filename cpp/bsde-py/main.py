#!/usr/bin/env python
# coding: utf-8

"""
Main script for training a dynamic scheduling model using BSDE solver.
Handles data loading, model configuration, and training execution.
"""

import logging
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import argparse

from torchbsde.equation.base import Equation
# from torchbsde.subnet import FeedForwardSubNet
from torchbsde.solver import BSDESolver

# Load 2D system data from CSV files
dim = 17
data_dir = f"tests/dynamic-scheduling/config_{dim}dim/"
mu = pd.read_csv(data_dir + f"mu_hourly_{dim}dim.csv", header=None)[0].to_numpy()  # Service rates
theta = pd.read_csv(data_dir + f"theta_hourly_{dim}dim.csv", header=None)[0].to_numpy()  # Abandonment rates  
cost = pd.read_csv(data_dir + f"cost_{dim}dim.csv", header=None)[0].to_numpy()  # Holding costs
lambda_matrix = pd.read_csv(data_dir + f"lambd_matrix_hourly_{dim}dim.csv", header=None, delimiter=",").to_numpy()  # Arrival rates
zeta_matrix = pd.read_csv(data_dir + f"zeta_matrix_hourly_{dim}dim.csv", header=None, delimiter=",").to_numpy()  # Second order terms

# Model configuration dictionary
config = {
    "equation_config": {
        "_comment": "Dynamic scheduling equation for 2D system",
        "eqn_name": "DynamicScheduling",
        "hours": 17,
        "total_time": 204,
        "dim": dim,
        "num_time_interval": 3060, 
        "theta": theta,
        "mu": mu, 
        "cost": cost,
        "lambd": lambda_matrix,
        "zeta": zeta_matrix
    },
    "network_config": {
        "use_bn_input": False,
        "use_bn_hidden": False,
        "use_bn_output": False,
        "use_shallow_y_init": False,
        "num_hiddens": [100, 100, 100, 100]
        # "num_hiddens": [110, 110]
    },
    "solver_config": {
        "batch_size": 256,
        "valid_size": 1024,
        "lr_scheduler": "reduce_on_plateau",
        "lr_start_value": 1e-3,
        "plateau_patience": 10, 
        "lr_decay_rate": 0.5,
        # "lr_boundaries": [500, 700, 900],
        "num_iterations": 10000,
        "logging_frequency": 10,
        "negative_loss_penalty": 0.0,
        "delta_clip": 50,
        "verbose": True,
    },
    "dtype": "float64",
    "test_folder_path": os.path.dirname(os.path.relpath(__file__)),
    "test_scenario_name": "scenario_1", 
    "timezone": "America/Chicago"
}

# Alternative configuration (commented out)
# config = {
#     "equation_config": {
#         "eqn_name": 'DynamicScheduling',
#         "total_time": 17, 
#         "dim": 2,
#         "num_time_interval": 204,
#         "theta": theta,
#         "mu": mu,
#         "cost": cost,
#         "lambd": lambda_matrix,
#         "zeta": zeta_matrix
#     },
#     "network_config": {
#         "num_hiddens": [100,100,100,100],
#         # "num_hiddens": [110,110],
#         "use_bn_input": True,
#         "use_bn_hidden": False,
#         "use_bn_output": False,
#         "use_shallow_y_init": False,
#         # "lr_values": [1e-3, 0.0005, 1e-4],
#         "lr_values": [1e-2, 0.005, 1e-3],
#         "lr_boundaries": [1200, 1500],
#         "num_iterations": 2000,
#         "batch_size": 512,
#         "valid_size": 512,
#         "logging_frequency": 10,
#         "dtype": "float64",
#         "verbose": True
#     },
#     "solver_config": {
#         "delta_clip": 50,
#         "negative_loss_penalty": 1.0
#     }
# }

#####################################
# Equation / PDE definitions
#####################################

class DynamicScheduling(Equation):
    """
    Implements the dynamic scheduling equation for a 2D queueing system.
    
    This class defines the specific equation components including:
    - System parameters (service rates, abandonment rates, costs)
    - Sampling methods for Brownian paths
    - Generator function f(t,x,y,z) 
    - Terminal condition g(t,x)
    """

    def __init__(self, equation_config, device=None, dtype=None):
        """Initialize equation parameters and move tensors to specified device."""
        super(DynamicScheduling, self).__init__(equation_config, device=device, dtype=dtype)
        
        # Convert numpy arrays to tensors and move to device
        self.zeta = torch.tensor(equation_config["zeta"], dtype=self.dtype, device=self.device)
        self.lambd = torch.tensor(equation_config["lambd"], dtype=self.dtype, device=self.device)
        self.mu = torch.tensor(equation_config["mu"], dtype=self.dtype, device=self.device)
        self.theta = torch.tensor(equation_config["theta"], dtype=self.dtype, device=self.device)
        self.cost = torch.tensor(equation_config["cost"], dtype=self.dtype, device=self.device)

        # Scale the parameters
        self.hours = equation_config["hours"]
        self.scaling_factor = self.total_time / self.hours
        self.zeta = self.zeta / self.scaling_factor    
        self.lambd = self.lambd / self.scaling_factor
        self.mu = self.mu / self.scaling_factor
        self.theta = self.theta / self.scaling_factor
        self.cost = self.cost / self.scaling_factor

        # Repeat lambd, zeta to match time intervals
        num_repeats = self.num_time_interval / self.total_time
        # Check if num_repeats is an integer
        if not num_repeats.is_integer():
            raise ValueError(f"num_repeats must be an integer but got {num_repeats}")
        num_repeats = int(num_repeats)

        # Repeat lambd, zeta to match time intervals
        self.lambd = self.lambd.view(self.dim, -1)  # ensure shape [DIM, something]
        self.lambd = self.lambd.repeat_interleave(num_repeats, dim=1)
        self.zeta = self.zeta.view(self.dim, -1)
        self.zeta = self.zeta.repeat_interleave(num_repeats, dim=1)
        
        # Initialize starting state as uniform(0, 1)
        # self.x_init = torch.tensor(
        #     np.random.uniform(low=0, high=1, size=[1]),
        #     dtype=self.dtype,
        #     device=self.device
        # )
        # Initialize starting state as uniform(-10, 10)
        self.x_init = torch.tensor(
            np.random.uniform(low=-10, high=10, size=[1]),
            dtype=self.dtype,
            device=self.device
        )
        self.c_bar = 2.12  # Terminal cost coefficient
        self.sigma = torch.sqrt(self.lambd)  # Volatility term

    def sample(self, num_sample):
        """
        Sample paths for the stochastic process.
        
        Args:
            num_sample: Number of sample paths to generate
            
        Returns:
            tuple: (dw_sample, x_sample) containing Brownian increments and state paths
        """
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) \
                    * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init.item()

        # Generate paths using Euler-Maruyama discretization
        for i in range(self.num_time_interval):
            for k in range(self.dim):
                x_sample[:, k, i+1] = (
                    x_sample[:, k, i]
                    + (self.zeta[k, i].item() - self.mu[k].item()*x_sample[:, k, i]) * self.delta_t
                    + np.sqrt(2.0)*self.sigma[k, i].item() * dw_sample[:, k, i]
                )
        return dw_sample, x_sample

    def f_torch(self, t, x, y, z):
        """Generator function: f(t,x,y,z)."""
        # x shape: (batch, dim)
        # sum along dim => (batch,)
        mx = torch.sum(x, dim=1)
        mx = torch.clamp(mx, min=0.0)  # tf.maximum(..., 0)
        mx = mx.view(-1, 1).double()   # reshape to (batch,1)

        # cost + (mu - theta)*z => shape (batch, dim)
        # reduce_min => shape (batch,)
        # multiply by mx => shape (batch,1)
        # We'll broadcast over dim.
        # cost is shape (dim,); z is shape (batch, dim)
        cost_broadcast = self.cost - self.theta + self.mu  # shape (dim,)
        # Actually we want cost + (mu - theta), so:
        cost_broadcast = self.cost + (self.mu - self.theta)
        # Then cost_broadcast * z => shape (batch, dim)
        term = cost_broadcast * z
        # reduce_min across dim => shape (batch,)
        out = torch.min(self.cost + (self.mu - self.theta) * z, dim=1).values
        # but let's do it carefully:
        #   cost + (mu - theta) is shape (dim,)
        #   z is shape (batch, dim)
        # so we want to do the per-sample min across dim
        g = torch.min(cost_broadcast + 0.0*z, dim=0)  # but that wouldn't be right.
        # Instead do:
        tmp = self.cost.view(1, -1) + (self.mu - self.theta).view(1, -1) * z
        # shape => (batch, dim)
        min_val = torch.min(tmp, dim=1, keepdim=True)[0]  # => (batch,1)
        return mx * min_val

    def g_torch(self, t, x):
        """
        Terminal condition g(t,x) for the BSDE.
        
        Computes final cost based on terminal state.
        """
        mx = torch.sum(x, dim=1, keepdim=True)
        mx = torch.clamp(mx, min=0.0).to(dtype=self.dtype, device=self.device)
        return self.c_bar * mx


def main():
    """Main execution function to setup and run the BSDE solver."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Dynamic Scheduling Model")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of hidden layers in the network"
    )
    args = parser.parse_args()

    # Update the network configuration with the specified number of layers
    num_layers = args.num_layers
    # Ensure at least one layer
    if num_layers < 1:
        raise ValueError("Number of layers must be at least 1")
    config["network_config"]["num_hiddens"] = [100] * num_layers
    # Update test scenario name based on number of layers
    config["test_scenario_name"] = f"exp1_{num_layers}layers"

    # Setup device and solver
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    bsde = DynamicScheduling(config["equation_config"], device=device, dtype=dtype)

    logging.info('Begin to solve %s ' % config["equation_config"]["eqn_name"])
    bsde_solver = BSDESolver(config, bsde, device=device, dtype=dtype)
    bsde_solver.train()
    bsde_solver.save_results()
    bsde_solver.plot_y0_history()
    bsde_solver.plot_training_history()
    bsde_solver.model.plot_subnet_gradients()


if __name__ == "__main__":
    main()
