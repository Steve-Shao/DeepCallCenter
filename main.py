#!/usr/bin/env python
# coding: utf-8

"""
Main script for training a dynamic scheduling model using BSDE solver.
Handles data loading, model configuration, and training execution.
"""

import logging
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import argparse

from equation import DynamicScheduling
from torchbsde.torchbsde.solver import BSDESolver
from simulator_nn import CallCenterNNSimulator

# Set dimension
dim = 17

# Set directories
bsde_config_dir = f"config-bsde/config_{dim}dim/"
simulator_config_dir = f"config-simulator/config_{dim}dim/"
test_results_dir = "results"

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    parser.add_argument(
        "--num_nodes", 
        type=int,
        default=100,
        help="Number of nodes per hidden layer"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="LeakyReLU",
        help="Activation function to use"
    )
    parser.add_argument(
        "--ref_policy",
        type=str,
        default="minimal",
        help="Reference policy to use"
    )
    parser.add_argument(
        "--smoothing_steps",
        type=int,
        default=3000,
        help="Number of smoothing preparation steps"
    )
    parser.add_argument(
        "--negative_grad_penalty",
        type=float,
        default=1.0,
        help="Penalty for negative gradients"
    )
    parser.add_argument(
        "--bn_input",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="Whether to use BN in input layer"
    )
    parser.add_argument(
        "--bn_hidden",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="Whether to use BN in hidden layers"
    )
    parser.add_argument(
        "--bn_output",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="Whether to use BN in output layer"
    )
    args = parser.parse_args()
    if args.num_layers < 1:
        raise ValueError("Number of layers must be at least 1")

    # Load data for bsde
    mu = pd.read_csv(bsde_config_dir + f"mu_hourly_{dim}dim.csv", header=None)[0].to_numpy()  # Service rates
    theta = pd.read_csv(bsde_config_dir + f"theta_hourly_{dim}dim.csv", header=None)[0].to_numpy()  # Abandonment rates  
    cost = pd.read_csv(bsde_config_dir + f"cost_{dim}dim.csv", header=None)[0].to_numpy()  # Holding costs
    lambda_matrix = pd.read_csv(bsde_config_dir + f"lambd_matrix_hourly_{dim}dim.csv", header=None, delimiter=",").to_numpy()  # Arrival rates
    zeta_matrix = pd.read_csv(bsde_config_dir + f"zeta_matrix_hourly_{dim}dim.csv", header=None, delimiter=",").to_numpy()  # Second order terms

    # Solver configuration dictionary
    config_bsde = {
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
            "zeta": zeta_matrix,
            "policy": args.ref_policy,
            "x_rand_radius": 10.0,
            "smoothing_prep_steps": args.smoothing_steps
        },
        "network_config": {
            "use_bn_input": args.bn_input,
            "use_bn_hidden": args.bn_hidden,
            "use_bn_output": args.bn_output,
            "use_shallow_y_init": False,
            "num_hiddens": [args.num_nodes] * args.num_layers,
            "activation_function": args.activation,
            "careful_nn_initialization": False
        },
        "solver_config": {
            "batch_size": 256,
            "valid_size": 1024,
            "lr_scheduler": "reduce_on_plateau",
            "lr_plateau_warmup_step": args.smoothing_steps,
            "lr_plateau_patience": 20, 
            "lr_plateau_threshold": 1e-5,
            "lr_plateau_cooldown": 20,
            "lr_plateau_min_lr": 1e-5,
            "lr_start_value": 1e-3,
            "lr_decay_rate": 0.5,
            # "lr_boundaries": [500, 700, 900],
            "num_iterations": 20000,
            "logging_frequency": 10,
            "negative_grad_penalty": args.negative_grad_penalty,
            "delta_clip": 50,
            "verbose": True,
        },
        "dtype": "float64",
        "test_folder_path": test_results_dir,
        "test_scenario_name": f"WS_{args.bn_input}_{args.bn_hidden}_{args.bn_output}", 
        "timezone": "America/Chicago"
    }

    bsde = DynamicScheduling(config_bsde["equation_config"], device=device, dtype=dtype)

    logging.info('Begin to solve %s ' % config_bsde["equation_config"]["eqn_name"])
    bsde_solver = BSDESolver(config_bsde, bsde, device=device, dtype=dtype)
    bsde_solver.train()
    bsde_solver.save_results()
    bsde_solver.plot_y0_history()
    bsde_solver.plot_training_history()
    bsde_solver.model.plot_subnet_gradients()

    # ==========

    lambd_5min = pd.read_csv(simulator_config_dir + f"main_test_total_arrivals_partial_5min.csv", header=None)[0].to_numpy()  # Arrival rates
    lambd_trans_hourly = pd.read_csv(simulator_config_dir + f"lambd_matrix_hourly_{dim}dim.csv", header=None, delimiter=",").to_numpy()  # Arrival rates
    mu_hourly = pd.read_csv(simulator_config_dir + f"mu_hourly_{dim}dim.csv", header=None)[0].to_numpy()  # Service rates
    theta_hourly = pd.read_csv(simulator_config_dir + f"theta_hourly_{dim}dim.csv", header=None)[0].to_numpy()  # Abandonment rates 
    arr_cdf = pd.read_csv(simulator_config_dir + f"cdf_{dim}dim.csv", header=None, delimiter=",").to_numpy() 
    cost_holding_hourly = pd.read_csv(simulator_config_dir + f"hourly_holding_cost_{dim}dim.csv", header=None)[0].to_numpy() 
    cost_abandonment = pd.read_csv(simulator_config_dir + f"abandonment_cost_{dim}dim.csv", header=None)[0].to_numpy() 
    cost_total_hourly = pd.read_csv(simulator_config_dir + f"hourly_total_cost_{dim}dim.csv", header=None)[0].to_numpy() 
    num_server = pd.read_csv(simulator_config_dir + f"main_test_agents.csv", header=None)[0].to_numpy()  # Arrival rates
    num_server_init = pd.read_csv(simulator_config_dir + f"initialization_{dim}dim.csv", header=None)[0].to_numpy() 


    # Simulator configuration dictionary
    config_simulator = {
        "_comment": "Dynamic scheduling config for call center system",
        "num_state_variables": dim,
        "policy": "c_mu_theta",
        "num_interval": 204,
        "lambda_5min": lambd_5min,
        "lambd_trans_hourly": lambd_trans_hourly,
        "mu_hourly": mu_hourly, 
        "theta_hourly": theta_hourly,
        "arr_cdf": arr_cdf,
        "cost_holding_hourly": cost_holding_hourly,
        "cost_abandonment": cost_abandonment,
        "cost_total_hourly": cost_total_hourly,
        "num_server": num_server,
        "num_server_init": num_server_init
    }

    # Initialize the simulator
    simulator = CallCenterNNSimulator(
        bsde_solver = bsde_solver,
        config = config_simulator,
        num_paths = 10000,
        device = device,
        seed = 42
    )

    simulator.simulate_nn_policy()

    # # Print all history records
    # print("=== Full History ===")
    # for i, record in enumerate(simulator.history):
    #     print(f"\nRecord {i}:")
    #     print("Time  :", record["time"].tolist())
    #     print("State :", record["state"].t().tolist())
    #     print("Reward:", record["reward"].tolist())

    # # Print final state and time
    # print("\n=== Final Results ===")
    # print("Terminal State:", [f"{x:.2f}" for x in simulator.current_states.float().mean(dim=1).tolist()])
    # print("Terminal Time:", f"{simulator.current_times.mean().item():.2f}")
    
    # print("Total Loss:", f"{simulator.total_reward.mean().item():.2f}")  # Negative since rewards are costs
    # print("Total Loss by Class:", [f"{x:.2f}" for x in simulator.total_reward_by_class.mean(dim=1).tolist()])
    
    # print(f"Ending simulation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()