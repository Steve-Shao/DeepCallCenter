#!/usr/bin/env python3

import os
import json
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -------------------------------------------------------------------
# 1) Load Config and Initialize
# -------------------------------------------------------------------
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config



# -------------------------------------------------------------------
# 2) Define Utility Functions
# -------------------------------------------------------------------
def leaky_relu_manual(x, slope=0.2):
    return F.leaky_relu(x, negative_slope=slope)

def calculate_negative_loss(tensor):
    """
    Replicates the 'calculate_negative_loss(func)' from Julia.
    In the Julia code:
      zero_func = min.(minimum(func, dims=1), 0.0)
      negative_loss = sum(zero_func.^2)
    We interpret that as: for each sample, we find the minimum across dimension=1,
    clamp it to 0 if positive, and square the result if negative.
    """
    # We need the minimum across dimension 1 (Julia dims=1 => across the row dimension).
    # Then clamp positives to 0, keep negatives as is, then square and sum.
    # Because in the original code, 'func' might be shape [DIM, batch_size], 
    # we first reduce across dim=0 in PyTorch (which is the "DIM" dimension).
    min_vals = torch.min(tensor, dim=0, keepdim=False).values  # shape [batch_size]
    # Keep only negatives
    clamped = torch.minimum(min_vals, torch.tensor(0.0, device=tensor.device))
    # Sum of squares
    return torch.sum(clamped ** 2)

def compute_mx(x_slice, a_lowbound):
    """
    In Julia:
      sum_x = sum(x_slice, dims=1)
      sum_x_p = max.(sum_x, 0.0)
      sum_x_n = min.(sum_x, 0.0)
      return sum_x_p + a_lowbound .* (exp.(sum_x_n) .- 1)
    x_slice: shape [DIM, batch_size]
    Returns a shape [batch_size]
    """
    sum_x = torch.sum(x_slice, dim=0)  # [batch_size]
    sum_x_p = torch.clamp(sum_x, min=0.0)
    sum_x_n = torch.minimum(sum_x, torch.tensor(0.0, device=x_slice.device))
    return sum_x_p + a_lowbound * (torch.exp(sum_x_n) - 1.0)

# -------------------------------------------------------------------
# 3) Neural Network Definitions
# -------------------------------------------------------------------
class DeepNNChain(nn.Module):
    """
    Replicates the Julia createDeepNNChain(dim, units, activation, output_units, output_activation).
    We use a fixed architecture with 4 hidden layers for demonstration:
      BatchNorm(dim) -> Dense(dim->units) -> Dense(units->units)*3 -> Dense(units->output_units)
    """
    def __init__(self, dim, units, hidden_activation, output_units, output_activation=None, bias_last=False):
        super(DeepNNChain, self).__init__()

        self.dim = dim
        self.units = units
        self.output_units = output_units
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # BatchNorm over the channel dimension = dim
        self.bn_in = nn.BatchNorm1d(dim, eps=1e-6, momentum=0.99)

        # Hidden layers
        self.fc1 = nn.Linear(dim, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)

        # Last layer (no bias if bias_last=False, like Julia "bias=false")
        self.fc_out = nn.Linear(units, output_units, bias=not (bias_last is False))
        # If bias_last is False, we manually remove the bias parameter
        # if bias_last is False:
        #     with torch.no_grad():
        #         self.fc_out.bias.zero_()
        #     self.fc_out.requires_grad_(False)

    def forward(self, x):
        """
        x: shape [batch_size, dim]
        However, in your original Julia code, the dimension is [DIM, batch_size].
        We'll adapt by transposing in the caller if needed.
        """
        # If x is [batch_size, dim], PyTorch BN1d expects [batch_size, dim].
        # So that matches well as is.
        x = self.bn_in(x)
        x = self.hidden_activation(self.fc1(x))
        x = self.hidden_activation(self.fc2(x))
        x = self.hidden_activation(self.fc3(x))
        x = self.hidden_activation(self.fc4(x))

        out = self.fc_out(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out

# A container model that holds multiple z-nets plus one y-net.
class NonsharedModel(nn.Module):
    def __init__(self, deepnn_z_list, deepnn_y):
        """
        deepnn_z_list: a list of DeepNNChain for each interval
        deepnn_y: a single DeepNNChain for y
        """
        super(NonsharedModel, self).__init__()
        self.deepnn_z_list = nn.ModuleList(deepnn_z_list)
        self.deepnn_y = deepnn_y

    def forward(self, a_lowbound, dw, x, u, 
                MU, THETA, COST, SIGMA, ZETA, LAMBD, DELTA_T,
                NUM_TIME_INTERVAL):
        """
        Replicates the (m::NonsharedModel)(a_lowbound, dw, x, u) call in Julia.
        Returns: (y, negative_loss).
        The shape of x is [DIM, NUM_TIME_INTERVAL+1, batch_size].
        The shape of dw is [DIM, NUM_TIME_INTERVAL, batch_size].
        The shape of u is [DIM, batch_size].
        """
        # We will accumulate negative_loss
        negative_loss_val = torch.tensor(0.0, device=x.device)

        # y-net is only used at x[:,0,:], which is shape [DIM, batch_size].
        x0 = x[:, 0, :]  # [DIM, batch_size]
        # We want to feed to deepnn_y as [batch_size, DIM]
        x0_t = x0.t()    # [batch_size, DIM]
        y = self.deepnn_y(x0_t)   # shape [batch_size, 1]
        # z at t=1
        z = self.deepnn_z_list[0](x0_t)  # shape [batch_size, DIM]

        # Negative loss from y and z at time 1
        negative_loss_val += calculate_negative_loss(y.squeeze().t())  # y is [batch_size,1], we replicate Julia approach
        negative_loss_val += calculate_negative_loss(z.t())

        # Loop over time intervals
        for t in range(NUM_TIME_INTERVAL-1):
            # x[:,t,:] shape is [DIM, batch_size]
            x_t = x[:, t, :]
            dw_t = dw[:, t, :]  # [DIM, batch_size]
            # z shape [batch_size, DIM]

            # compute mx
            mx = compute_mx(x_t, a_lowbound)  # shape [batch_size]

            # replicate first_term = sum((MU - THETA).*z.*u, dims=1)
            # but now we have to do elementwise for each dimension i, then sum
            # z and u are [batch_size, DIM], so let's do z*u => [batch_size, DIM], (MU - THETA) => [DIM]
            # we want sum over DIM => dim=1 in PyTorch since z is [batch_size, DIM]
            MU_minus_THETA = (MU - THETA).unsqueeze(0)  # shape [1, DIM]
            first_term = torch.sum(MU_minus_THETA * z * u.t(), dim=1)  # shape [batch_size]
            
            # second_term = minimum(COST .+ (MU-THETA).*z, dims=1)
            # The shape is also [batch_size, DIM].
            # COST is shape [DIM], so expand => [1, DIM], (MU - THETA)*z => [batch_size, DIM]
            # We want elementwise addition, then min across DIM => dim=1
            cost_plus = COST.unsqueeze(0) + MU_minus_THETA * z
            second_term, _ = torch.min(cost_plus, dim=1)  # shape [batch_size]

            w = mx * (first_term - second_term)  # shape [batch_size]

            # Update y
            # y is shape [batch_size, 1], w is [batch_size], DELTA_T is scalar
            y = y + DELTA_T * w.unsqueeze(1) + torch.sum(SIGMA[:, t].unsqueeze(0) * z * dw_t.t(), dim=1, keepdim=True)

            # z at t+1
            x_next = x[:, t+1, :]  # [DIM, batch_size]
            z = self.deepnn_z_list[t+1](x_next.t())  # shape [batch_size, DIM]
            negative_loss_val += calculate_negative_loss(z.t())

        # final step
        x_end = x[:, -1, :]  # [DIM, batch_size]
        mx_end = compute_mx(x_end, a_lowbound)

        MU_minus_THETA = (MU - THETA).unsqueeze(0)
        first_term = torch.sum(MU_minus_THETA * z * u.t(), dim=1)
        cost_plus = COST.unsqueeze(0) + MU_minus_THETA * z
        second_term, _ = torch.min(cost_plus, dim=1)
        w = mx_end * (first_term - second_term)

        y = y + DELTA_T * w.unsqueeze(1) + torch.sum(SIGMA[:, -1].unsqueeze(0) * z * dw[:, -1, :].t(), dim=1, keepdim=True)

        return y, negative_loss_val


# -------------------------------------------------------------------
# 4) Data Sampling Functions
# -------------------------------------------------------------------
def generate_dw_sample(dim, num_time_interval, batch_size, SQRT_DELTA_T, device):
    # randn(DIM, NUM_TIME_INTERVAL, batch_size) * sqrt_delta_t
    # In PyTorch: shape [DIM, NUM_TIME_INTERVAL, batch_size]
    return torch.randn(dim, num_time_interval, batch_size, device=device) * SQRT_DELTA_T

def generate_u_sample(DIRICHLET_DIM, POLICY, batch_size, device):
    """
    Emulates the logic of 'generate_u_sample' in Julia.
    Returns shape [DIM, batch_size] in PyTorch so that each column = one sample.
    """
    DIM = DIRICHLET_DIM
    if POLICY == "even":
        # 1 / DIM for all
        arr = np.ones((DIM, batch_size), dtype=np.float32) * (1.0 / DIM)
        return torch.from_numpy(arr).to(device)
    elif POLICY == "random":
        # Dirichlet(ones(DIM)) for each sample
        # We can do this with np.random.gamma
        alpha = np.ones(DIM, dtype=np.float32)
        out = np.zeros((DIM, batch_size), dtype=np.float32)
        for i in range(batch_size):
            # each sample
            sample = np.random.gamma(alpha, 1.0)
            sample /= sample.sum()
            out[:, i] = sample
        return torch.from_numpy(out).to(device)
    elif POLICY == "minimal":
        # zeros
        return torch.zeros(DIM, batch_size, dtype=torch.float32, device=device)
    elif POLICY == "weighted_split":
        # For 3D example in your code:
        #  0.18, 0.18, 0.18 => sum=0.54, rest spread => (0.46 / 14) each
        # But we keep the logic exactly the same as in the code:
        # for a general DIM.  If your code is truly for 3D, adapt as needed.
        arr = np.ones((DIM, batch_size), dtype=np.float32) * (0.46 / 14.0)
        arr[0, :] = 0.18
        arr[1, :] = 0.18
        arr[2, :] = 0.18
        return torch.from_numpy(arr).to(device)
    elif POLICY == "best":
        # For 3D example => dimension #2 is 1
        arr = np.zeros((DIM, batch_size), dtype=np.float32)
        arr[1, :] = 1.0
        return torch.from_numpy(arr).to(device)
    elif POLICY == "best_var":
        # For 3D example => dimension #1 is 1
        arr = np.zeros((DIM, batch_size), dtype=np.float32)
        arr[0, :] = 1.0
        return torch.from_numpy(arr).to(device)
    else:
        # default
        return torch.zeros(DIM, batch_size, dtype=torch.float32, device=device)

def sample(dim, num_time_interval, batch_size,
           ZETA, MU, THETA, SIGMA,
           DELTA_T, POLICY, device):
    """
    Emulates the Julia `sample(num_sample)` function.
    Returns dw_sample, x_sample, u_sample
      dw_sample: [DIM, NUM_TIME_INTERVAL, batch_size]
      x_sample:  [DIM, NUM_TIME_INTERVAL+1, batch_size]
      u_sample:  [DIM, batch_size]
    """
    # Generate dw
    # We assume we have a global SQRT_DELTA_T = sqrt(DELTA_T)
    sqrt_delta_t = np.sqrt(DELTA_T).astype(np.float32)
    dw_sample = generate_dw_sample(dim, num_time_interval, batch_size, sqrt_delta_t, device)

    # Generate u
    u_sample = generate_u_sample(dim, POLICY, batch_size, device)

    # x_sample
    x_sample = torch.zeros(dim, num_time_interval + 1, batch_size, dtype=torch.float32, device=device)
    # Initialize x[:,0,:] as uniform(-10, 10)
    x_sample[:, 0, :] = (20.0 * torch.rand(dim, batch_size, device=device) - 10.0)

    # Fill in the Euler-like step:
    # x_{t+1} = x_t + (ZETA - MU*x_t)*DELTA_T + SIGMA*dW_t + (mx*((MU-THETA)*u))*DELTA_T
    for i in range(num_time_interval):
        x_t = x_sample[:, i, :]  # [DIM, batch_size]
        sum_x = torch.sum(x_t, dim=0)
        mx = torch.clamp(sum_x, min=0.0)
        # (MU - THETA)*u => shape [DIM, batch_size]
        mu_minus_theta_u = (MU - THETA).unsqueeze(1) * u_sample

        x_next = x_t + (ZETA[:, i].unsqueeze(1) - MU.unsqueeze(1) * x_t) * DELTA_T \
                  + SIGMA[:, i].unsqueeze(1) * dw_sample[:, i, :] \
                  + (mx.unsqueeze(0) * mu_minus_theta_u) * DELTA_T
        x_sample[:, i+1, :] = x_next

    return dw_sample, x_sample, u_sample


# -------------------------------------------------------------------
# 5) Loss Function
# -------------------------------------------------------------------
def loss_fn(model, a_lowbound, dw, x, u,
            OVERTIME_COST, LAMBDA, 
            MU, THETA, COST, SIGMA, ZETA, LAMBD, DELTA_T,
            training,
            NUM_TIME_INTERVAL):

    if training:
        model.train()
    else:
        model.eval()

    # Forward pass
    y_terminal, negative_loss_val = model(a_lowbound, dw, x, u,
                                          MU, THETA, COST, SIGMA, ZETA, LAMBD, 
                                          DELTA_T, NUM_TIME_INTERVAL)
    # y_terminal shape [batch_size, 1]

    # g_tf = OVERTIME_COST * max.(sum(x[:, end, :], dims=1), 0.0)
    sum_x_end = torch.sum(x[:, -1, :], dim=0)  # shape [batch_size]
    g_tf = OVERTIME_COST * torch.clamp(sum_x_end, min=0.0)

    # delta = y_terminal - g_tf
    delta = y_terminal.squeeze() - g_tf  # shape [batch_size]

    # MSE + LAMBDA * negative_loss
    loss_val = torch.mean(delta**2) + LAMBDA * negative_loss_val
    return loss_val


# -------------------------------------------------------------------
# 6) Main Training Loop (Example)
# -------------------------------------------------------------------
def main():
    # Set random seeds to replicate `Random.seed!(73)` from Julia
    np.random.seed(73)
    torch.manual_seed(73)

    # Load config
    dim = 17
    data_dir = f"tests/dynamic-scheduling/config_{dim}dim/"
    config = load_config(data_dir + f"config_{dim}dim.json")
    neural_network_params = config["neural_network_parameters"]
    system_params = config["system_parameters"]

    # Extract parameters from config
    LAMBDA = neural_network_params["LAMBDA"]
    MINS_IN_HOUR = 60
    NUMBER_HOUR = neural_network_params["HOURS"]
    PRECISION = neural_network_params["PRECISION"]
    TOTAL_TIME = neural_network_params["TOTAL_TIME"]
    DIM = neural_network_params["DIM"]
    NUM_TIME_INTERVAL = neural_network_params["NUM_TIME_INTERVAL"]
    PRINT_INTERVAL = neural_network_params["PRINT_INTERVAL"]
    NUM_ITERATIONS = neural_network_params["NUM_ITERATIONS"]
    NUM_NEURONS = neural_network_params["NUM_NEURONS"]
    COVAR_MULTIPLIER = neural_network_params["COVAR_MULTIPLIER"]
    BATCH_SIZE = neural_network_params["BATCH_SIZE"]
    VALID_SIZE = neural_network_params["VALID_SIZE"]

    LEARNING_RATES = neural_network_params["LEARNING_RATES"]  # e.g. [1e-2, 1e-3, ...]
    DECAY_STEPS = neural_network_params["DECAY_STEPS"]        # e.g. [1000, 2000, ...]

    # System parameters
    SCALING_FACTOR = MINS_IN_HOUR // PRECISION
    POLICY = system_params["POLICY"]
    OVERTIME_COST = system_params["OVERTIME_COST"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------
    # 2) Read CSV data and move to device
    # -------------------------------------------------------------------
    # Make sure to adapt paths / reading as in your environment
    MU = pd.read_csv(data_dir + system_params["MU_FILE"], header=None).values.astype(np.float32) / SCALING_FACTOR
    THETA = pd.read_csv(data_dir + system_params["THETA_FILE"], header=None).values.astype(np.float32) / SCALING_FACTOR
    COST = pd.read_csv(data_dir + system_params["COST_FILE"], header=None).values.astype(np.float32) / SCALING_FACTOR
    LAMBD = pd.read_csv(data_dir + system_params["LAMBD_FILE"], header=None).values.astype(np.float32) / SCALING_FACTOR
    ZETA = pd.read_csv(data_dir + system_params["ZETA_FILE"], header=None).values.astype(np.float32) / SCALING_FACTOR

    # Convert to torch and repeat
    MU = torch.from_numpy(MU.squeeze()).to(device)      # shape [DIM]
    THETA = torch.from_numpy(THETA.squeeze()).to(device)
    COST = torch.from_numpy(COST.squeeze()).to(device)
    LAMBD = torch.from_numpy(LAMBD.squeeze()).to(device)
    ZETA = torch.from_numpy(ZETA.squeeze()).to(device)

    # Repeat LAMBD, ZETA to match time intervals as in Julia
    # LAMBD = repeat(LAMBD, inner=[1, NUM_TIME_INTERVAL//TOTAL_TIME])
    # If LAMBD is shape [DIM], we have to see how many time points were in the CSV.
    # Often, your original code implies LAMBD is [DIM, TOTAL_TIME].
    # We replicate along axis=1 so that the total dimension becomes NUM_TIME_INTERVAL.
    # For demonstration, let's assume LAMBD is shape [DIM, TOTAL_TIME].
    # Then we do:
    LAMBD = LAMBD.view(DIM, -1)  # ensure shape [DIM, something]
    # LAMBD = LAMBD.repeat(1, NUM_TIME_INTERVAL // TOTAL_TIME)
    LAMBD = LAMBD.repeat_interleave(NUM_TIME_INTERVAL // TOTAL_TIME, dim=1)

    ZETA = ZETA.view(DIM, -1)
    # ZETA = ZETA.repeat(1, NUM_TIME_INTERVAL // TOTAL_TIME)
    ZETA = ZETA.repeat_interleave(NUM_TIME_INTERVAL // TOTAL_TIME, dim=1)

    # SIGMA = COVAR_MULTIPLIER .* sqrt.(2 * LAMBD)
    SIGMA = COVAR_MULTIPLIER * torch.sqrt(2.0 * LAMBD)

    DELTA_T = TOTAL_TIME / float(NUM_TIME_INTERVAL)
    # sqrt is used in sample() function.

    # -------------------------------------------------------------------
    # 3) Build Model(s)
    # -------------------------------------------------------------------
    # Create z networks (NUM_TIME_INTERVAL) + y network
    def create_deep_nn_chain(dim, units, output_units, output_activation=None, bias_last=False):
        # For your code: 4 hidden layers of size 'units' + final layer
        return DeepNNChain(
            dim=dim,
            units=units,
            hidden_activation=lambda x: leaky_relu_manual(x, slope=0.2),
            output_units=output_units,
            output_activation=output_activation,
            bias_last=bias_last
        )

    # A list of z networks
    DeepNN_z = [create_deep_nn_chain(DIM, NUM_NEURONS, DIM, output_activation=None, bias_last=False).to(device)
                for _ in range(NUM_TIME_INTERVAL)]
    # Single y network
    DeepNN_y = create_deep_nn_chain(DIM, NUM_NEURONS, 1, output_activation=None, bias_last=False).to(device)

    global_model = NonsharedModel(DeepNN_z, DeepNN_y).to(device)

    # -------------------------------------------------------------------
    # 4) Define Optimizer and (Manual) LR Schedule
    # -------------------------------------------------------------------
    # In Julia: ClipNorm(15), Adam(LEARNING_RATES[1], (0.9, 0.999), 1.0e-7)
    # We replicate with PyTorch:
    current_lr = LEARNING_RATES[0]
    optimizer = optim.Adam(global_model.parameters(),
                           lr=current_lr, betas=(0.9, 0.999), eps=1.0e-7)

    # -------------------------------------------------------------------
    # 5) Prepare validation data
    # -------------------------------------------------------------------
    dw_sample_valid, x_sample_valid, u_sample_valid = sample(
        DIM, NUM_TIME_INTERVAL, VALID_SIZE, 
        ZETA, MU, THETA, SIGMA, DELTA_T, POLICY, device
    )

    # -------------------------------------------------------------------
    # 6) Training Loop
    # -------------------------------------------------------------------
    training_history = np.zeros((NUM_ITERATIONS + 1, 4), dtype=np.float32)
    start_time = time.time()

    for step in range(NUM_ITERATIONS + 1):
        # Create new batch sample
        dw_sample, x_sample, u_sample = sample(
            DIM, NUM_TIME_INTERVAL, BATCH_SIZE,
            ZETA, MU, THETA, SIGMA, DELTA_T, POLICY, device
        )

        # a_lowbound = max(1 - step/3000, 0)
        a_lowbound = max(1.0 - step / 3000.0, 0.0)

        # Validation loss
        valid_loss = loss_fn(global_model, a_lowbound,
                             dw_sample_valid, x_sample_valid, u_sample_valid,
                             OVERTIME_COST, LAMBDA,
                             MU, THETA, COST, SIGMA, ZETA, LAMBD, DELTA_T,
                             training=False,
                             NUM_TIME_INTERVAL=NUM_TIME_INTERVAL).item()

        elapsed_time = time.time() - start_time

        # Adjust learning rate if needed
        if step in DECAY_STEPS:
            # Move to next LR from LEARNING_RATES
            idx = DECAY_STEPS.index(step)
            if idx + 1 < len(LEARNING_RATES):
                new_lr = LEARNING_RATES[idx + 1]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

        # Compute training loss and do backprop
        optimizer.zero_grad()
        train_loss_t = loss_fn(global_model, a_lowbound,
                               dw_sample, x_sample, u_sample,
                               OVERTIME_COST, LAMBDA,
                               MU, THETA, COST, SIGMA, ZETA, LAMBD, DELTA_T,
                               training=True,
                               NUM_TIME_INTERVAL=NUM_TIME_INTERVAL)
        train_loss_t.backward()

        # ClipNorm(15)
        torch.nn.utils.clip_grad_norm_(global_model.parameters(), 15.0)
        optimizer.step()

        train_loss = train_loss_t.item()

        # Print every PRINT_INTERVAL
        if step % PRINT_INTERVAL == 0:
            with torch.no_grad():
                y_init_val = global_model.deepnn_y(x_sample[:,:,0].transpose(0,1))
                y_init = y_init_val.cpu().numpy().mean()
            print(f"Step: {step}, Elapsed: {elapsed_time:.2f}s, "
                  f"Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, "
                  f"y_init: {y_init:.6f}")

        # Store history
        training_history[step] = [step, train_loss, valid_loss, elapsed_time]

    # -------------------------------------------------------------------
    # 7) Save Training History
    # -------------------------------------------------------------------
    run_name = "neural_network_policy_name"
    logs_dir = f"logs_{run_name}"
    if not os.path.isdir(logs_dir):
        os.mkdir(logs_dir)
    np.savetxt(os.path.join(logs_dir, "training_history.csv"),
               training_history, delimiter=",")

    # -------------------------------------------------------------------
    # 8) Save Model States
    # -------------------------------------------------------------------
    # In Julia, you do JLD2. In Python, we can do torch.save() for the state_dict.
    # We'll replicate the structure similarly.
    z_nets_cpu = [z_net.cpu().state_dict() for z_net in global_model.deepnn_z_list]
    y_net_cpu = global_model.deepnn_y.cpu().state_dict()

    # Just as an example, we can store each z_i state in separate .pt files
    for i in range(NUM_TIME_INTERVAL):
        torch.save(z_nets_cpu[i], os.path.join(logs_dir, f"z{i+1}.pt"))
    torch.save(y_net_cpu, os.path.join(logs_dir, "y.pt"))

    # Save optimizer state
    torch.save(optimizer.state_dict(), os.path.join(logs_dir, "final_optimizer.pt"))

    # -------------------------------------------------------------------
    # 9) Save Weights in .npy / Text Files for Potential C++ Ingestion
    # -------------------------------------------------------------------
    # This part mimics your approach of storing each layer’s weights & batchnorm.
    weights_dir = f"weights_{run_name}"
    if not os.path.isdir(weights_dir):
        os.mkdir(weights_dir)

    # Put model in eval mode
    global_model.eval()

    # Utility to get number of layers in a DeepNNChain
    def tuple_len(layers):
        return len(layers)

    # We replicate your loop for z_nets
    for i, z_net in enumerate(global_model.deepnn_z_list, start=1):
        layer_list = list(z_net.children())
        # layer_list[0] => BatchNorm1d
        # layer_list[1..4] => fc1..fc4
        # layer_list[5] => fc_out
        # Actually, we have 6 “children” if we count each fc as a separate child. 
        # We'll save them similarly.

        for j, layer in enumerate(layer_list, start=1):
            # Check if it's batchnorm
            if isinstance(layer, nn.BatchNorm1d):
                # layer.weight = gamma, layer.bias = beta
                # layer.running_mean = mu, layer.running_var = var
                gamma = layer.weight.data.cpu().numpy()
                beta = layer.bias.data.cpu().numpy()
                running_mean = layer.running_mean.data.cpu().numpy()
                running_var = layer.running_var.data.cpu().numpy()

                A = np.vstack([gamma, beta, running_mean, running_var]).T  # shape [dim, 4]
                np.save(os.path.join(weights_dir, f"z{i}_layer{j}.npy"), A)

            elif isinstance(layer, nn.Linear):
                # if layer has bias
                w = layer.weight.data.cpu().numpy()
                if layer.bias is not None:
                    b = layer.bias.data.cpu().numpy()
                    # shape them similarly
                    A = np.hstack([w.flatten(), b.flatten()])
                    # or store them in a 2D layout for clarity
                    A_2D = np.concatenate(
                        [w, b.reshape(1, -1)],
                        axis=0
                    )
                    np.save(os.path.join(weights_dir, f"z{i}_layer{j}.npy"), A_2D)
                else:
                    # no bias
                    np.save(os.path.join(weights_dir, f"z{i}_layer{j}.npy"), w)

    # Similarly for y_net
    layer_list = list(global_model.deepnn_y.children())
    for j, layer in enumerate(layer_list, start=1):
        if isinstance(layer, nn.BatchNorm1d):
            gamma = layer.weight.data.cpu().numpy()
            beta = layer.bias.data.cpu().numpy()
            running_mean = layer.running_mean.data.cpu().numpy()
            running_var = layer.running_var.data.cpu().numpy()

            A = np.vstack([gamma, beta, running_mean, running_var]).T
            np.save(os.path.join(weights_dir, f"y_layer{j}.npy"), A)
        elif isinstance(layer, nn.Linear):
            w = layer.weight.data.cpu().numpy()
            if layer.bias is not None:
                b = layer.bias.data.cpu().numpy()
                A_2D = np.concatenate([w, b.reshape(1, -1)], axis=0)
                np.save(os.path.join(weights_dir, f"y_layer{j}.npy"), A_2D)
            else:
                np.save(os.path.join(weights_dir, f"y_layer{j}.npy"), w)

    # -------------------------------------------------------------------
    # 10) Optionally Save Text Files for C++ If Desired
    # -------------------------------------------------------------------
    # The original Julia code writes out gamma, beta, mu, batchnorm_denom, etc. 
    # Here we show how to replicate that with np.savetxt, for example.
    cppweights_dir = f"cppweights_{run_name}"
    if not os.path.isdir(cppweights_dir):
        os.mkdir(cppweights_dir)

    # Example: writing out batchnorm parameters for each z
    for i, z_net in enumerate(global_model.deepnn_z_list, start=1):
        layer_list = list(z_net.children())
        bn_layer = layer_list[0]  # the BatchNorm1d
        if isinstance(bn_layer, nn.BatchNorm1d):
            gamma = bn_layer.weight.data.cpu().numpy()
            beta = bn_layer.bias.data.cpu().numpy()
            running_mean = bn_layer.running_mean.data.cpu().numpy()
            running_var = bn_layer.running_var.data.cpu().numpy()
            denom = np.sqrt(running_var + 1e-6)

            np.savetxt(os.path.join(cppweights_dir, f"z{i}_gamma.txt"), gamma[None, :], fmt="%.6f")
            np.savetxt(os.path.join(cppweights_dir, f"z{i}_beta.txt"), beta[None, :], fmt="%.6f")
            np.savetxt(os.path.join(cppweights_dir, f"z{i}_mu.txt"), running_mean[None, :], fmt="%.6f")
            np.savetxt(os.path.join(cppweights_dir, f"z{i}_batchnorm_denom.txt"), denom[None, :], fmt="%.6f")

        # Then save each Linear
        # layer_list[1] => fc1, layer_list[2] => fc2, ...
        # We mirror your Julia naming scheme: z{i}_w{(j-1)}.txt, etc.
        fc_counter = 0
        for j, layer in enumerate(layer_list[1:], start=1):
            if isinstance(layer, nn.Linear):
                w = layer.weight.data.cpu().numpy()
                if layer.bias is not None:
                    b = layer.bias.data.cpu().numpy()
                    np.savetxt(os.path.join(cppweights_dir, f"z{i}_w{fc_counter}.txt"), w.transpose(), fmt="%.6f")
                    np.savetxt(os.path.join(cppweights_dir, f"z{i}_b{fc_counter}.txt"), b[None, :], fmt="%.6f")
                else:
                    # no bias
                    np.savetxt(os.path.join(cppweights_dir, f"z{i}_w{fc_counter}.txt"), w.transpose(), fmt="%.6f")
                fc_counter += 1

    # Do the same for y_net
    layer_list = list(global_model.deepnn_y.children())
    bn_layer = layer_list[0]
    if isinstance(bn_layer, nn.BatchNorm1d):
        gamma = bn_layer.weight.data.cpu().numpy()
        beta = bn_layer.bias.data.cpu().numpy()
        running_mean = bn_layer.running_mean.data.cpu().numpy()
        running_var = bn_layer.running_var.data.cpu().numpy()
        denom = np.sqrt(running_var + 1e-6)

        np.savetxt(os.path.join(cppweights_dir, f"y_gamma.txt"), gamma[None, :], fmt="%.6f")
        np.savetxt(os.path.join(cppweights_dir, f"y_beta.txt"), beta[None, :], fmt="%.6f")
        np.savetxt(os.path.join(cppweights_dir, f"y_mu.txt"), running_mean[None, :], fmt="%.6f")
        np.savetxt(os.path.join(cppweights_dir, f"y_batchnorm_denom.txt"), denom[None, :], fmt="%.6f")

    fc_counter = 0
    for j, layer in enumerate(layer_list[1:], start=1):
        if isinstance(layer, nn.Linear):
            w = layer.weight.data.cpu().numpy()
            if layer.bias is not None:
                b = layer.bias.data.cpu().numpy()
                np.savetxt(os.path.join(cppweights_dir, f"y_w{fc_counter}.txt"), w.transpose(), fmt="%.6f")
                np.savetxt(os.path.join(cppweights_dir, f"y_b{fc_counter}.txt"), b[None, :], fmt="%.6f")
            else:
                np.savetxt(os.path.join(cppweights_dir, f"y_w{fc_counter}.txt"), w.transpose(), fmt="%.6f")
            fc_counter += 1

    print("Training complete. All logs and weights have been saved.")

# -------------------------------------------------------------------
# 7) Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
