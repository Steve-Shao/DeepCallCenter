#!/usr/bin/env python
# coding: utf-8
import logging
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os


dim = 17
data_dir = f"tests/dynamic-scheduling/config_{dim}dim/"

mu = pd.read_csv(data_dir + f"mu_hourly_{dim}dim.csv",header = None)[0].to_numpy() #hourly mu 
theta = pd.read_csv(data_dir + f"theta_hourly_{dim}dim.csv",header = None)[0].to_numpy() #hourly theta
cost = pd.read_csv(data_dir + f"cost_{dim}dim.csv", header = None)[0].to_numpy() #hourly cost
lambda_matrix = pd.read_csv(data_dir + f"lambd_matrix_hourly_{dim}dim.csv",header = None,  delimiter =",").to_numpy() #hourly limiting arrival rates
zeta_matrix = pd.read_csv(data_dir + f"zeta_matrix_hourly_{dim}dim.csv",header = None,  delimiter =",").to_numpy() #hourly second order term zeta

config = {
    "eqn_config": {
        "eqn_name": 'HJBLQ',
        "total_time": 17, 
        "dim": dim,
        "num_time_interval": 204,
        "theta": theta,
        "mu": mu,
        "cost": cost,
        "lambd": lambda_matrix,
        "zeta": zeta_matrix
    },
    "net_config": {
        "num_hiddens": [100,100,100,100],
        "lr_values": [1e-3, 0.0005, 1e-4],
        "lr_boundaries": [1200, 1500],
        "num_iterations": 2000,
        "batch_size": 512,
        "valid_size": 512,
        "logging_frequency": 10,
        "dtype": "float64",
        "verbose": True
    }
}

#####################################
# 1) Equation / PDE definitions
#####################################
class Equation(object):
    """Base class for PDE-related functions."""
    def __init__(self, eqn_config):
        self.dim = eqn_config["dim"]
        self.total_time = eqn_config["total_time"]
        self.num_time_interval = eqn_config["num_time_interval"]
        self.delta_t = self.total_time / self.num_time_interval 
        self.sqrt_delta_t = np.sqrt(self.delta_t)

        # Convert from numpy to torch
        self.zeta = torch.tensor(eqn_config["zeta"], dtype=torch.float64)
        self.lambd = torch.tensor(eqn_config["lambd"], dtype=torch.float64)
        self.mu = torch.tensor(eqn_config["mu"], dtype=torch.float64)
        self.theta = torch.tensor(eqn_config["theta"], dtype=torch.float64)
        self.cost = torch.tensor(eqn_config["cost"], dtype=torch.float64)
        self.y_init = None

    def sample(self, num_sample):
        """Return (dw, X) paths."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError


class HJBLQ(Equation):
    """Implements the specifics of the HJBLQ equation."""
    def __init__(self, eqn_config):
        super(HJBLQ, self).__init__(eqn_config)
        # Initialize x0 as uniform(0, 1)
        self.x_init = torch.tensor(
            np.random.uniform(low=0, high=1, size=[1]),
            dtype=torch.float64
        )
        # Initialize x0 as uniform(-10, 10)
        # self.x_init = torch.tensor(
        #     np.random.uniform(low=-10, high=10, size=[1]),
        #     dtype=torch.float64
        # )
        self.c_bar = 2.12
        self.sigma = torch.sqrt(self.lambd)

    def sample(self, num_sample):
        """Sample Brownian increments dw and paths x."""
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) \
                    * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init.item()

        for i in range(self.num_time_interval):
            for k in range(self.dim):
                # x_{t+1} = x_t + (zeta - mu*x_t)*dt + sqrt(2)*sigma * dw
                x_sample[:, k, i+1] = (
                    x_sample[:, k, i]
                    + (self.zeta[k, i].item() - self.mu[k].item()*x_sample[:, k, i]) * self.delta_t
                    + np.sqrt(2.0)*self.sigma[k, i].item() * dw_sample[:, k, i]
                )
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
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

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        # x shape: (batch, dim, ?)
        # we get x[:,:,-1], but we get it as (batch,dim).
        # Summation along dim => (batch,1)
        mx = torch.sum(x, dim=1, keepdim=True)
        mx = torch.clamp(mx, min=0.0).double()
        return self.c_bar * mx

#####################################
# 2) Feed-forward networks
#####################################
class FeedForwardSubnet(nn.Module):
    """Neural network for each sublayer (z)."""
    def __init__(self, config):
        super(FeedForwardSubnet, self).__init__()
        dim = config["eqn_config"]["dim"]
        num_hiddens = config["net_config"]["num_hiddens"]

        # BatchNorm1d wants 'num_features' = number of features in each layer
        # We only do minimal usage. In TF code, you had bn_layers but they
        # weren't all applied. We'll keep exactly one BN at input for minimal difference.
        # For more rigorous 1-1 mapping, you'd add BN after every Dense. 
        self.bn_input = nn.BatchNorm1d(dim, affine=True)  

        # Dense layers
        self.linears = nn.ModuleList()
        for hidden_size in num_hiddens:
            self.linears.append(nn.Linear(dim, hidden_size))
            dim = hidden_size
        # final layer to produce dimension = original PDE dimension
        self.linears.append(nn.Linear(dim, config["eqn_config"]["dim"]))

    def forward(self, x, training=True):
        # If you want to switch BN between train/eval:
        if training:
            self.train()
        else:
            self.eval()

        # x shape: (batch, dim)
        x = self.bn_input(x)
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            x = torch.relu(x)
        x = self.linears[-1](x)
        return x


class FeedForwardSubnet1(nn.Module):
    """Neural network for the initial V(0,.) scalar output."""
    def __init__(self, config):
        super(FeedForwardSubnet1, self).__init__()
        dim = config["eqn_config"]["dim"]
        num_hiddens = config["net_config"]["num_hiddens"]

        self.bn_input = nn.BatchNorm1d(dim, affine=True)

        self.linears = nn.ModuleList()
        for hidden_size in num_hiddens:
            self.linears.append(nn.Linear(dim, hidden_size))
            dim = hidden_size
        # final layer: scalar output
        self.linears.append(nn.Linear(dim, 1))

    def forward(self, x, training=True):
        if training:
            self.train()
        else:
            self.eval()

        x = self.bn_input(x)
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            x = torch.relu(x)
        x = self.linears[-1](x)
        return x

#####################################
# 3) Main model that combines subnets
#####################################
class NonsharedModel(nn.Module):
    def __init__(self, config, bsde):
        super(NonsharedModel, self).__init__()
        self.eqn_config = config["eqn_config"]
        self.net_config = config["net_config"]
        self.bsde = bsde

        # convert lambd => sigma
        self.lambd = torch.tensor(self.eqn_config["lambd"], dtype=torch.float64)
        self.sigma = np.sqrt(2.0) * torch.sqrt(self.lambd)  # shape: (dim, N)

        # subnetwork for each time step: z
        self.subnet_z = nn.ModuleList(
            [FeedForwardSubnet(config) for _ in range(self.bsde.num_time_interval)]
        )
        # single subnetwork for y(0,.)
        self.subnet_y = FeedForwardSubnet1(config)

    def forward(self, inputs, training=True):
        """Compute final Y_T given (dw, x) paths."""
        dw, x = inputs  # dw shape: (batch, dim, N), x shape: (batch, dim, N+1)
        # convert to torch
        dw = torch.tensor(dw, dtype=torch.float64)
        x  = torch.tensor(x, dtype=torch.float64)

        # y init
        y = self.subnet_y(x[:,:,0], training=training)
        # z init
        z = self.subnet_z[0](x[:,:,0], training=training)

        for t in range(0, self.bsde.num_time_interval - 1):
            # y_{t+1} = y_t - delta_t * f(t, x_t, y_t, z_t) + sum(z_t * sigma[:,t] * dw_t)
            f_val = self.bsde.f_tf(t, x[:,:,t], y, z)
            incr = torch.sum(z * (self.sigma[:,t] * dw[:,:,t]), dim=1, keepdim=True)
            y = y - self.bsde.delta_t * f_val + incr
            z = self.subnet_z[t + 1](x[:,:,t+1], training=training)

        # final step
        f_val = self.bsde.f_tf(self.bsde.num_time_interval - 1, x[:,:,-2], y, z)
        incr = torch.sum(z * dw[:,:,-1], dim=1, keepdim=True)
        y = y - self.bsde.delta_t * f_val + incr
        return y

#####################################
# 4) BSDE Solver
#####################################
class BSDESolver(object):
    def __init__(self, config, bsde):
        self.eqn_config = config["eqn_config"]
        self.net_config = config["net_config"]
        self.bsde = bsde

        self.model = NonsharedModel(config, bsde).double()

        # We replace TF piecewise-constant scheduler with a small custom function
        self.lr_boundaries = self.net_config["lr_boundaries"]
        self.lr_values     = self.net_config["lr_values"]

        # Start with the first LR
        self.current_lr = self.lr_values[0]

        # PyTorch Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.current_lr, eps=1e-8)

    def lr_schedule(self, step):
        """Mimic TF's PiecewiseConstantDecay with boundaries and values."""
        boundaries = self.lr_boundaries
        values = self.lr_values
        # Example: boundaries=[1200,1500], values=[1e-3,5e-4,1e-4]
        #   step <1200 => 1e-3
        #   1200<=step<1500 => 5e-4
        #   step>=1500 => 1e-4
        if step < boundaries[0]:
            return values[0]
        elif step < boundaries[1]:
            return values[1]
        else:
            return values[2]

    def train(self):
        start_time = time.time()
        training_history = []

        # Generate validation data
        valid_data = self.bsde.sample(self.net_config["valid_size"])
        valid_dw, valid_x = valid_data

        for step in range(self.net_config["num_iterations"] + 1):
            # Update LR if needed
            self.current_lr = self.lr_schedule(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.current_lr

            # Logging
            if step % self.net_config["logging_frequency"] == 0:
                with torch.no_grad():
                    loss_val = self.loss_fn(valid_data, training=False).item()
                    y_init_val = self.model.subnet_y(torch.tensor(valid_x[:,:,0], dtype=torch.float64), training=False)
                    y_init = y_init_val.data.cpu().numpy().mean()
                    elapsed_time = time.time() - start_time
                    print(step, loss_val, y_init, elapsed_time)

            # Train
            training_data = self.bsde.sample(self.net_config["batch_size"])
            train_loss = self.train_step(training_data)  # updates model
            valid_loss = self.loss_fn(valid_data, training=False).item()
            elapsed_time = time.time() - start_time

            training_history.append([step, valid_loss, train_loss, elapsed_time])

        return np.array(training_history)

    def loss_fn(self, inputs, training):
        dw, x = inputs
        # forward pass
        y_terminal = self.model((dw, x), training=training)
        # terminal loss
        # x[:,:,-1] => shape [batch, dim]
        x_terminal = torch.tensor(x, dtype=torch.float64)[:,:,-1]
        g_val = self.bsde.g_tf(self.bsde.total_time, x_terminal)
        delta = y_terminal - g_val
        loss = torch.mean(delta**2)
        return loss

    def train_step(self, train_data):
        self.optimizer.zero_grad()
        loss = self.loss_fn(train_data, training=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()

#####################################
# 5) Running everything
#####################################
class main():
    bsde = HJBLQ(config["eqn_config"])
    # set default dtype
    torch.set_default_dtype(torch.float64)

    logging.info('Begin to solve %s ' % config["eqn_config"]["eqn_name"])

    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()

    # Save training history
    np.savetxt('training_history.csv',
               training_history,
               fmt=['%d', '%.5e','%.5e', '%.3f'],
               delimiter=",",
               header='step,valid_loss,train_loss,elapsed_time',
               comments='')

    # Create directories for saving subnets
    string = 'modelsave{}'
    directory_list = []
    parent_dir = "/home/ekasikar/NeuralNetwork/2dimensional/outputs/"

    for i in range(config["eqn_config"]["num_time_interval"]):
        path = os.path.join(parent_dir, string.format(i))
        os.mkdir(path)
        directory_list.append(path)
        # Minimal approach for saving: PyTorch doesn't have .save() on Modules
        # the same way Keras does. We'll just do:
        torch.save(bsde_solver.model.subnet_z[i].state_dict(), os.path.join(path, "subnet_z.pt"))

# Plot the losses just like in TF
data = pd.read_csv("training_history.csv")
plt.scatter(data["step"], data["valid_loss"], color="blue", label="validation_loss")
plt.scatter(data["step"], data["train_loss"], color="black", label="training_loss")
plt.xticks(
    [0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000],
    rotation=0
)
plt.legend()
plt.ylabel('Training Loss vs. Validation Loss')
plt.xlabel('Number of iteration steps')
plt.title('Objective of Stochastic Gradient Descent')
plt.autoscale()
plt.gcf().set_size_inches(9, 6)
plt.savefig('loss.png')
# plt.show()
