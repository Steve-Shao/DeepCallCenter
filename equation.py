import numpy as np
import torch

from torchbsde.torchbsde.equation.base import Equation


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

        # Get policy and "engineering tricks" parameters
        self.policy = equation_config.get("policy", "minimal")
        self.x_rand_radius = equation_config.get("x_rand_radius", 0.0)
        self.smoothing_prep_steps = equation_config.get("smoothing_prep_steps", 0)
        
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
            np.random.uniform(low=-self.x_rand_radius, high=self.x_rand_radius, size=[1]),
            dtype=self.dtype,
            device=self.device
        )
        self.c_bar = 2.12  # Terminal cost coefficient
        self.sigma = torch.sqrt(2.0 * self.lambd)  # Volatility term

    def _get_policy(self, num_sample):
        """
        Get policy of shape (num_sample, dim, num_time_interval).
        """
        u = torch.zeros(num_sample, self.dim, self.num_time_interval, device=self.device, dtype=self.dtype)
        if self.policy == "minimal":
            pass
        elif self.policy == "even":
            u = torch.ones(num_sample, self.dim, self.num_time_interval, device=self.device, dtype=self.dtype) / self.dim
        elif self.policy == "random":
            # Generate gamma samples directly on device using exponential distribution
            alpha = torch.ones(num_sample, self.dim, self.num_time_interval, device=self.device, dtype=self.dtype)
            # Gamma(1,1) is equivalent to Exponential(1)
            gamma_samples = torch.distributions.Exponential(alpha).sample()
            # Normalize to get Dirichlet samples
            u = gamma_samples / gamma_samples.sum(dim=1, keepdim=True)
        elif self.policy == "weighted_split":
            u = torch.ones(num_sample, self.dim, self.num_time_interval, device=self.device, dtype=self.dtype) * (0.46 / 14.0)
            u[:, 0] = 0.18
            u[:, 1] = 0.18
            u[:, 2] = 0.18
        elif self.policy == "best":
            # For 3D example => dimension #2 is 1
            u = torch.zeros(num_sample, self.dim, self.num_time_interval, device=self.device, dtype=self.dtype)
            u[:, 1] = 1.0
        elif self.policy == "best_var":
            # For 3D example => dimension #1 is 1
            u = torch.zeros(num_sample, self.dim, self.num_time_interval, device=self.device, dtype=self.dtype)
            u[:, 0] = 1.0
        else:
            raise ValueError(f"Unknown policy: {self.policy}")
        return u

    def sample(self, num_sample):
        """
        Sample paths for the stochastic process using PyTorch tensor operations.
        
        Args:
            num_sample: Number of sample paths to generate
            
        Returns:
            tuple: (dw_sample, x_sample) containing Brownian increments and state paths
        """
        with torch.no_grad():
            # Generate Brownian increments
            dw_sample = torch.randn(
                num_sample, 
                self.dim, 
                self.num_time_interval, 
                device=self.device, 
                dtype=self.dtype
            ) * self.sqrt_delta_t

            # Initialize state paths
            x_sample = torch.zeros(
                num_sample, 
                self.dim, 
                self.num_time_interval + 1, 
                device=self.device, 
                dtype=self.dtype
            )
            x_sample[:, :, 0] = self.x_init
            u_sample = self._get_policy(num_sample)

            # Iterate over each time interval
            for i in range(self.num_time_interval):
                mx = torch.sum(x_sample[:, :, i], dim=1)
                mx = torch.clamp(mx, min=0.0)
                mu_minus_theta_u = (self.mu - self.theta).unsqueeze(0) * u_sample[:, :, i]
                # Update state based on the dynamic scheduling equation
                x_sample[:, :, i + 1] = (
                    x_sample[:, :, i]
                    + (self.zeta[:, i].unsqueeze(0) - self.mu.unsqueeze(0) * x_sample[:, :, i]) * self.delta_t
                    # + torch.sqrt(torch.tensor(2.0, dtype=self.dtype, device=self.device)) * self.sigma[:, i].unsqueeze(0) * dw_sample[:, :, i]
                    + self.sigma[:, i].unsqueeze(0) * dw_sample[:, :, i]
                    + (mx.unsqueeze(1) * mu_minus_theta_u) * self.delta_t
                )

            return dw_sample, x_sample, u_sample

    def f_torch(self, t, x, y, z, u, step):
        """Generator function: f(t,x,y,z)."""

        # # x shape: (batch, dim)
        # # sum along dim => (batch,)
        # mx = torch.sum(x, dim=1)
        # mx = torch.clamp(mx, min=0.0)  # tf.maximum(..., 0)
        # mx = mx.view(-1, 1).double()   # reshape to (batch,1)
        
        # # cost + (mu - theta)*z => shape (batch, dim)
        # # reduce_min => shape (batch,)
        # # multiply by mx => shape (batch,1)
        # # We'll broadcast over dim.
        # # cost is shape (dim,); z is shape (batch, dim)
        # cost_broadcast = self.cost - self.theta + self.mu  # shape (dim,)
        # # Actually we want cost + (mu - theta), so:
        # cost_broadcast = self.cost + (self.mu - self.theta)
        # # Then cost_broadcast * z => shape (batch, dim)
        # term = cost_broadcast * z
        # # reduce_min across dim => shape (batch,)
        # out = torch.min(self.cost + (self.mu - self.theta) * z, dim=1).values
        # # but let's do it carefully:
        # #   cost + (mu - theta) is shape (dim,)
        # #   z is shape (batch, dim)
        # # so we want to do the per-sample min across dim
        # g = torch.min(cost_broadcast + 0.0*z, dim=0)  # but that wouldn't be right.
        # # Instead do:
        # tmp = self.cost.view(1, -1) + (self.mu - self.theta).view(1, -1) * z
        # # shape => (batch, dim)
        # min_val = torch.min(tmp, dim=1, keepdim=True)[0]  # => (batch,1)
        
        # return mx * min_val

        alpha = max(1.0 - (step + 1.0) / (self.smoothing_prep_steps + 1.0), 0.0)
        mx = torch.sum(x, dim=1)

        # Three methods: non-smoothing, elu smoothing, leaky_relu smoothing
        # mx = torch.clamp(mx, min=0.0)  # shape (batch,)
        mx = torch.nn.functional.elu(mx, alpha = alpha)  # shape (batch,)
        # mx = torch.nn.functional.leaky_relu(mx, negative_slope = alpha)  # shape (batch,)
        # mx = torch.nn.functional.leaky_relu(mx, negative_slope = alpha * 0.1)  # shape (batch,)

        mu_theta_z = self.mu - self.theta                          # shape (dim,)
        mu_theta_z = mu_theta_z.unsqueeze(0) * z                   # shape (batch, dim)
        min_class_cost = self.cost.unsqueeze(0) + mu_theta_z       # shape (batch, dim)
        min_class_cost = torch.min(min_class_cost, dim=1).values   # shape (batch,)

        # reverted by -1 to match han et al.'s implementation
        return mx.unsqueeze(1) * (min_class_cost - torch.sum(mu_theta_z * u, dim=1))

    def g_torch(self, t, x, step):
        """
        Terminal condition g(t,x) for the BSDE.
        
        Computes final cost based on terminal state.
        """
        mx = torch.sum(x, dim=1, keepdim=True)
        mx = torch.clamp(mx, min=0.0).to(dtype=self.dtype, device=self.device)
        return self.c_bar * mx
