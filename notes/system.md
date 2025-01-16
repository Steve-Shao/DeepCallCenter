
## Continuous-time Markov Decision Process

**Primitives / Data**: 
- Finite planning horizon $[0, T]$
- A single pool of homogeneous agents answering calls
    - Number of agents at time $t$: $N(t)$
    - Overtime pay (after time $T$) to serve callers: $\bar{c}$
- $K$ classes of callers
    - Poisson arrival rate $\lambda_k(t): [0, T] \to \mathbb{R}$ (time inhomogeneous)
    - Callers leave the system by receiving service or abandoning the queue
        - Exponential service rate with mean $1/\mu_k$
        - Exponential abandonment rate with mean $1/\theta_k$
    - Cost rate per caller: $c_k = h_k + \theta_k p_k$
        - Holding cost per unit time $h_k$
        - Abandonment cost per caller $p_k$

**System state**: $X(t) = (X_1(t), \ldots, X_K(t))$
- $X_k(t)$: number of waiting class $k$ callers at time $t$

**Action / Control**: $\psi(t) = (\psi_1(t), \ldots, \psi_K(t))$
- $\psi_k(t)$: number of class $k$ callers being served at time $t$
- Action space given state $X(t) = x$: 
$$
\mathcal{A}(t,x) = \left\{ a \in \mathbb{R}_+^K: a \le x, \ \mathbb{1}^\top a = \min[\mathbb{1}^\top x, N(t)] \right\}
$$

**Objective**: Minimize total cost: 
$$
J(t, x, \psi) = \mathbb{E}_x^{\psi} \left[ \int_t^T c^\top (X(s) - \psi(s)) \, \mathrm{d}s + g(X(T)) \right]
$$
where $g$ is the terminal cost function: 
$$
g(x) = \bar{c} \, (\mathbb{1}^\top x - N(t))^+
$$

## Diffusion Process at the (Heavy-traffic) Limit 

Heavy-traffic assumption: the system is balanced and high-volume: 
$$
\sum_{k=1}^K \dfrac{\lambda_k(t)}{\mu_k} = N(t), \ t \in [0, T]
$$


