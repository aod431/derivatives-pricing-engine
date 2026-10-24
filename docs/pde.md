# Finite-difference pricing (Crank-Nicolson)

Implementation: [`src/pricing/pde.py`](../src/pricing/pde.py)
Verification notebook: [`notebooks/09_pde_tree_mc_comparison.ipynb`](../notebooks/09_pde_tree_mc_comparison.ipynb)

## 1. The equation

From [`foundations.md`](foundations.md), $V(t,S)$ solves the Black-Scholes PDE. Change variables to time-to-maturity $\tau=T-t$ and $x=\ln S$ so the coefficients become constant:

$$V_\tau = \tfrac12\sigma^2 V_{xx} + \mu V_x - rV, \qquad \mu = r-q-\tfrac12\sigma^2, \qquad V(0,x)=\text{payoff}(e^x)$$

The grid is uniform in $x$, centred at the expected log-spot, $\pm 6$ standard deviations wide by default. The payoff is known at $\tau=0$ and the solver marches forward in $\tau$ to maturity.

## 2. The scheme

Space is discretised with central differences, giving a tridiagonal operator $L$. For the time step:

$$(I-\theta\,\Delta\tau\,L)V^{n+1} = (I+(1-\theta)\Delta\tau\,L)V^{n}$$

- $\theta=1$: fully implicit. Unconditionally stable, first order in time, heavily damping.
- $\theta=\tfrac12$: **Crank-Nicolson**. Unconditionally stable and second order in time and space.

Each step solves a tridiagonal system in $O(N)$ with `solve_banded`, so the whole price is $O(N_xN_t)$.

**Rannacher smoothing.** The payoff has a kink at the strike, and Crank-Nicolson does not damp high frequencies, so the kink produces spurious oscillations in the Greeks (and some in the price). The first two time steps use $\theta=1$, which damps them, and then CN takes over. The cost is negligible.

**Boundaries.** At $x$ small and large the code imposes the asymptotic value: a call is $0$ at the bottom and $Se^{-q\tau}-Ke^{-r\tau}$ at the top; a put is $Ke^{-r\tau}-Se^{-q\tau}$ at the bottom and $0$ at the top. The boundary values enter the right-hand side of the first and last interior equations. With a $\pm6\sigma\sqrt T$ grid the boundary error is negligible.

**American exercise.** After each time step take $V\leftarrow\max(V,\text{payoff})$. This projection is easy and accurate near first order around the free boundary; a proper treatment solves the linear complementarity problem (PSOR or Brennan-Schwartz).

## 3. Convergence

European call, $S_0=K=100$, $T=1$, $r=5\%$, $\sigma=20\%$, $q=0$, BS $=10.450584$, with $N_x=N_t=N$:

| $N$ | PDE price | Abs. error | Time (s) |
|---|---|---|---|
| 50 | 10.516983 | 0.066400 | 0.001 |
| 100 | 10.461045 | 0.010462 | 0.001 |
| 200 | 10.455193 | 0.004610 | 0.003 |
| 400 | 10.449760 | 0.000823 | 0.007 |
| 800 | 10.450378 | 0.000206 | 0.020 |

From 400 to 800 the error drops by a factor of 4, as second-order convergence predicts; at coarse grids the pattern is noisier because the strike falls between nodes in a different place each time (the same effect as in the binomial tree). Compared with the other methods in this repository, a price within $10^{-3}$ takes a few milliseconds here, against a Monte Carlo that would need about 200 million paths to reach a standard error of $10^{-3}$ (the standard error is 0.044 at 100k paths, from [`monte_carlo.md`](monte_carlo.md), and falls as $1/\sqrt N$). For the American put with $N=600$ the PDE gives 6.089485 against 6.089990 from a 2000-step tree.

## 4. When to use it, when not to

PDE solvers are the best tool in one or two dimensions, including early exercise, barriers (zero boundary) and local volatility, where $\sigma(S,t)$ simply enters the coefficients. They are also smooth: Greeks come straight from the grid. The cost grows exponentially with dimension (the "curse of dimensionality"), so beyond about three factors Monte Carlo is the choice.

## Interview questions

**1. Why Crank-Nicolson instead of explicit Euler?**
Explicit schemes need $\Delta\tau\le\Delta x^2/\sigma^2$ for stability, so refining space forces many more time steps. CN is unconditionally stable and second order in time.

**2. Why do you get oscillations with CN and how does Rannacher fix them?**
CN has amplification factor tending to $-1$ for high frequencies, so the discontinuous second derivative of the payoff is not damped. A couple of implicit steps (factor tending to 0) kill those modes first.

**3. Why work in $\ln S$?**
Coefficients become constant, so the uniform grid gives constant matrix entries and a uniform relative resolution around the strike.

**4. How do you impose boundary conditions?**
Dirichlet with the asymptotic payoff value, or a linearity condition $V_{SS}=0$. The domain must be wide enough that the truncation is irrelevant.

**5. How would you add early exercise exactly?**
Solve the linear complementarity problem $\min(V-\text{payoff},\,-\mathcal{L}V)=0$ at each step with PSOR or a penalty method, rather than the plain projection.

**6. What changes for local volatility?**
$\sigma$ becomes $\sigma(S,t)$ on the grid, so the diagonals vary with node and time; the algorithm is otherwise identical.
