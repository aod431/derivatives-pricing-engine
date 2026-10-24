# Greeks by Monte Carlo (and quasi-Monte Carlo)

Implementation: [`src/pricing/mc_greeks.py`](../src/pricing/mc_greeks.py), [`src/pricing/quasi_mc.py`](../src/pricing/quasi_mc.py)
Related: [`greeks.md`](greeks.md) for the closed-form Greeks, [`monte_carlo.md`](monte_carlo.md) for the pricer.

## 1. The problem

A Monte Carlo price is $\mathbb{E}[e^{-rT}f(S_T)]$. A Greek is the derivative of that expectation with respect to a parameter $\theta$. There are three ways to differentiate, depending on where you push the derivative.

**Pathwise.** Move $\partial_\theta$ inside the expectation and differentiate the payoff along each path:

$$\frac{\partial}{\partial S}\mathbb{E}[e^{-rT}(S_T-K)^+] = e^{-rT}\,\mathbb{E}\!\left[\mathbf{1}_{S_T>K}\,\frac{S_T}{S}\right], \qquad \mathrm{Vega}=e^{-rT}\,\mathbb{E}\!\left[\mathbf{1}_{S_T>K}\,S_T(\sqrt T Z-\sigma T)\right]$$

using $S_T=S\exp((r-q-\tfrac12\sigma^2)T+\sigma\sqrt TZ)$. Unbiased and low variance, but it needs a payoff that is differentiable almost everywhere. For a **digital**, $\mathbf 1_{S_T>K}$ has derivative zero a.e. and pathwise gives 0, which is wrong. The code raises an error rather than return that.

**Likelihood ratio.** Leave the payoff alone and differentiate the density of $S_T$ instead. For GBM the score is $Z/(S\sigma\sqrt T)$, so

$$\Delta = e^{-rT}\,\mathbb{E}\!\left[f(S_T)\,\frac{Z}{S\sigma\sqrt T}\right]$$

It works for any payoff, including discontinuous ones, but multiplies by a random weight with large variance, and it needs the density to be known.

**Bump and reprice.** Central difference $[\hat V(S+h)-\hat V(S-h)]/2h$. Model-free, but sensitive to how the random numbers are used.

## 2. Worked example: call delta

$S_0=K=100$, $T=1$, $r=5\%$, $\sigma=20\%$, $q=0$, 400k paths. Exact delta (closed form): 0.636831.

| Method | Estimate | Std error |
|---|---|---|
| Pathwise | 0.63637 | 0.00091 |
| Likelihood ratio | 0.63881 | 0.00234 |
| Bump (1%), common random numbers | 0.63636 | 0.00090 |
| Bump (1%), independent draws | 0.64500 | 0.01645 |

All are consistent with the exact value. Likelihood ratio has about 2.6 times the standard error of pathwise on a smooth payoff, which is the price of its generality. Independent draws are 18 times worse than common random numbers.

## 3. Why common random numbers matter

With independent draws the two prices have independent errors, so the difference has variance of order $\sigma_V^2/h^2$: **it explodes as the bump shrinks**. With the same $Z$ the errors cancel. Standard error of the delta, by bump size, 400k paths:

| Bump $h/S$ | Independent draws | Common random numbers |
|---|---|---|
| 0.1 | 0.0017 | 0.00083 |
| 0.01 | 0.0165 | 0.00090 |
| 0.001 | 0.1644 | 0.00091 |

The independent-draw error grows exactly as $1/h$, while with common numbers it stays flat. The usual advice follows: always bump with common random numbers and a bump that is not so small that rounding matters.

## 4. A case where only likelihood ratio works: the digital

Digital call $\mathbf 1_{S_T>K}$, same parameters, 800k paths. Exact delta: 0.018762.

| Method | Estimate | Std error |
|---|---|---|
| Likelihood ratio | 0.01875 | 0.00003 |
| Bump 1% (common numbers) | 0.01881 | 0.00010 |
| Bump 0.1% (common numbers) | 0.01922 | 0.00034 |
| Pathwise | not valid (raises) | |

Likelihood ratio is the best here. For the bump, even with common numbers, the indicator makes the finite difference a rare-event estimate: a smaller bump reduces the bias but the standard error goes up (0.0001 to 0.00034) because fewer paths fall in the narrow band between the two strikes.

## 5. Quasi-Monte Carlo

The Monte Carlo rate is $N^{-1/2}$ regardless of the integrand. Quasi-Monte Carlo replaces random points by a low-discrepancy sequence (here scrambled **Sobol**), which fills the unit cube more evenly, and for smooth, low-dimensional integrands the error decays close to $1/N$. A deterministic sequence has no error bar, so `qmc_price` uses **randomised QMC**: 16 independent scrambles, each with $2^m$ points, give 16 unbiased estimates whose spread is the standard error.

European call, $S_0=K=100$, $T=1$, $r=5\%$, $\sigma=20\%$, $q=0$, BS $=10.450584$. Total points are 16 replicates of $2^m$ points each, $m=8,10,12,14$:

| Total points | QMC error | QMC std error | MC error | MC std error |
|---|---|---|---|---|
| 4,096 | -0.026566 | 0.007191 | -0.29016 | 0.22675 |
| 16,384 | -0.001366 | 0.002605 | -0.09810 | 0.11567 |
| 65,536 | -0.000694 | 0.000472 | -0.00558 | 0.05779 |
| 262,144 | -0.000089 | 0.000150 | 0.00371 | 0.02888 |

Over a 64x increase in points the QMC standard error falls by a factor of about 48 and the MC one by about 8 ($=\sqrt{64}$). QMC is therefore close to $1/N$ here, and at 262k points its standard error is about 190 times smaller than MC's, which in MC terms is worth roughly $190^2\approx 37{,}000$ times more paths.

Caveat: this is a best case. One dimension means the Sobol points are used perfectly. For a path-dependent payoff with 250 time steps QMC loses much of its advantage unless the path is built with a Brownian bridge or PCA so that the first coordinates carry most of the variance.

## Interview questions

**1. Why does the pathwise method fail for a digital option?**
The payoff derivative is zero except at the jump, which has probability zero, so the estimator returns 0 while the true delta is positive. The derivative and expectation cannot be swapped because the integrand is not Lipschitz.

**2. When would you choose likelihood ratio over pathwise?**
For discontinuous payoffs (digitals, barriers) and when the payoff is a black box but the density of the underlying is known.

**3. Why do common random numbers help bump-and-reprice?**
$\mathrm{Var}(\hat V_+-\hat V_-)=\mathrm{Var}(\hat V_+)+\mathrm{Var}(\hat V_-)-2\mathrm{Cov}$. Same draws make the covariance close to the variance, so the difference is small and its variance scales like $O(1)$ rather than $O(1/h^2)$ after dividing by $2h$.

**4. What does adjoint algorithmic differentiation add?**
All first-order sensitivities for a cost that is a small multiple of one pricing, independent of the number of parameters. Pathwise is the same idea applied by hand to one path.

**5. Why does QMC need randomisation?**
To get an error estimate and an unbiased result. Deterministic Sobol points give a fixed number with no standard error and a worst-case bound that is hard to evaluate.

**6. Why does QMC lose efficiency in high dimension?**
The $1/N$ rate comes with a $(\log N)^d$ constant, and the low-discrepancy property is poor in the later coordinates. Reordering the dimensions so the important ones come first restores most of the gain.
