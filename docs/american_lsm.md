# American options by Longstaff-Schwartz

Implementation: [`src/pricing/lsm.py`](../src/pricing/lsm.py)
Verification notebook: [`notebooks/10_american_lsm.ipynb`](../notebooks/10_american_lsm.ipynb)

## 1. The problem

An American option is an optimal stopping problem: exercise when the intrinsic value exceeds the value of waiting. A tree ([`binomial.md`](binomial.md)) or a PDE ([`pde.md`](pde.md)) solves it by backward induction on a grid, but both die with several risk factors or path dependence. Monte Carlo simulates forward in time, which is the wrong direction for this problem: at a given node you need the *conditional* expectation of the future, and a single path gives only one sample of it.

## 2. The idea

Longstaff and Schwartz (2001) estimate the continuation value $\mathbb{E}[\text{discounted future cash flow}\mid S_t]$ by **regression** across paths. Algorithm (put, exercise dates $t_1<\dots<t_n=T$):

1. Simulate all paths up to $T$. Set the cash flow at $t_n$ to the payoff.
2. Step back to $t_{j}$. Discount the cash flows by one step. Among the paths that are **in the money** at $t_j$, regress the discounted cash flows on a polynomial of $S_{t_j}/K$. The fitted value is the estimate of the continuation value.
3. Where intrinsic value exceeds the fitted continuation, exercise: replace the cash flow with the intrinsic value.
4. Repeat down to $t_1$, then average the discounted cash flows (and take the maximum with exercising at $t_0$).

Two details that matter. Only in-the-money paths enter the regression: out of the money you never exercise, and including those paths puts the kink of the payoff where the polynomial fits worst. And the regression is used only to decide **when** to exercise; the cash flow credited is the actual realised one, not the fitted value. That second point is what stops the estimator from being biased high.

## 3. Bias: why the price is a lower bound

The regression-based exercise rule is a valid exercise strategy, but not the optimal one. Any feasible strategy is worth at most the American price, so LSM is **low-biased** (up to Monte Carlo noise). Fewer exercise dates also bias it low, because a Bermudan is worth less than an American. A richer basis moves the price up, a finer date grid moves it up.

## 4. Worked example

American put, $S_0=K=100$, $T=1$, $r=5\%$, $\sigma=20\%$, $q=0$, 100k paths, cubic basis unless stated. References: tree with 2000 steps 6.089990, PDE (600x600) 6.089485, European put 5.573526.

| Exercise dates | Basis degree | LSM price | Std error |
|---|---|---|---|
| 10 | 2 | 6.0328 | 0.0233 |
| 10 | 3 | 6.0356 | 0.0230 |
| 10 | 5 | 6.0398 | 0.0229 |
| 50 | 2 | 6.0247 | 0.0225 |
| 50 | 3 | 6.0465 | 0.0225 |
| 50 | 5 | 6.0508 | 0.0224 |

Every estimate sits 0.04 to 0.065 under the tree price, which is roughly 2 to 3 standard errors below. That is consistent with a low-biased estimator, and for the 10-date rows part of the gap is the Bermudan-versus-American difference. The standard error does not decrease with a better basis; it is set by the number of paths. The effect of the basis degree (+0.007 with 10 dates and +0.026 with 50 dates, going from degree 2 to 5) is about the size of the standard error or smaller. All runs share the same seed, so these differences are less noisy than the standard error suggests, but I would not claim more than a small upward drift with the basis from one seed. All the estimates are well above the European price (5.5735), so the early-exercise premium of about 0.5 is captured.

## 5. Practical points

- Memory: the whole $n_{paths}\times n_{steps}$ array is stored; for 100k paths and 50 dates that is 5 million floats, fine, but it scales badly.
- The simulation is exact for GBM, so no discretisation error enters. For Heston or other models there is an Euler bias on top.
- To get an upper bound, a dual method (Andersen-Broadie) is needed; the low and high estimates then bracket the true price.
- LSM works for Greeks only through bump-and-reprice with common random numbers; pathwise differentiation through the exercise decision is delicate.

## Interview questions

**1. Why regress only on in-the-money paths?**
Exercise is only relevant when intrinsic value is positive, and out of the money the payoff is flat at zero, so it adds a kink the polynomial would have to fit. Restricting to ITM paths concentrates the regression where it matters.

**2. Why is the estimate biased low?**
The fitted rule is one admissible exercise policy; the true price is the supremum over policies. The estimate is the value of a suboptimal policy.

**3. What do you regress on, and why is the estimate unbiased for that policy?**
The discounted realised cash flows on functions of the state. Using the realised cash flow, not the fitted value, as the payoff means the sample average is an unbiased estimate of the value of the chosen policy, while using the fitted value would let the regression noise leak in (biased high).

**4. How do you choose the basis?**
Polynomials (or Laguerre) of low degree in $S/K$ are enough for a vanilla put. For multi-asset or path-dependent products add cross terms and the relevant path statistics; check price stability as the basis grows.

**5. Compare LSM, a tree and a PDE for an American put on one stock.**
PDE and tree are more accurate and faster in one dimension. LSM's advantage is dimension: cost scales linearly in the number of factors.

**6. Why does an American call on a non-dividend stock need no LSM?**
It is never exercised early, so its price equals the European one.
