# Binomial trees (Cox-Ross-Rubinstein)

Implementation: [`src/pricing/binomial.py`](../src/pricing/binomial.py)
Verification notebook: [`notebooks/05_binomial.ipynb`](../notebooks/05_binomial.ipynb)

## 1. The model

Split $[0,T]$ into $n$ steps of length $\Delta t = T/n$. In each step the spot is multiplied by $u$ or $d$. CRR picks

$$u = e^{\sigma\sqrt{\Delta t}}, \qquad d = 1/u, \qquad p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$

$p$ is the **risk-neutral** up probability: it is chosen so that the expected spot after one step is the forward, $\mathbb{E}^{\mathbb{Q}}[S_{t+\Delta t}] = S_t e^{(r-q)\Delta t}$. The real-world probability never appears. Because $ud=1$ the tree recombines: after $n$ steps there are $n+1$ nodes, not $2^n$, so backward induction is $O(n^2)$.

Pricing is backward induction from the payoff: $V = e^{-r\Delta t}\left[pV_u + (1-p)V_d\right]$. The code does this one time step at a time with vectorised numpy, not node by node.

## 2. Convergence to Black-Scholes

As $n\to\infty$, the binomial distribution of $\ln S_T$ converges to a normal with the GBM mean and variance (CLT), so the tree price converges to Black-Scholes. Reference: $S_0=K=100$, $T=1$, $r=5\%$, $\sigma=20\%$, $q=2\%$, European call, BS $= 9.227006$.

| $n$ | Tree price | Error |
|---|---|---|
| 10 | 9.035326 | -0.191680 |
| 11 | 9.396402 | +0.169397 |
| 50 | 9.188225 | -0.038781 |
| 51 | 9.263285 | +0.036280 |
| 100 | 9.207590 | -0.019416 |
| 101 | 9.245305 | +0.018299 |
| 500 | 9.223118 | -0.003887 |
| 1000 | 9.225062 | -0.001944 |

## 3. An honest finding: the error oscillates

The convergence is $O(1/n)$ (error at $n=500$ is 0.0039, at $n=1000$ it is 0.0019) but it is **not monotone**: the sign flips between consecutive $n$ (see $n=10$ vs $11$, $50$ vs $51$). With $S_0=K$ the strike sits exactly on a node for even $n$ and halfway between nodes for odd $n$, and the payoff kink is sampled differently. This is why a convergence plot of a tree against $n$ is a sawtooth, and why people average the prices for $n$ and $n+1$ or use Richardson extrapolation, or tune the tree so the strike lies on a node (Leisen-Reimer). It also means that reporting "converged" from two consecutive values of $n$ can be misleading.

## 4. American options

At each node take the maximum of the continuation value and the exercise value. This is the dynamic programming principle and needs no new theory: early exercise is just an optimal stopping problem solved on the tree.

With the same parameters and $n=1000$:

| | European | American |
|---|---|---|
| Put, $q=2\%$ | 6.328137 | 6.659759 |
| Put, $q=0$ | 5.571527 | 6.089595 |
| Call, $q=2\%$ | 9.225062 | 9.225062 |
| Call, $q=0$ | 10.448584 | 10.448584 |

- The put always carries an early-exercise premium (here 0.33 and 0.52): exercising early converts the strike into cash that earns interest.
- At the money the American call equals the European call (the two differ by 3e-8 at $q=2\%$), and at $q=0$ they are identical at any spot. A premium appears only when the call is deep in the money and the dividend yield is positive. With $n=500$, $K=100$, $q=2\%$, the American-minus-European gap is 0.000263 at $S=150$, 1.063 at $S=300$ and 5.024 at $S=500$. The intuition is that holding a deep ITM call costs the dividend income $qS$ and gains the interest on the strike $rK$ and some time value; once $qS$ clearly exceeds $rK$ ($6 > 5$ at $S=300$), exercising wins.

## 5. Limits

The tree handles American exercise naturally but scales badly with the number of underlyings or with path-dependence (a barrier needs the running extremum, a non-recombining state). Past one factor the choice is Monte Carlo with regression ([`american_lsm.md`](american_lsm.md)); in one factor, [finite differences](pde.md) give a more accurate result for the same cost.

## Interview questions

**1. Why is the tree probability $p$ not the real-world probability?**
It is chosen to make the discounted spot a martingale. Replication in a one-step tree fixes the price uniquely whatever the real-world $p$, so the real-world probability is irrelevant.

**2. Why does the tree recombine?**
Choosing $d = 1/u$ makes an up-down path land on the same node as a down-up path, so nodes at step $k$ number $k+1$ instead of $2^k$.

**3. When is $p$ outside $[0,1]$?**
When $e^{(r-q)\Delta t} > u$, i.e. large drift and small $\sigma\sqrt{\Delta t}$. It gives a negative probability, which signals an arbitrage in the tree; refining $n$ fixes it because $u-1\sim\sigma\sqrt{\Delta t}$ dominates $(r-q)\Delta t$.

**4. Why is an American call on a non-dividend stock never exercised early?**
Exercising gives $S-K$, but $C \ge S - Ke^{-rT} > S - K$ when $r>0$, so selling the option beats exercising. With dividends the bound becomes $S e^{-qT} - Ke^{-rT}$, which can fall below $S-K$ for deep ITM calls, so early exercise can pay off there.

**5. Why is the tree price not monotone in $n$?**
The strike's position relative to the final nodes changes with $n$; the payoff kink is integrated with a different discretisation error each time.

**6. How would you get Greeks from the tree?**
Delta and gamma from the first two layers of nodes (no bumping needed); theta from the middle node two steps ahead; vega and rho by bump-and-reprice.
