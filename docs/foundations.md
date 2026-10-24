# Foundations: stochastic calculus and risk-neutral pricing

These are the results every other document in this folder relies on. Nothing here is implemented as code; it is the reasoning behind [`black_scholes.py`](../src/pricing/black_scholes.py) and the Monte Carlo, PDE and tree pricers.

## 1. Brownian motion and Ito's lemma

A Brownian motion $W_t$ has $W_0=0$, independent increments, and $W_t - W_s \sim \mathcal{N}(0, t-s)$. Its paths are continuous but nowhere differentiable, and its quadratic variation is $[W]_t = t$. In differential notation this is the rule $dW^2 = dt$, with $dt\,dW = dt^2 = 0$.

That rule is what makes ordinary calculus fail. For a smooth $f(t, X_t)$ with $dX = a\,dt + b\,dW$, Taylor-expanding to second order and keeping the $dW^2 = dt$ term gives **Ito's lemma**:

$$df = \left(f_t + a f_x + \tfrac12 b^2 f_{xx}\right)dt + b f_x\,dW$$

The extra $\tfrac12 b^2 f_{xx}$ term is the whole reason convexity matters in option pricing.

## 2. Geometric Brownian motion

The Black-Scholes model assumes $dS = \mu S\,dt + \sigma S\,dW$. Apply Ito to $f = \ln S$: $f_S = 1/S$, $f_{SS} = -1/S^2$, so

$$d\ln S = \left(\mu - \tfrac12\sigma^2\right)dt + \sigma\,dW \quad\Rightarrow\quad S_T = S_0 \exp\!\left[\left(\mu - \tfrac12\sigma^2\right)T + \sigma\sqrt{T}\,Z\right]$$

with $Z\sim\mathcal{N}(0,1)$. Two consequences: $S_T$ is log-normal, and $\mathbb{E}[S_T] = S_0 e^{\mu T}$ even though the median is lower. The $-\tfrac12\sigma^2$ is exactly what keeps the mean equal to $S_0 e^{\mu T}$. This is the formula used in `monte_carlo.py` (with $\mu\to r-q$).

## 3. Risk-neutral pricing and Girsanov

The price of a derivative does not depend on $\mu$. The reason is replication: a delta-hedged position is riskless, so it must earn $r$, and $\mu$ cancels out of the hedging argument. Pricing can therefore be done in a world where every asset has drift $r-q$ (the **risk-neutral measure** $\mathbb{Q}$):

$$V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[\text{payoff}(S_T)]$$

**Girsanov's theorem** is what justifies changing the drift. If $\tilde W_t = W_t + \lambda t$ with $\lambda = (\mu - r + q)/\sigma$ (the market price of risk), then there is a measure $\mathbb{Q}$ with density

$$\frac{d\mathbb{Q}}{d\mathbb{P}} = \exp\!\left(-\lambda W_T - \tfrac12\lambda^2 T\right)$$

under which $\tilde W$ is a Brownian motion. Substituting $dW = d\tilde W - \lambda\,dt$ into the GBM replaces $\mu$ by $r-q$. The volatility is unchanged: a change of measure shifts drifts, never diffusion coefficients. That is why volatility can be estimated from real-world data and then used for pricing, but drift cannot.

## 4. Feynman-Kac: from expectation to PDE

Let $V(t,S) = e^{-r(T-t)}\mathbb{E}^{\mathbb{Q}}[\text{payoff}(S_T)\mid S_t = S]$. The discounted value process $e^{-rt}V(t,S_t)$ is a $\mathbb{Q}$-martingale, so its $dt$ term vanishes. Applying Ito to it gives the Black-Scholes PDE:

$$V_t + \tfrac12\sigma^2 S^2 V_{SS} + (r-q) S V_S - rV = 0, \qquad V(T,S) = \text{payoff}(S)$$

So Monte Carlo (section 1 of [`monte_carlo.md`](monte_carlo.md)) and finite differences ([`pde.md`](pde.md)) are two numerical routes to the same object. Every pricer in this repository is a different way of computing this one quantity, which is why they can validate each other.

## 5. Put-call parity and no-arbitrage bounds

Long call plus short put at the same strike and maturity pays $S_T-K$ in every state, so

$$C - P = S e^{-qT} - K e^{-rT}$$

This uses no model at all, only the absence of arbitrage. With $S=100$, $K=100$, $T=1$, $r=5\%$, $q=2\%$, $\sigma=20\%$ the code gives $C = 9.227006$, $P = 6.330081$, and $C-P = 2.896925$ equals $Se^{-qT} - Ke^{-rT} = 2.896925$ to machine precision.

Model-free bounds that any quote must respect:

$$\max\!\left(Se^{-qT} - Ke^{-rT},\,0\right) \le C \le Se^{-qT}, \qquad \max\!\left(Ke^{-rT} - Se^{-qT},\,0\right) \le P \le Ke^{-rT}$$

These are exactly the bounds used in `implied_vol.py` to decide whether a quoted price has any implied vol at all. Here the call lower bound is 2.896925: a call quoted below that is an arbitrage, not a pricing problem.

## 6. Where the model breaks

Constant $\sigma$ is the weak point. Real markets show a smile, so the market quotes Black-Scholes *implied* vol as a function of strike and maturity instead. The documents on [Heston](heston.md), [SABR](sabr.md) and [SVI](svi.md) are about modelling that surface.

## Interview questions

**1. Why does the real-world drift $\mu$ not appear in the Black-Scholes price?**
Because the option is replicated by a dynamically hedged portfolio of stock and cash. Replication is pathwise, so the price is whatever the replicating portfolio costs, and $\mu$ cancels when the delta hedge removes the $dW$ term.

**2. Why is $\mathbb{E}[S_T] = S_0e^{\mu T}$ even though $S_T = S_0\exp((\mu-\sigma^2/2)T + \sigma W_T)$?**
$\mathbb{E}[e^{\sigma W_T}] = e^{\sigma^2 T/2}$ (the log-normal mean), which cancels the $-\sigma^2/2$ drift correction.

**3. What does Girsanov change and what does it leave alone?**
It changes the drift of the process and leaves the quadratic variation (hence $\sigma$) alone. Measures equivalent to $\mathbb{P}$ agree on which events are impossible.

**4. State Ito's lemma and use it on $W_t^2$.**
$d(W^2) = 2W\,dW + dt$. The extra $dt$ is why $\mathbb{E}[W_t^2]=t$ and why $W^2 - t$ is a martingale, not $W^2$.

**5. Derive put-call parity without a model.**
Buy a call, sell a put, same $K$, $T$: payoff $S_T - K$ in all states. A forward has the same payoff and costs $Se^{-qT} - Ke^{-rT}$ today, so the two portfolios must have the same price.

**6. A call is quoted at 2.5 with the bound above at 2.897. What do you do?**
It is below the lower bound: buy the call, short $e^{-qT}$ shares, borrow-lend $Ke^{-rT}$, lock in the difference. In practice check for stale quotes or wrong dividends first.
