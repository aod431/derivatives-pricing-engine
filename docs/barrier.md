# Barrier options

Implementation: [`src/pricing/barrier.py`](../src/pricing/barrier.py)
Verification notebook: [`notebooks/11_barrier_options.ipynb`](../notebooks/11_barrier_options.ipynb)

## 1. What they are

A barrier option is a vanilla that is created (**knock-in**) or destroyed (**knock-out**) when the spot touches a level $H$. Four types for each of call and put: down-and-out, down-and-in, up-and-out, up-and-in. They are cheaper than vanillas and the payoff is path-dependent, which is why they are priced with the reflection principle rather than with the terminal distribution only.

## 2. In-out parity

If you hold a knock-in and a knock-out with the same $K$, $H$, $T$, exactly one of them is alive at maturity:

$$\text{knock-in} + \text{knock-out} = \text{vanilla}$$

So only one family needs a formula. The code builds the knock-in prices from the Reiner-Rubinstein blocks $A,B,C,D$ (Haug's notation), which combine standard normal CDFs and the reflection term $(H/S)^{2\mu}$ with $\mu=\frac{r-q-\sigma^2/2}{\sigma^2}$, and gets knock-outs by subtraction. The assumptions are continuous monitoring, zero rebate, and a barrier not yet hit.

Parameters: $S_0=100$, $K=100$, $T=1$, $r=5\%$, $\sigma=20\%$, $q=0$, call. Vanilla BS = 10.450584.

| Type | Barrier | Closed form |
|---|---|---|
| down-and-out | 90 | 8.665472 |
| down-and-in | 90 | 1.785112 |
| up-and-out | 130 | 3.332858 |
| up-and-in | 130 | 7.117726 |

Down-and-out plus down-and-in is $10.450584$, the vanilla, as required (tested to $10^{-10}$ in `test_barrier.py`). As $H\to0$ the down-and-out converges to the vanilla; as $H\to S_0$ it goes to zero: 10.3513 for $H=80$, 8.6655 for $H=90$, 5.6363 for $H=95$, 1.3675 for $H=99$.

## 3. Monte Carlo and the discrete-monitoring bias

Simulating on $n$ dates and checking $S_{t_i}\le H$ only at those dates misses crossings *between* dates. The paths that crossed and came back are counted as alive, so a knock-out is **overpriced**. This is a bias, not noise, and it does not go away by adding paths. Down-and-out call, $H=90$, 400k paths, exact 8.665472:

| Steps | Naive (discrete check) | With Brownian bridge | Bridge std error |
|---|---|---|---|
| 12 | 9.6021 | 8.6973 | 0.0227 |
| 50 | 9.1643 | 8.6539 | 0.0228 |
| 250 | 8.9587 | 8.7047 | 0.0231 |

The naive bias is +0.94, +0.50 and +0.29, falling roughly like $1/\sqrt{n}$ (the known $O(\sqrt{\Delta t})$ barrier shift). The bridge version stays within about 2 standard errors of the exact price at every $n$, even at 12 steps.

**Brownian bridge correction.** Conditional on endpoints $x_i, x_{i+1}$ (log-spot) both above $\ln H$, the probability that the path touched $\ln H$ in between is

$$\exp\!\left(-\frac{2(x_i-\ln H)(x_{i+1}-\ln H)}{\sigma^2\Delta t}\right)$$

Each path is weighted by the product of $1 - p_{cross}$ over the steps, which estimates the *continuously* monitored price without simulating a finer grid. This also lowers variance, since survival is a probability instead of a 0/1 indicator. Real contracts are often monitored daily, which is closer to the naive version: `brownian_bridge=False` prices that discretely monitored contract and the closed form does not apply to it.

## 4. Risk: why desks care

Close to the barrier a knock-out's delta becomes very large and its gamma changes sign: the value has to fall to zero at $H$, so near $H$ the price is very steep in $S$. The same steepness shows up as sensitivity to $H$ itself (moving $H$ from 95 to 99 cuts the price from 5.6363 to 1.3675). Hedging this with a vanilla delta is unreliable. Traders shift the barrier ("barrier shift") to price a safer value, or hedge with a static portfolio of vanillas.

## Interview questions

**1. State in-out parity and what it is used for.**
Knock-in plus knock-out equals the vanilla (with zero rebate). It cuts the work in half and is a sanity check for any implementation.

**2. Why does a discretely monitored knock-out cost more than a continuous one?**
Fewer monitoring dates means fewer chances to be knocked out, so it is more likely to be alive. The gap shrinks like $\sqrt{\Delta t}$.

**3. What is the reflection principle?**
For a driftless Brownian motion, paths that hit $H$ and end below it are in bijection (by reflecting after the hit) with paths ending above $H$, so $P(\max\ge H, W_T\le x)=P(W_T\ge 2H-x)$. This produces the $(H/S)^{2\mu}$ terms.

**4. Why does the bridge correction not need a finer time grid?**
Given the two endpoints, the path in between is a Brownian bridge, whose hitting probability of a level is known in closed form.

**5. Why is the vega of a knock-out option sometimes negative?**
Higher volatility makes the barrier more likely to be hit, which destroys value. For up-and-out calls this can outweigh the usual positive vega.

**6. How would you hedge a down-and-out call?**
Dynamically with delta (unstable near $H$) or statically by selling a vanilla and buying a replicating portfolio of options that offsets it at the barrier.
