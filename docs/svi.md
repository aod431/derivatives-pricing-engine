# SVI and arbitrage-free implied-volatility surfaces

Implementation: [`src/pricing/svi.py`](../src/pricing/svi.py)
Verification notebook: [`notebooks/08_sabr_svi_calibration.ipynb`](../notebooks/08_sabr_svi_calibration.ipynb)
Used in production-style form by [`vol-surface-risk/`](../vol-surface-risk/).

## 1. Raw SVI

SVI ("stochastic volatility inspired", Gatheral) parametrises **total implied variance** $w(k)=\sigma_{imp}^2(k)\,T$ as a function of log-moneyness $k=\ln(K/F)$:

$$w(k) = a + b\left(\rho(k-m) + \sqrt{(k-m)^2 + s^2}\right)$$

- $a$: overall level. $b$: slope of the wings (angle). $\rho$: skew (rotation). $m$: horizontal shift. $s$: smoothing of the ATM vertex.

Working in total variance is deliberate: the no-arbitrage conditions are clean in $w$, and for large $|k|$ it is linear in $k$, which is the shape Lee's moment formula allows. Lee's bound says the wing slopes of $w$ cannot exceed 2; for SVI the wing slopes are $b(1+\rho)$ on the right and $b(1-\rho)$ on the left, so the constraint is $b(1+|\rho|)\le 2$. (Other sources quote a bound of 4; I use the form that follows directly from Lee's statement for total variance.) The bad slice used in section 2 has $b(1+\rho)=5.7$.

Example used below: $a=0.02$, $b=0.1$, $\rho=-0.4$, $m=0$, $s=0.2$. Wing slopes are $0.06$ and $0.14$, well inside the bound.

| $k$ | -0.50 | -0.25 | 0 | 0.25 | 0.50 |
|---|---|---|---|---|---|
| $w(k)$ | 0.0939 | 0.0620 | 0.0400 | 0.0420 | 0.0539 |
| $\sigma_{imp}$ at $T=1$ | 0.3064 | 0.2490 | 0.2000 | 0.2050 | 0.2321 |

## 2. Static arbitrage

A surface of call prices is free of static arbitrage if there is no butterfly or calendar spread with negative cost.

**Butterfly (strike direction).** Butterfly prices are the risk-neutral density times a positive factor, so the density must be non-negative. For SVI, Gatheral and Jacquier show that

$$g(k) = \left(1-\frac{k\,w'}{2w}\right)^2 - \frac{w'^2}{4}\left(\frac1w+\frac14\right) + \frac{w''}{2} \;\ge\; 0 \quad\forall k$$

is the density condition. `butterfly_g` evaluates it analytically with $w'=b\left(\rho + \frac{k-m}{\sqrt{(k-m)^2+s^2}}\right)$ and $w''=\frac{bs^2}{((k-m)^2+s^2)^{3/2}}$.

- Example slice: $\min g = 0.2798$ on $[-1.5,1.5]$, and $g(0)=1.2399$. No arbitrage.
- Bad slice $a=0$, $b=3$, $\rho=0.9$, $m=0$, $s=0.01$: $\min g=-47.92$. It is flagged by `has_butterfly_arbitrage`.

**Calendar (time direction).** Total variance must not decrease with maturity at a fixed $k$: $w(k,T_2)\ge w(k,T_1)$ for $T_2>T_1$. Otherwise a calendar spread has negative value. `calendar_violations` compares adjacent slices on a grid and returns the offending maturity pairs.

Both conditions are needed. Passing $g\ge0$ on each slice says nothing about the relation between slices.

## 3. Fitting a slice

`fit_svi` minimises squared error in total variance, with bounds ($b>0$, $|\rho|<1$, $s>0$) and a penalty $10\cdot\min(g,0)$ evaluated on a grid wider than the data, because wings with no quotes are exactly where arbitrage creeps in. The objective is non-convex, so it runs from 12 starting points ($\rho_0\in\{-0.7,-0.3,0,0.3\}$ times $s_0\in\{0.05,0.2,0.5\}$) and keeps the best.

Test on the example slice at 21 points in $k\in[-0.5,0.5]$ with 3% multiplicative noise on $w$: the fit gives $a=0.0154$, $b=0.107$, $\rho=-0.392$, $m=-0.003$, $s=0.234$, the maximum error against the noise-free curve is 0.00114 in total variance, and the result has no butterfly arbitrage on the check grid.

## 4. Limits worth knowing

- The penalty is a soft constraint. It does not guarantee $g\ge 0$ outside the grid, so always re-check with `has_butterfly_arbitrage` after fitting.
- $a,b,\rho,m,s$ are strongly correlated; quite different parameter sets can describe the same slice within noise, so do not interpret parameters one by one.
- Fitting each maturity separately does not guarantee the calendar condition. The fix is to check and re-fit, or to use a surface parametrisation (SSVI) where the conditions become constraints on a few parameters.
- Outside the quoted strikes SVI extrapolates linearly in variance. That is the Lee bound at work, not market information.

## Interview questions

**1. Why fit total variance instead of implied vol?**
Calendar no-arbitrage is monotonicity of $w$ in $T$, and the butterfly condition is written in $w$, $w'$, $w''$. In vol terms both are messier. The wings are also asymptotically linear in $w$.

**2. What does the butterfly condition mean economically?**
A butterfly pays $\max(0,\Delta-|S_T-K|)$, so its price is a positive-weighted average of the density. A negative price means a negative density, which is an arbitrage.

**3. How is the density obtained from the smile?**
Breeden-Litzenberger: $p(K)=e^{rT}\,\partial^2 C/\partial K^2$. Because $C$ depends on $K$ directly and through $\sigma_{imp}(K)$, a smile with too much curvature gives $p<0$.

**4. What do the SVI parameters do?**
$a$ shifts the level, $b$ scales both wings, $\rho$ rotates (skew), $m$ translates horizontally, $s$ rounds off the vertex.

**5. What is Lee's moment formula?**
It links the tail behaviour of the density to the wing slopes of total variance: $\limsup_{k\to\pm\infty} w(k)/|k| \le 2$. A steeper wing implies an infinite moment of the stock, which is incompatible with a finite forward price.

**6. A fitted slice has $g\ge0$ but the surface still has arbitrage. How?**
A calendar violation between maturities, or $g<0$ outside the grid that was tested.
