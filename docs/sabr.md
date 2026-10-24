# SABR

Implementation: [`src/pricing/sabr.py`](../src/pricing/sabr.py)
Verification notebook: [`notebooks/08_sabr_svi_calibration.ipynb`](../notebooks/08_sabr_svi_calibration.ipynb)

## 1. The model

SABR describes a forward $F$ with its own stochastic volatility $\alpha$:

$$dF = \alpha F^\beta\,dW^1, \qquad d\alpha = \nu\alpha\,dW^2, \qquad d\langle W^1,W^2\rangle=\rho\,dt$$

- $\alpha$: initial volatility level (sets the ATM vol).
- $\beta\in[0,1]$: backbone, how normal ($\beta=0$) or lognormal ($\beta=1$) the dynamics are.
- $\rho$: correlation, tilts the smile.
- $\nu$: vol of vol, adds curvature.

It is the market standard for interest-rate smiles (swaptions, caps) and common in FX, because it comes with a fast closed-form implied vol.

## 2. Hagan's formula

There is no closed-form price, but Hagan et al. (2002) derived an asymptotic expansion for the Black implied vol:

$$\sigma_B(K) \approx \frac{\alpha}{(FK)^{\frac{1-\beta}{2}}\left[1+\frac{(1-\beta)^2}{24}\ln^2\frac FK + \dots\right]}\cdot\frac{z}{x(z)}\cdot\left[1 + \left(\frac{(1-\beta)^2\alpha^2}{24(FK)^{1-\beta}} + \frac{\rho\beta\nu\alpha}{4(FK)^{\frac{1-\beta}{2}}} + \frac{2-3\rho^2}{24}\nu^2\right)T\right]$$

with $z = \frac\nu\alpha (FK)^{\frac{1-\beta}2}\ln\frac FK$ and $x(z) = \ln\frac{\sqrt{1-2\rho z+z^2}+z-\rho}{1-\rho}$. At the money $z\to0$ and $z/x(z)\to1$; the code switches to this limit when $|z|<10^{-7}$ to avoid $0/0$.

You then feed $\sigma_B$ into Black's formula to get a price. Everything is a small expansion in $T$, so it is an approximation, not an exact solution of the SABR SDE.

## 3. Effect of the parameters

$F=100$, $T=1$, $\beta=0.5$, $\alpha=2.0$ (so ATM vol $\approx 20\%$). Implied vols at strikes 80 / 100 / 120:

| Setting | 80 | 100 | 120 |
|---|---|---|---|
| $\rho=-0.6$, $\nu=0.5$ | 0.2481 | 0.2005 | 0.1692 |
| $\rho=0$, $\nu=0.5$ | 0.2252 | 0.2043 | 0.2020 |
| $\rho=+0.6$, $\nu=0.5$ | 0.1883 | 0.2035 | 0.2243 |
| $\rho=-0.4$, $\nu=0.2$ | 0.2215 | 0.2002 | 0.1849 |
| $\rho=-0.4$, $\nu=0.6$ | 0.2497 | 0.2034 | 0.1822 |
| $\rho=-0.4$, $\nu=1.2$ | 0.3087 | 0.2159 | 0.2026 |

$\rho$ moves the skew, $\nu$ the curvature, as in Heston. The $\beta$ backbone mostly acts as an extra skew: with $\alpha$ rescaled to keep the ATM vol near 20%, $\beta=0$ gives 0.2552 at $K=80$ and 0.1741 at $K=120$, $\beta=1$ gives 0.2294 and 0.1893 (with $\rho=-0.4$, $\nu=0.5$).

## 4. Calibration and an honest finding about $\beta$

`calibrate_sabr` fixes $\beta$ and fits $(\alpha,\rho,\nu)$ with bounded least squares, seeding $\alpha$ from the ATM vol. On a synthetic smile generated with $(\alpha,\beta,\rho,\nu)=(2,0.5,-0.4,0.6)$ it recovers $\alpha=2.0000$, $\rho=-0.4000$, $\nu=0.6000$.

The question is whether $\beta$ can be fitted too. Fitting the same smile with the *wrong* $\beta$:

| Fixed $\beta$ | Fitted $\rho$ | Fitted $\nu$ | Max vol error |
|---|---|---|---|
| 0.0 | -0.225 | 0.546 | 0.00117 |
| 0.5 (true) | -0.400 | 0.600 | $1\times10^{-11}$ |
| 1.0 | -0.536 | 0.671 | 0.00128 |

A wrong $\beta$ is absorbed by $\rho$ and $\nu$ and costs only about 12 bp of vol across the smile. On real, noisy quotes that is within the bid-ask spread, so $\beta$ is not identifiable from one smile. The convention is to choose it from the market regime (e.g. 0.5 or 0 for rates) and calibrate the rest. The practical consequence is that $\rho$ and $\nu$ are **not** meaningful on their own: only the combination with the chosen $\beta$ is.

## 5. Known problems

- The expansion degrades for long maturities and very low strikes. With $\alpha=2$, $\beta=0.5$, $\rho=-0.4$, $\nu=1.2$, $T=5$, the formula gives vol 1.1273 at $K=20$ and 0.7821 at $K=40$ against 0.2796 at $K=100$. These are the kind of numbers that imply a negative density, so the smile cannot be used there.
- Fixed-$\beta$ SABR gives no consistency across maturities; each expiry is calibrated separately, so calendar arbitrage is not controlled.

Those are the reasons the equity world prefers [SVI](svi.md) with explicit arbitrage constraints.

## Interview questions

**1. What does $\beta$ do and why is it usually fixed?**
It sets the backbone: how ATM vol moves when the forward moves. It is nearly collinear with $\rho$ in a single smile (table above), so it is chosen by regime or from historical backbone behaviour.

**2. Which parameters set ATM level, skew and curvature?**
$\alpha$ (level), $\rho$ (skew), $\nu$ (curvature).

**3. Why is Hagan's formula an approximation and when does it fail?**
It is a first-order expansion in $T$ around the ATM point. It fails for long maturities, low strikes and large $\nu$, where densities go negative.

**4. Why can SABR handle negative rates?**
With $\beta=0$ (normal SABR) or a shifted SABR the forward can be negative; a lognormal-based backbone cannot.

**5. How do you hedge with SABR?**
Delta must include the vol move that comes with a forward move: $\Delta_{SABR} = \Delta_{BS} + \mathrm{Vega}\cdot\partial\sigma_B/\partial F$. This depends on $\beta$ and $\rho$, which is why the choice of $\beta$ matters for risk even if it barely affects the fit.

**6. How would you calibrate the full surface?**
One slice at a time, with parameters interpolated in $T$, then check for calendar arbitrage; or move to a parametrisation such as SVI with explicit no-arbitrage conditions.
