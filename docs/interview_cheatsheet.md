# Quant interview cheat sheet

One page of things to know cold. Each item points to the document where it is derived. Numeric checks use $S=K=100$, $T=1$, $r=5\%$, $\sigma=20\%$, $q=2\%$ (call 9.227006, put 6.330081).

## Pricing and parity

- Risk-neutral price: $V_0=e^{-rT}\mathbb{E}^{\mathbb{Q}}[\text{payoff}]$; under $\mathbb{Q}$, $dS=(r-q)S\,dt+\sigma S\,dW$ ([foundations](foundations.md)).
- Black-Scholes: $C=Se^{-qT}N(d_1)-Ke^{-rT}N(d_2)$, $d_{1,2}=\frac{\ln(S/K)+(r-q\pm\sigma^2/2)T}{\sigma\sqrt T}$.
- Parity: $C-P=Se^{-qT}-Ke^{-rT}$ (2.896925 in the example, checked to machine precision).
- Bounds: $C\ge\max(Se^{-qT}-Ke^{-rT},0)$, $C\le Se^{-qT}$; $P\le Ke^{-rT}$.
- ATM approximation (with $r=q=0$): $C\approx0.4\,S\sigma\sqrt T$. For $\sigma=20\%$ that is 8.0 against an exact 7.966.
- Black-Scholes PDE: $V_t+\tfrac12\sigma^2S^2V_{SS}+(r-q)SV_S-rV=0$.
- Digital call price: $e^{-rT}N(d_2)$; its delta is $e^{-rT}\phi(d_2)/(S\sigma\sqrt T)$.

## Greeks ([greeks](greeks.md))

- $\Delta_C=e^{-qT}N(d_1)$, $\Delta_P=\Delta_C-e^{-qT}$. In the example $\Delta_C=0.586851$.
- $\Gamma=\frac{e^{-qT}\phi(d_1)}{S\sigma\sqrt T}$ and $\mathcal V=Se^{-qT}\phi(d_1)\sqrt T$ are the same for calls and puts (0.018951 and 37.9012).
- $\Gamma$ and vega are related: $\mathcal V=S^2\sigma T\,\Gamma$ (for $q$-adjusted $\Gamma$). With the numbers above $100^2\cdot0.2\cdot0.018951=37.90$.
- **Gamma-theta relation:** $\Theta+\tfrac12\sigma^2S^2\Gamma+(r-q)S\Delta-rV=0$. In the example the left side is $1.4\times10^{-15}$. Theta is $-5.0893$ and the gamma term is $+3.7901$.
- Delta of an ATM option is about 0.5 (0.587 here because the forward is above spot by the carry $r-q$).

## Volatility

- Implied vol: the $\sigma$ solving $BS(\sigma)=$ price. Price is increasing in $\sigma$, so the inverse is unique ([implied_vol](implied_vol.md)).
- Skew, smile: $\rho<0$ in Heston or SABR gives equity-style downward skew; $\xi$ or $\nu$ gives curvature ([heston](heston.md), [sabr](sabr.md)).
- Total variance $w=\sigma^2T$. Butterfly arbitrage free: $g(k)\ge0$. Calendar arbitrage free: $w$ non-decreasing in $T$ ([svi](svi.md)).
- Breeden-Litzenberger: density $=e^{rT}\,\partial^2C/\partial K^2$; digital $=-\partial C/\partial K$.
- Feller: $2\kappa\theta\ge\xi^2$.

## Numerical methods

- Monte Carlo error $\propto N^{-1/2}$; halving the error costs 4x paths. Standard error $=s/\sqrt N$ ([monte_carlo](monte_carlo.md)).
- Tree: $u=e^{\sigma\sqrt{\Delta t}}$, $d=1/u$, $p=\frac{e^{(r-q)\Delta t}-d}{u-d}$; error $O(1/n)$ with sawtooth ([binomial](binomial.md)).
- Crank-Nicolson: second order, unconditionally stable, needs Rannacher start-up for the payoff kink ([pde](pde.md)).
- Greeks by MC: pathwise (smooth payoffs), likelihood ratio (discontinuous), bump with common random numbers ([mc_greeks](mc_greeks.md)).
- LSM is low-biased; discrete barrier monitoring overprices knock-outs ([american_lsm](american_lsm.md), [barrier](barrier.md)).
- Knock-in + knock-out = vanilla.

## Stochastic calculus

- Ito: $df=(f_t+af_x+\tfrac12b^2f_{xx})dt+bf_x\,dW$; $dW^2=dt$.
- GBM: $S_T=S_0\exp((\mu-\tfrac12\sigma^2)T+\sigma W_T)$, $\mathbb{E}S_T=S_0e^{\mu T}$.
- $\mathbb{E}[e^{\sigma W_T}]=e^{\sigma^2T/2}$; $W_t^2-t$ is a martingale.
- Girsanov: changes drift, not volatility.
- Delta-hedged P&L $\approx\int\tfrac12\Gamma S^2(\sigma_{imp}^2-\sigma_{real}^2)dt$ ([hedging](hedging.md)).
- Hedge error std $\propto1/\sqrt n$ rebalances.

## Early exercise

- American call on a no-dividend asset: never exercised early; $C_{Am}=C_{Eu}$.
- American put: always an early-exercise premium (6.0900 vs 5.5735 for $q=0$, $r=5\%$).
- Calls on dividend payers are exercised just before the ex-date when dividends exceed the interest on the strike.

## Probability you get asked

- $P(\text{ITM at }T)$ under $\mathbb{Q}$ is $N(d_2)$; under the share measure $N(d_1)$.
- For a standard normal $Z$: $\mathbb{E}[Z^2]=1$, $\mathbb{E}[Z^4]=3$, $\mathbb{E}[\max(Z,0)]=1/\sqrt{2\pi}$.
- Variance of a sum: $\mathrm{Var}(X+Y)=\mathrm{Var}X+\mathrm{Var}Y+2\mathrm{Cov}$. A correlation of $\rho$ with a control variate cuts variance by $1-\rho^2$.
