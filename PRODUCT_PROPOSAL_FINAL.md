---
title: "Structured Gold Forward with Knock-Out Barriers"
subtitle: "Product Development Memorandum"
author: "Derivatives Structuring Desk"
date: "February 2026"
subject: "Pricing and Risk Analysis — Zeus Gold Group Hedging Facility"
---

\newpage

**CONFIDENTIAL**

**TO:** Product Committee, Alphabank S.A.

**FROM:** Derivatives Structuring Desk

**DATE:** February 2026

**RE:** Pricing and Risk Analysis — Zeus Gold Group AG Hedging Facility

---

# Executive Summary

This memorandum presents our analysis of a proposed structured hedging facility for Zeus Gold Group AG ("Z Group"). The product combines exposure to LBMA gold prices with automatic termination features linked to EUR/USD exchange rate movements.

We have developed a comprehensive pricing framework validated through multiple methodologies. Our analysis identifies several structural considerations that warrant discussion before proceeding to term sheet finalization.

**Transaction Summary**

| Parameter | Specification |
|-----------|---------------|
| Notional Principal | EUR 500,000,000 |
| Reference Asset | LBMA Gold PM Fixing (USD/oz) |
| Strike Price | USD 4,600 per troy ounce |
| Tenor | 2 years (March 2026 — February 2028) |
| Knock-Out Barriers | EUR/USD < 1.05 or EUR/USD > 1.25 |

**Key Findings**

| Metric | Result |
|--------|--------|
| Z Group Present Value | EUR +64 million |
| Alphabank Present Value | EUR −64 million |
| Knock-Out Probability | 64% |
| Expected Contract Duration | 12 months |

The positive present value for Z Group reflects gold's surge above the strike price, placing the forward at 113% moneyness. The 64% knock-out probability is driven primarily by the upper barrier's proximity to current EUR/USD spot (5.8% distance), representing a fundamental reversal from earlier market conditions where the lower barrier dominated.

\newpage

# 1. Transaction Overview

Zeus Gold Group, a Frankfurt-headquartered jewelry manufacturer, seeks to hedge its USD-denominated gold procurement costs while managing EUR/USD translation risk. The proposed facility would run for two years commencing March 2026.

## 1.1 Settlement Mechanics

At settlement time $\tau$ (maturity or knock-out, whichever occurs first), with LBMA gold fixing at price $P$:

$$\text{Z Group Payoff} = N \times \frac{P_{\tau} - K}{K}$$

$$\text{Alphabank Payoff} = N \times \frac{K - P_{\tau}}{K}$$

where:

- $N = 500{,}000{,}000$ EUR (notional principal)
- $K = 4{,}600$ USD/oz (strike price)
- $P_{\tau}$ = LBMA Gold PM fixing at settlement

## 1.2 Knock-Out Mechanism

The contract terminates immediately upon the first occurrence of:

$$X_t < L \quad \text{or} \quad X_t > U$$

where $X_t$ denotes the EUR/USD rate, $L = 1.05$ (lower barrier), and $U = 1.25$ (upper barrier).

The stopping time is defined as:

$$\tau_{KO} = \inf\{t \geq 0 : X_t \notin (L, U)\}$$

Settlement occurs at $\tau = \min(T, \tau_{KO})$ where $T = 2$ years.

\newpage

# 2. Mathematical Framework

## 2.1 Stochastic Model

Both underlying assets follow geometric Brownian motion under the risk-neutral measure $\mathbb{Q}$.

**Gold Price Dynamics (USD):**

$$\frac{dS_t}{S_t} = (r_{USD} - q)\,dt + \sigma_S\,dW_t^{(1)}$$

**EUR/USD Exchange Rate:**

$$\frac{dX_t}{X_t} = (r_{EUR} - r_{USD})\,dt + \sigma_X\,dW_t^{(2)}$$

**Correlation Structure:**

$$\mathbb{E}[dW_t^{(1)} \cdot dW_t^{(2)}] = \rho\,dt$$

## 2.2 Parameter Estimates

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Gold spot | $S_0$ | USD 5,203/oz | yfinance live (GC=F, Feb 2026) |
| EUR/USD spot | $X_0$ | 1.181 | yfinance live (EURUSD=X, Feb 2026) |
| USD risk-free rate | $r_{USD}$ | 3.6% | 13-week T-bill (^IRX) |
| EUR risk-free rate | $r_{EUR}$ | 2.0% | ECB deposit rate (configured) |
| Gold volatility | $\sigma_S$ | 41% | EWMA (λ=0.94, GC=F) |
| EUR/USD volatility | $\sigma_X$ | 6.2% | EWMA (λ=0.94, EURUSD=X) |
| Correlation | $\rho$ | −0.30 | 126-day rolling (fallback) |
| Gold convenience yield | $q$ | 3.6% | Futures term structure (GC=F vs GCJ26.CMX) |

## 2.3 Risk-Neutral Valuation

The present value under the EUR risk-neutral measure:

$$V_0 = \mathbb{E}^{\mathbb{Q}}\left[e^{-r_{EUR} \cdot \tau} \cdot \text{Payoff}_{\tau}\right]$$

The path-dependent barrier feature precludes closed-form solutions, necessitating Monte Carlo methods.

\newpage

# 3. Numerical Implementation

## 3.1 Simulation Methodology

We employ Monte Carlo simulation with the following specifications:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Simulation paths | 100,000 | Adequate precision for indicative pricing |
| Time steps | 504 | Daily monitoring over 2 years |
| Random seed | Fixed | Reproducibility |

**Discretization Scheme**

Asset prices are simulated using the exact log-normal solution:

$$S_{t+\Delta t} = S_t \cdot \exp\left[\left(\mu_S - \frac{\sigma_S^2}{2}\right)\Delta t + \sigma_S\sqrt{\Delta t}\,Z_1\right]$$

$$X_{t+\Delta t} = X_t \cdot \exp\left[\left(\mu_X - \frac{\sigma_X^2}{2}\right)\Delta t + \sigma_X\sqrt{\Delta t}\,Z_2\right]$$

where $Z_1, Z_2$ are correlated standard normals generated via Cholesky decomposition:

$$Z_2 = \rho Z_1 + \sqrt{1-\rho^2}\,Z_{\perp}$$

## 3.2 Variance Reduction

Two techniques are implemented to improve computational efficiency:

**Antithetic Variates:** For each path with innovations $\{Z_t\}$, we also simulate the reflected path $\{-Z_t\}$. The negative correlation between paired paths reduces variance.

**Control Variate:** The vanilla gold forward (without barriers) serves as a control:

$$\hat{V}_{adj} = \hat{V}_{exotic} + \beta\left(V_{vanilla}^{analytical} - \hat{V}_{vanilla}\right)$$

where $\beta$ is the optimal control coefficient estimated from sample covariance.

Combined, these techniques reduce standard errors by approximately 60%.

\newpage

# 4. Pricing Results

## 4.1 Base Case Valuation

| Metric | Value |
|--------|-------|
| Z Group Present Value | EUR +63,968,230 |
| Alphabank Present Value | EUR −63,968,230 |
| Standard Error | EUR 896,173 |
| 95% Confidence Interval | [+62.2M, +65.7M] |

## 4.2 Barrier Analysis

| Metric | Value |
|--------|-------|
| Overall Knock-Out Rate | 64.2% |
| Lower Barrier Breaches | 27.3% |
| Upper Barrier Breaches | 36.9% |
| Average Time to Knock-Out | 0.98 years (11.8 months) |

The barrier breach profile has shifted dramatically compared to earlier market conditions. The upper barrier now accounts for the majority of knock-outs (37% vs 27% lower), reflecting the EUR/USD spot at 1.181—only 5.8% from the upper barrier versus 11.1% from the lower. The interest rate differential ($r_{EUR} - r_{USD} = -1.6\%$) still implies euro depreciation drift, but the proximity asymmetry dominates.

**[INSERT FIGURE: monte_carlo_paths.png]**

## 4.3 Convergence Verification

Monte Carlo estimates stabilize as path counts increase:

| Paths | Price Estimate | Standard Error |
|-------|---------------|----------------|
| 5,000 | EUR +61.5M | EUR 3,844K |
| 10,000 | EUR +61.6M | EUR 2,698K |
| 25,000 | EUR +62.7M | EUR 1,768K |
| 50,000 | EUR +63.1M | EUR 1,241K |
| 100,000 | EUR +64.0M | EUR 896K |

Standard errors decay proportionally to $1/\sqrt{n}$, confirming proper convergence behavior.

**[INSERT FIGURE: convergence_analysis.png]**

\newpage

# 5. Critical Assessment

## 5.1 Strike Price Analysis

The specified strike of USD 4,600/oz warrants careful examination.

**Forward Price Calculation:**

$$F_{0,T} = S_0 \cdot e^{(r_{USD} - q) \cdot T} = 5203 \cdot e^{(0.036 - 0.036) \cdot 2} \approx \text{USD } 5{,}203\text{/oz}$$

With convenience yield approximately equal to the USD risk-free rate, the forward is near spot. The strike sits 12% below the forward, placing Z Group in the money:

$$\text{Moneyness} = \frac{F_{0,T}}{K} = \frac{5203}{4600} = 113.1\%$$

**Alternative Strike Analysis:**

| Strike | Forward Relationship | Z Group PV |
|--------|---------------------|------------|
| USD 4,000 | 23% below forward | EUR +146M |
| USD 4,300 | 17% below forward | EUR +101M |
| USD 4,600 | 12% below forward | EUR +63M |
| USD 4,900 | 6% below forward | EUR +29M |
| USD 5,200 | At-the-money | EUR ~0M |
| USD 5,500 | 6% above forward | EUR −27M |

**[INSERT FIGURE: scenario_strike.png]**

## 5.2 Barrier Configuration

With EUR/USD at 1.181, the upper barrier at 1.25 is now the proximate risk:

$$\text{Distance to Upper Barrier} = \frac{U - X_0}{X_0} = \frac{1.25 - 1.181}{1.181} = 5.8\%$$

$$\text{Distance to Lower Barrier} = \frac{X_0 - L}{X_0} = \frac{1.181 - 1.05}{1.181} = 11.1\%$$

With 6.2% annual EUR/USD volatility, the asymmetric positioning creates an upper-barrier-dominated knock-out profile.

**Alternative Configurations:**

| Corridor | Knock-Out Rate | Expected Duration |
|----------|---------------|-------------------|
| [1.05, 1.25] | 64% | 12 months |
| [1.00, 1.30] | 27% | 15 months |
| [0.95, 1.35] | 10% | 17 months |

**[INSERT FIGURE: scenario_barrier.png]**

\newpage

# 6. Risk Sensitivities

## 6.1 Greeks Summary

| Greek | Value | Interpretation |
|-------|-------|----------------|
| $\Delta_{gold}$ | EUR 105,677 per USD 1 | First-order gold sensitivity |
| $\Gamma_{gold}$ | EUR −669 | Gold convexity |
| $\Delta_{FX}$ | EUR −8.18M per 0.01 FX | EUR/USD sensitivity (negative: upper barrier risk) |
| $\mathcal{V}_{gold}$ | EUR −2.88M per 1% vol | Gold vega (higher vol increases KO probability) |
| $\rho_{EUR}$ | EUR −1,031M per 1bp | EUR rate sensitivity |
| $\rho_{corr}$ | EUR −973K per 0.05 corr | Correlation sensitivity |

**[INSERT FIGURE: greeks_summary.png]**

## 6.2 Hedging Implications

**Delta Hedging:** The gold delta of EUR 106K per dollar implies a hedge ratio of approximately:

$$\text{Gold Hedge} = \frac{\Delta_{gold}}{S_0} \times K = \frac{105{,}677}{5203} \times 4600 \approx 93{,}400 \text{ oz}$$

**Barrier Risk:** With EUR/USD at 1.181, the upper barrier at 1.25 is only 5.8% away. As EUR/USD approaches either barrier, gamma and delta become increasingly unstable—the characteristic "pin risk" of barrier options. The negative FX delta (EUR −8.2M per 0.01) reflects that EUR appreciation toward 1.25 destroys contract value through knock-out.

\newpage

# 7. Model Validation

## 7.1 Alternative Specifications

To ensure robustness, we compared valuations across three model specifications:

| Model | Z Group PV | Knock-Out Rate |
|-------|-----------|----------------|
| Base GBM | EUR +63.1M | 64.5% |
| Heston Stochastic Vol | EUR +64.1M | 64.1% |
| Merton Jump-Diffusion | EUR +65.7M | 64.1% |

All models converge within 4%, with Heston and Merton producing slightly higher valuations due to stochastic volatility and jump dynamics amplifying the in-the-money payoff. Model specification risk remains secondary to market parameter uncertainty.

**[INSERT FIGURE: model_comparison.png]**

## 7.2 Analytical Benchmark

The vanilla gold forward (without barriers) provides a sanity check:

$$V_{vanilla} = e^{-r_{EUR} \cdot T} \cdot N \cdot \frac{F_{0,T} - K}{K} = e^{-0.020 \times 2} \cdot 500M \cdot \frac{5203 - 4600}{4600}$$

$$V_{vanilla} = \text{EUR } +63{,}004{,}791$$

The knock-out version (EUR +64.0M) is EUR 1.0M higher than the vanilla, reflecting that early termination locks in profits when gold is above strike — a reversal from the prior regime where knock-outs destroyed value.

\newpage

# 8. Conclusions and Recommendations

## 8.1 Summary of Findings

The proposed structure is technically sound and priceable using standard Monte Carlo techniques. The February 2026 market environment has fundamentally altered the risk profile:

1. **Position reversal:** Gold's surge to $5,203/oz places Z Group firmly in the money (113% moneyness), with a positive PV of EUR +64M. This contrasts sharply with earlier market conditions where the position was deeply out-of-the-money.

2. **Barrier risk shift:** The upper EUR/USD barrier at 1.25 is now only 5.8% from spot, making it the primary knock-out driver (37% of paths). The 64% total knock-out rate and 12-month expected duration represent a more balanced risk profile than the prior 93%/5-month scenario.

3. **Elevated volatility:** Gold volatility at 41% (EWMA) is more than double historical norms, increasing both the potential upside for Z Group and the mark-to-market volatility for Alphabank's hedging book.

## 8.2 Recommendations

We recommend proceeding to term sheet stage with the following considerations:

1. Alphabank should carefully assess credit exposure given the positive Z Group PV, ensuring adequate collateral arrangements

2. The upper barrier proximity (5.8%) warrants active monitoring—EUR appreciation events could trigger knock-out and crystallize Z Group's gain

3. Consider whether alternative barrier configurations (e.g., [1.00, 1.30] with 27% KO rate) would provide Z Group with more durable hedging protection

## 8.3 Next Steps

Upon Committee approval, we will:

- Finalize term sheet documentation
- Establish hedging framework with Trading Desk
- Coordinate credit approval with Risk Management
- Schedule client presentation

---

**[END OF MEMORANDUM]**

\newpage

# Appendix A: Nomenclature

| Symbol | Description |
|--------|-------------|
| $S_t$ | Gold spot price (USD/oz) at time $t$ |
| $X_t$ | EUR/USD exchange rate at time $t$ |
| $K$ | Strike price (USD 4,600/oz) |
| $N$ | Notional principal (EUR 500M) |
| $T$ | Maturity (2 years) |
| $L$, $U$ | Lower (1.05) and upper (1.25) barriers |
| $r_{EUR}$, $r_{USD}$ | Risk-free rates |
| $\sigma_S$, $\sigma_X$ | Volatilities |
| $\rho$ | Correlation coefficient |
| $q$ | Gold convenience yield |
| $\tau$ | Settlement time |
| $\tau_{KO}$ | Knock-out time |

# Appendix B: Figure Index

1. **monte_carlo_paths.png** — Sample simulation paths
2. **convergence_analysis.png** — Monte Carlo convergence
3. **scenario_strike.png** — Strike sensitivity analysis
4. **scenario_barrier.png** — Barrier width analysis
5. **greeks_summary.png** — Risk sensitivities
6. **model_comparison.png** — Cross-model validation
