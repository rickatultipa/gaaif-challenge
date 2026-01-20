---
title: "Structured Gold Forward with Knock-Out Barriers"
subtitle: "Product Development Memorandum"
author: "Derivatives Structuring Desk"
date: "January 2026"
subject: "Pricing and Risk Analysis — Zeus Gold Group Hedging Facility"
---

\newpage

**CONFIDENTIAL**

**TO:** Product Committee, Alphabank S.A.

**FROM:** Derivatives Structuring Desk

**DATE:** January 2026

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
| Z Group Present Value | EUR −192 million |
| Alphabank Present Value | EUR +192 million |
| Knock-Out Probability | 93% |
| Expected Contract Duration | 5 months |

The negative present value for Z Group reflects the strike price being set 54% above the two-year gold forward. The high knock-out probability stems from the lower barrier's proximity to current spot (2.8% distance) combined with negative EUR/USD drift from interest rate differentials.

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
| Gold spot | $S_0$ | USD 2,750/oz | LBMA Jan 2026 |
| EUR/USD spot | $X_0$ | 1.08 | ECB reference |
| USD risk-free rate | $r_{USD}$ | 4.5% | OIS curve |
| EUR risk-free rate | $r_{EUR}$ | 2.5% | OIS curve |
| Gold volatility | $\sigma_S$ | 18% | 1Y ATM implied |
| EUR/USD volatility | $\sigma_X$ | 8% | 1Y ATM implied |
| Correlation | $\rho$ | −0.25 | 1Y historical |
| Gold convenience yield | $q$ | 0.5% | GOFO proxy |

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
| Z Group Present Value | EUR −191,900,647 |
| Alphabank Present Value | EUR +191,900,647 |
| Standard Error | EUR 123,877 |
| 95% Confidence Interval | [−192.1M, −191.7M] |

## 4.2 Barrier Analysis

| Metric | Value |
|--------|-------|
| Overall Knock-Out Rate | 92.99% |
| Lower Barrier Breaches | 86.02% |
| Upper Barrier Breaches | 6.97% |
| Average Time to Knock-Out | 0.43 years (5.2 months) |

The asymmetry between barrier breaches reflects the negative EUR/USD drift implied by interest rate parity. With $r_{EUR} - r_{USD} = -2\%$ annually, the euro faces persistent depreciation pressure, making the lower barrier far more likely to be reached.

**[INSERT FIGURE: monte_carlo_paths.png]**

## 4.3 Convergence Verification

Monte Carlo estimates stabilize as path counts increase:

| Paths | Price Estimate | Standard Error |
|-------|---------------|----------------|
| 5,000 | EUR −191.4M | EUR 551K |
| 10,000 | EUR −191.7M | EUR 381K |
| 25,000 | EUR −191.8M | EUR 247K |
| 50,000 | EUR −192.1M | EUR 174K |
| 100,000 | EUR −191.9M | EUR 124K |

Standard errors decay proportionally to $1/\sqrt{n}$, confirming proper convergence behavior.

**[INSERT FIGURE: convergence_analysis.png]**

\newpage

# 5. Critical Assessment

## 5.1 Strike Price Analysis

The specified strike of USD 4,600/oz warrants careful examination.

**Forward Price Calculation:**

$$F_{0,T} = S_0 \cdot e^{(r_{USD} - q) \cdot T} = 2750 \cdot e^{(0.045 - 0.005) \cdot 2} \approx \text{USD } 2{,}979\text{/oz}$$

The strike exceeds the forward by 54%, placing Z Group in a deeply out-of-the-money position:

$$\text{Moneyness} = \frac{F_{0,T}}{K} = \frac{2979}{4600} = 64.8\%$$

**Alternative Strike Analysis:**

| Strike | Forward Relationship | Z Group PV |
|--------|---------------------|------------|
| USD 2,800 | 6% below forward | EUR +2M |
| USD 3,000 | At-the-money | EUR −31M |
| USD 3,500 | 17% above forward | EUR −97M |
| USD 4,600 | 54% above forward | EUR −192M |

**[INSERT FIGURE: scenario_strike.png]**

## 5.2 Barrier Configuration

The lower barrier at 1.05 sits only 2.8% below current spot:

$$\text{Distance to Lower Barrier} = \frac{X_0 - L}{X_0} = \frac{1.08 - 1.05}{1.08} = 2.78\%$$

Given 8% annual EUR/USD volatility and negative drift, barrier breach is near-certain over a two-year horizon.

**Alternative Configurations:**

| Corridor | Knock-Out Rate | Expected Duration |
|----------|---------------|-------------------|
| [1.05, 1.25] | 93% | 5 months |
| [1.00, 1.30] | 66% | 10 months |
| [0.95, 1.35] | 39% | 14 months |

**[INSERT FIGURE: scenario_barrier.png]**

\newpage

# 6. Risk Sensitivities

## 6.1 Greeks Summary

| Greek | Value | Interpretation |
|-------|-------|----------------|
| $\Delta_{gold}$ | EUR 109,545 per USD 1 | First-order gold sensitivity |
| $\Gamma_{gold}$ | EUR −491 | Gold convexity |
| $\Delta_{FX}$ | EUR 2.26M per 0.01 FX | EUR/USD sensitivity |
| $\mathcal{V}_{gold}$ | EUR −691K per 1% vol | Gold vega |
| $\rho_{EUR}$ | EUR −11.7M per 1bp | EUR rate sensitivity |

**[INSERT FIGURE: greeks_summary.png]**

## 6.2 Hedging Implications

**Delta Hedging:** The gold delta of EUR 110K per dollar implies a hedge ratio of approximately:

$$\text{Gold Hedge} = \frac{\Delta_{gold}}{S_0} \times K = \frac{109{,}545}{2750} \times 4600 \approx 183{,}000 \text{ oz}$$

**Barrier Risk:** As EUR/USD approaches either barrier, gamma and delta become increasingly unstable—the characteristic "pin risk" of barrier options. Hedging costs will escalate significantly in the final days before a potential knock-out.

\newpage

# 7. Model Validation

## 7.1 Alternative Specifications

To ensure robustness, we compared valuations across three model specifications:

| Model | Z Group PV | Knock-Out Rate |
|-------|-----------|----------------|
| Base GBM | EUR −192.1M | 92.9% |
| Heston Stochastic Vol | EUR −191.8M | 93.0% |
| Merton Jump-Diffusion | EUR −191.7M | 93.0% |

All models converge within 0.2%, confirming that the barrier structure dominates pricing dynamics. Model specification risk is secondary.

**[INSERT FIGURE: model_comparison.png]**

## 7.2 Analytical Benchmark

The vanilla gold forward (without barriers) provides a sanity check:

$$V_{vanilla} = e^{-r_{EUR} \cdot T} \cdot N \cdot \frac{F_{0,T} - K}{K} = e^{-0.025 \times 2} \cdot 500M \cdot \frac{2979 - 4600}{4600}$$

$$V_{vanilla} = \text{EUR } -167{,}598{,}411$$

The knock-out version (EUR −192M) is EUR 24M worse than the vanilla, representing the expected cost of early termination when gold is below strike.

\newpage

# 8. Conclusions and Recommendations

## 8.1 Summary of Findings

The proposed structure is technically sound and priceable using standard Monte Carlo techniques. However, two features merit discussion:

1. **Strike positioning:** The USD 4,600 strike creates a deeply out-of-the-money position for Z Group. Clarification of the commercial rationale is recommended.

2. **Barrier proximity:** The 93% knock-out probability results in an expected contract life of only 5 months—potentially misaligned with a 2-year hedging mandate.

## 8.2 Recommendations

We recommend proceeding to term sheet stage contingent upon:

1. Confirmation from Relationship Management regarding Z Group's acceptance of the strike level and its implications

2. Discussion of whether alternative barrier configurations (e.g., [1.00, 1.30]) would better serve the client's hedging objectives

3. Documentation of appropriate risk disclosures regarding the high knock-out probability

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
