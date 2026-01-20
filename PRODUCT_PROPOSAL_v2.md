# Structured Gold Forward with Knock-Out Barriers
## Product Development Memorandum

**To:** Product Committee, Alphabank S.A.
**From:** Derivatives Structuring Desk
**Date:** January 2026
**Re:** Pricing and Risk Analysis — Zeus Gold Group Hedging Facility

---

### Introduction

This memorandum presents our analysis of a proposed structured hedging facility for Zeus Gold Group AG. The product combines exposure to LBMA gold prices with automatic termination features linked to EUR/USD exchange rate movements. We have developed a comprehensive pricing framework, validated our approach through multiple methodologies, and identified several structural considerations that warrant discussion before proceeding to term sheet finalization.

The core economics of this transaction merit careful examination. As currently specified, the contract parameters create a position that is substantially out-of-the-money from Z Group's perspective, with a high probability of early termination. While such structures are not uncommon in the market—particularly when clients seek zero-premium solutions or are expressing specific directional views—we believe the Committee should understand these dynamics before approval.

---

### 1. Transaction Overview

Zeus Gold Group, a Frankfurt-headquartered jewelry manufacturer, seeks to hedge its USD-denominated gold procurement costs while managing EUR/USD translation risk. The proposed facility would run for two years commencing March 2026.

**Principal Terms**

The notional amount is EUR 500 million, referenced against LBMA Gold PM fixing prices. Settlement occurs either at the scheduled maturity in February 2028, or upon early termination triggered by EUR/USD barrier breach. The strike price is set at USD 4,600 per troy ounce.

The knock-out feature terminates the contract immediately if the European Central Bank's EUR/USD reference rate prints below 1.05 or above 1.25 at any point during the life of the trade. Upon settlement (whether at maturity or knock-out), the payoff to Z Group equals the notional multiplied by the percentage deviation of the fixing price from strike. Alphabank receives the mirror-image payoff.

**Payoff Mechanics**

At settlement time τ, with gold fixing at price P:

- Z Group receives: EUR 500mm × (P − 4,600) / 4,600
- Alphabank receives: EUR 500mm × (4,600 − P) / 4,600

The structure is economically equivalent to a gold forward from Z Group's perspective, with the caveat that the contract may terminate early if currency markets move beyond the specified corridor.

---

### 2. Market Context and Parameter Selection

Before presenting our pricing analysis, we establish the market data underlying our valuations. These parameters reflect conditions as of mid-January 2026 and should be refreshed at execution.

**Spot References**

Gold currently trades near USD 2,750 per ounce, having consolidated after reaching all-time highs in late 2025. The EUR/USD rate stands at approximately 1.08, roughly mid-way between the proposed knock-out barriers.

**Interest Rate Environment**

The transatlantic rate differential remains substantial. ECB deposit facility rates hover around 2.5% following the gradual normalization cycle, while the Federal Reserve maintains the fed funds target near 4.5%. This 200 basis point differential implies persistent depreciation pressure on EUR/USD under covered interest parity—a critical consideration for barrier analysis that we address below.

**Volatility and Correlation Estimates**

One-year at-the-money implied volatilities indicate approximately 18% for gold and 8% for EUR/USD. These levels are consistent with recent realized volatility, though we note that gold volatility in particular can spike significantly during risk-off episodes.

The correlation between gold returns and EUR/USD movements has historically been negative, typically ranging from −0.15 to −0.35. This relationship reflects gold's role as a dollar-denominated safe haven: when the dollar strengthens (EUR/USD falls), gold often rises in response to the same flight-to-quality dynamics. We use −0.25 as our central estimate, though we examine sensitivity to this parameter given its importance to the joint dynamics.

---

### 3. Pricing Framework

The presence of path-dependent barriers on a correlated two-asset structure precludes closed-form solutions. We therefore employ Monte Carlo simulation under the risk-neutral measure, with careful attention to numerical accuracy and variance reduction.

**Stochastic Model Specification**

Both gold and EUR/USD follow geometric Brownian motion with drift terms determined by no-arbitrage conditions. For gold, the risk-neutral drift equals the dollar funding rate less any convenience yield or lease rate (approximately 50 basis points). For EUR/USD, interest rate parity dictates a drift equal to the EUR-USD rate differential.

The correlation structure is implemented via Cholesky decomposition of the covariance matrix. Specifically, if Z₁ and Z₂ are independent standard normals, the correlated increments are constructed as:

W₁ = Z₁
W₂ = ρZ₁ + √(1−ρ²)Z₂

This preserves the marginal distributions while inducing the desired dependence.

**Barrier Monitoring**

The knock-out feature requires continuous monitoring of EUR/USD against both barriers. In practice, we discretize time into 504 steps (approximately daily over two years), checking the barrier condition at each point. This introduces modest discretization bias—true continuous monitoring would yield slightly higher knock-out probabilities—but the effect is small relative to other uncertainties.

When a barrier breach occurs, we record both the time and the contemporaneous gold price, as the latter determines the settlement amount.

**Variance Reduction Techniques**

Raw Monte Carlo estimates converge slowly, with standard errors declining only as the inverse square root of the path count. We implement two variance reduction methods to improve efficiency.

First, antithetic variates: for each random path, we also simulate its reflection (replacing all random draws with their negatives). The two paths are negatively correlated, so averaging them reduces variance substantially.

Second, a control variate based on the vanilla gold forward. The barrier-free forward has a known analytical price, so any deviation between the Monte Carlo estimate and this benchmark indicates sampling error that can be partially corrected in the exotic price estimate.

Together, these techniques reduce our standard errors by roughly a factor of three compared to naive simulation.

---

### 4. Valuation Results

We present results based on 100,000 simulation paths, which yields standard errors below EUR 150,000—adequate precision for indicative pricing purposes. Final execution pricing would use larger samples.

**Base Case Valuation**

Under current market conditions and the specified contract terms, the present value to Z Group is approximately **negative EUR 192 million**, with Alphabank holding the corresponding positive position.

The 95% confidence interval spans roughly EUR 191.7 million to EUR 192.1 million, confirming that Monte Carlo noise is not materially affecting our conclusions.

**Barrier Breach Analysis**

The knock-out probability is strikingly high: approximately **93% of simulated paths** terminate early due to barrier breach. Of these early terminations, the vast majority (86 percentage points) result from EUR/USD falling below 1.05, with only 7 percentage points attributable to upper barrier breaches.

This asymmetry is not surprising given the interest rate environment. With EUR rates 200 basis points below USD rates, covered interest parity implies steady depreciation pressure on the euro. Starting from spot of 1.08, the lower barrier at 1.05 is only 2.8% away—easily within reach given 8% annual volatility, even before accounting for the negative drift.

The average time to knock-out, conditional on breach occurring, is approximately **5 months**. This short expected duration has significant implications for both hedging and credit risk.

**[INSERT FIGURE: output/monte_carlo_paths.png — Sample simulation paths illustrating the high frequency of early termination via lower barrier breach]**

---

### 5. Critical Assessment of Contract Parameters

Our analysis reveals two structural features that warrant explicit discussion with the Committee.

**The Strike Price and Forward Relationship**

At USD 4,600 per ounce, the strike sits **54% above** the two-year gold forward price of approximately USD 2,980. In options terminology, this contract is deeply out-of-the-money from Z Group's perspective.

To place this in context: gold has never traded above USD 2,900 in history. For Z Group to receive a positive payoff at maturity, gold would need to appreciate by roughly 67% from current levels within two years—an outcome that, while not impossible, lies well into the tail of any reasonable probability distribution.

The table below illustrates how valuation changes across alternative strike levels:

| Strike (USD/oz) | Relationship to Forward | Z Group Present Value |
|-----------------|------------------------|----------------------|
| 2,800 | 6% discount | +EUR 2 million |
| 3,000 | At-the-money | −EUR 31 million |
| 3,500 | 17% premium | −EUR 97 million |
| 4,600 (specified) | 54% premium | −EUR 192 million |

A strike near USD 3,000 would represent fair value for both parties in the absence of barriers. The specified USD 4,600 strike implies Z Group is effectively paying Alphabank a substantial premium—potentially appropriate if this is structured as a zero-cost collar or if Z Group holds a strong bullish view, but worth confirming.

**Barrier Proximity and Knock-Out Probability**

The lower barrier at 1.05 sits only 2.8% below current spot. Combined with the negative EUR/USD drift and 8% volatility, this creates near-certainty of knock-out over a two-year horizon.

We examined alternative barrier configurations:

| Barrier Corridor | Knock-Out Rate | Average Duration |
|-----------------|----------------|------------------|
| [1.05, 1.25] specified | 93% | 5 months |
| [1.00, 1.30] wider | 66% | 10 months |
| [0.95, 1.35] much wider | 39% | 14 months |

Wider barriers dramatically extend expected contract duration and reduce the probability of early termination. Whether this is desirable depends on the client's objectives—some prefer the "escape valve" of likely knock-out, while others want sustained hedge protection.

**[INSERT FIGURE: output/scenario_strike.png — Valuation sensitivity to strike price]**

**[INSERT FIGURE: output/scenario_barrier.png — Knock-out probability across barrier configurations]**

---

### 6. Model Validation and Robustness

Given the materiality of this transaction, we validated our primary pricing model through several approaches.

**Convergence Analysis**

Monte Carlo estimates should stabilize as path counts increase, with standard errors declining proportionally to 1/√n. Our convergence tests confirm this behavior:

| Paths | Price Estimate | Standard Error | Change from Prior |
|-------|---------------|----------------|-------------------|
| 5,000 | −EUR 191.4mm | EUR 551k | — |
| 20,000 | −EUR 192.0mm | EUR 276k | EUR 0.6mm |
| 50,000 | −EUR 192.1mm | EUR 174k | EUR 0.1mm |
| 100,000 | −EUR 191.9mm | EUR 124k | EUR 0.2mm |

The price estimate stabilizes around EUR −192 million, with diminishing changes as we add paths. This provides confidence that we have achieved adequate convergence.

**[INSERT FIGURE: output/convergence_analysis.png — Monte Carlo convergence demonstrating stability]**

**Alternative Model Specifications**

We re-priced the structure using two alternative stochastic models for gold dynamics:

The *Heston stochastic volatility model* allows gold volatility itself to fluctuate randomly, capturing the volatility clustering observed in commodity markets. Under this specification, we obtain a Z Group value of EUR −191.8 million.

The *Merton jump-diffusion model* augments standard Brownian motion with occasional discrete jumps, representing sudden price dislocations. This yields EUR −191.7 million.

Both alternative models produce valuations within 0.2% of our base GBM estimate. This convergence across methodologies is reassuring: the dominant factor driving valuation is the barrier structure, which affects all models similarly. The choice of diffusion specification is second-order.

**[INSERT FIGURE: output/model_comparison.png — Cross-model validation results]**

**Analytical Benchmarks**

As a final check, we computed the analytical value of a vanilla gold forward (without barriers) using the same parameters. This came to EUR −168 million, compared to EUR −192 million for the knock-out version.

The EUR 24 million difference represents the expected cost of early termination—predominantly knock-outs occurring when gold is below strike, crystallizing losses earlier than the full two-year horizon would. This directional relationship (barrier version worth less than vanilla for the party who is out-of-the-money) accords with financial intuition.

---

### 7. Risk Sensitivities

Alphabank will need to hedge this exposure dynamically. The key risk measures are summarized below.

**Gold Price Sensitivity (Delta)**

The position exhibits delta of approximately EUR 110,000 per USD 1 move in gold. This is positive for Z Group, negative for Alphabank—intuitive given the forward-like payoff structure. Gamma is modest at current spot levels but would increase significantly if gold approached the strike.

**EUR/USD Sensitivity**

The FX delta is substantial: roughly EUR 2.3 million per 0.01 move in EUR/USD. This sensitivity reflects two channels. First, EUR/USD movements affect knock-out timing. Second, because the payoff is EUR-denominated while gold is USD-priced, there is embedded translation exposure.

Notably, FX delta becomes highly unstable as EUR/USD approaches either barrier, exhibiting the "pin risk" characteristic of barrier options.

**Volatility Exposure (Vega)**

Gold vega is approximately EUR −690,000 per 1% increase in implied volatility. The negative sign indicates that higher gold volatility marginally reduces value to Z Group under current conditions—somewhat counterintuitive until one recognizes that the position is deeply out-of-the-money, and higher volatility primarily increases the probability of adverse outcomes.

**Correlation Sensitivity**

A 5 percentage point increase in gold-EURUSD correlation changes value by approximately EUR 590,000. More positive correlation (gold and EUR/USD moving together) modestly benefits Z Group, as it reduces the probability of the adverse combination where EUR/USD falls (triggering knock-out) while gold remains low.

**[INSERT FIGURE: output/greeks_summary.png — Summary of risk sensitivities]**

---

### 8. Stress Testing

We examined valuation under stressed market conditions:

**Volatility Spike Scenarios**

Doubling gold volatility to 36% shifts valuation by approximately EUR 15 million. Doubling FX volatility to 16% has a larger impact, reducing value to Z Group by EUR 25 million as knock-out becomes even more certain.

**Correlation Breakdown**

If correlation shifted from −0.25 to zero, the impact is modest (roughly EUR 5 million). This suggests the structure is not heavily dependent on correlation assumptions.

**Spot Moves**

A 10% gold rally improves Z Group's position by EUR 55 million, while a 10% decline worsens it by EUR 50 million—roughly symmetric as expected for a forward-like payoff.

Moving EUR/USD closer to either barrier accelerates expected knock-out and modestly reduces value to Z Group.

---

### 9. Implementation Considerations

**Hedging Requirements**

Alphabank would need to establish:
- Gold delta hedge via futures or forwards, rebalanced frequently given gamma exposure
- EUR/USD delta hedge, with particular attention as spot approaches barriers
- Vega hedge through gold options if the desk has limits on volatility exposure

Barrier proximity will require careful monitoring. As EUR/USD nears 1.05, delta and gamma become increasingly unstable, and hedging costs will rise.

**Margin and Credit**

Given the negative mark-to-market for Z Group, credit exposure runs from Alphabank to the client. Standard ISDA documentation with appropriate thresholds should apply. The high knock-out probability actually mitigates long-dated credit risk, as most scenarios terminate within six months.

**Documentation**

The knock-out trigger should reference a specific ECB fixing time to avoid disputes. Settlement mechanics following knock-out need clear specification—we recommend T+2 settlement with the gold price referenced to the LBMA PM fixing on the knock-out date.

---

### 10. Conclusions and Recommendations

Our analysis supports the following conclusions:

First, the structure is priceable and hedgeable using standard techniques. Monte Carlo simulation under correlated GBM provides robust valuations that are confirmed by alternative model specifications.

Second, the specified parameters create a transaction that is substantially favorable to Alphabank. Whether this reflects Z Group's genuine market view, a zero-premium structure where the strike premium offsets option value, or a potential specification error merits clarification.

Third, knock-out is near-certain under current conditions. The Committee should consider whether this aligns with the stated hedging objectives, or whether alternative barrier configurations would better serve the client relationship.

We recommend proceeding to term sheet stage contingent on confirmation from Relationship Management regarding the commercial rationale for the current strike level. If the USD 4,600 strike is intentional, we suggest documenting the client's acknowledgment that this represents an out-of-the-money position with limited probability of positive payoff.

---

### Appendix A: Mathematical Specification

**Asset Dynamics (Risk-Neutral Measure)**

Gold price S and EUR/USD rate X evolve according to:

dS/S = (r_USD − q) dt + σ_S dW^S

dX/X = (r_EUR − r_USD) dt + σ_X dW^X

with instantaneous correlation dW^S · dW^X = ρ dt.

**Knock-Out Condition**

Define the stopping time τ_KO = inf{t : X_t ∉ (L, U)} where L = 1.05 and U = 1.25.

**Settlement**

At time τ = min(T, τ_KO), payoffs are:

V_ZGroup = N · (S_τ − K) / K

V_ABank = −V_ZGroup

**Present Value**

The risk-neutral valuation is:

PV = E^Q[ e^{−r_EUR · τ} · V ]

computed via Monte Carlo integration.

---

### Appendix B: Parameter Summary

| Parameter | Value | Source |
|-----------|-------|--------|
| Gold spot | USD 2,750/oz | LBMA, Jan 2026 |
| EUR/USD spot | 1.08 | ECB reference |
| EUR risk-free rate | 2.5% | OIS curve |
| USD risk-free rate | 4.5% | OIS curve |
| Gold volatility | 18% | 1Y ATM implied |
| EUR/USD volatility | 8% | 1Y ATM implied |
| Correlation | −0.25 | 1Y historical |
| Gold convenience yield | 0.5% | GOFO proxy |

---

*End of Memorandum*
