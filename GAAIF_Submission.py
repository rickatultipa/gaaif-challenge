#!/usr/bin/env python3
"""
================================================================================
GAAIF CHALLENGE SUBMISSION - STRUCTURED FORWARD WITH DOUBLE KNOCK-OUT BARRIER
================================================================================

Product: Gold Forward Contract with EUR/USD Double Knock-Out Barriers
Issuer: Alphabank S.A.
Client: Zeus Gold Group AG

Contract Specifications:
- Notional Principal: EUR 500 Million
- Strike Price (K): $4,600/oz (LBMA Gold Spot)
- Tenor: 2 years (March 1, 2026 - February 28, 2028)
- Lower Barrier: EUR/USD = 1.05 (Knock-Out)
- Upper Barrier: EUR/USD = 1.25 (Knock-Out)
- Settlement: Z Group: N × (P - K) / K; A Bank: N × (K - P) / K

Mathematical Framework:
- Two-factor correlated Geometric Brownian Motion (GBM)
- Risk-neutral pricing with appropriate measure
- Monte Carlo simulation with variance reduction techniques

Author: GAAIF Challenge Submission
Date: January 2026
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarketData:
    """
    Container for market data parameters.

    All rates are annualized and continuously compounded.
    Volatilities are annualized (sqrt(252) scaling for daily).
    """
    # Spot prices
    gold_spot: float = 2750.0       # Current LBMA gold price (USD/oz)
    eurusd_spot: float = 1.08       # Current EUR/USD exchange rate

    # Risk-free interest rates (continuous compounding)
    r_eur: float = 0.025            # EUR risk-free rate (~2.5% ECB)
    r_usd: float = 0.045            # USD risk-free rate (~4.5% Fed)

    # Implied volatilities (annualized)
    sigma_gold: float = 0.18        # Gold volatility (~18%)
    sigma_eurusd: float = 0.08      # EUR/USD volatility (~8%)

    # Asset correlation
    rho: float = -0.25              # Gold-EURUSD correlation (typically negative)

    # Gold convenience yield / lease rate
    gold_yield: float = 0.005       # ~0.5% convenience yield


@dataclass
class ContractTerms:
    """
    Container for contract specifications.

    All barrier monitoring is assumed continuous (European style early termination).
    """
    notional: float = 500_000_000   # EUR 500 Million notional
    strike: float = 4600.0          # Gold strike price (USD/oz)
    tenor: float = 2.0              # Contract tenor (years)
    barrier_lower: float = 1.05     # Lower EUR/USD knock-out barrier
    barrier_upper: float = 1.25     # Upper EUR/USD knock-out barrier


# =============================================================================
# CORRELATED GBM SIMULATOR
# =============================================================================

class CorrelatedGBMSimulator:
    """
    Simulates correlated Geometric Brownian Motion paths for Gold and EUR/USD.

    Mathematical Model:
    ------------------
    Under the risk-neutral measure Q (EUR numeraire):

    Gold (in USD) with QUANTO ADJUSTMENT:
        dS_t / S_t = (r_USD - q - ρ × σ_S × σ_X) dt + σ_S dW^S_t

        The quanto adjustment (-ρ × σ_S × σ_X) accounts for the fact that
        the underlying (gold) is denominated in USD but the payoff is in EUR.
        With ρ < 0, this term is positive, slightly increasing gold's drift.

    EUR/USD:
        dX_t / X_t = (r_EUR - r_USD) dt + σ_X dW^X_t

    Correlation structure:
        dW^S_t × dW^X_t = ρ dt

    Implementation uses Cholesky decomposition for correlated Brownian motions
    and exact log-normal simulation for numerical stability.
    """

    def __init__(self, market: MarketData, contract: ContractTerms,
                 seed: Optional[int] = None, use_quanto_adjustment: bool = True):
        """
        Initialize simulator with market data and contract terms.

        Args:
            market: MarketData object with spot prices, rates, and vols
            contract: ContractTerms object with product specifications
            seed: Random seed for reproducibility
            use_quanto_adjustment: Apply quanto drift adjustment (default True)
        """
        self.market = market
        self.contract = contract
        self.rng = np.random.default_rng(seed)
        self.use_quanto_adjustment = use_quanto_adjustment

    def _generate_correlated_normals(self, n_paths: int,
                                     n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate correlated standard normal random variables.

        Uses Cholesky decomposition:
        [W1]   [1         0        ] [Z1]
        [W2] = [ρ    √(1-ρ²)] [Z2]

        where Z1, Z2 are independent standard normals.

        Args:
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps

        Returns:
            Tuple of (W1, W2) correlated normal arrays
        """
        # Generate independent standard normals
        Z1 = self.rng.standard_normal((n_paths, n_steps))
        Z2 = self.rng.standard_normal((n_paths, n_steps))

        # Apply Cholesky correlation structure
        rho = self.market.rho
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        return W1, W2

    def simulate_paths(self, n_paths: int, n_steps: int,
                       antithetic: bool = True) -> Dict[str, np.ndarray]:
        """
        Simulate correlated Gold and EUR/USD price paths.

        Uses exact log-normal simulation:
        S_{t+dt} = S_t × exp[(μ - σ²/2)dt + σ√dt × W]

        Args:
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps
            antithetic: Use antithetic variates for variance reduction

        Returns:
            Dictionary containing:
            - gold_paths: Array of gold price paths [n_paths × (n_steps+1)]
            - eurusd_paths: Array of EUR/USD paths [n_paths × (n_steps+1)]
            - times: Time grid array
            - dt: Time step size
        """
        dt = self.contract.tenor / n_steps
        sqrt_dt = np.sqrt(dt)

        # For antithetic variates, generate half and mirror
        actual_paths = n_paths // 2 if antithetic else n_paths

        W1, W2 = self._generate_correlated_normals(actual_paths, n_steps)

        if antithetic:
            W1 = np.vstack([W1, -W1])
            W2 = np.vstack([W2, -W2])
            actual_paths = n_paths

        # Initialize path arrays
        gold_paths = np.zeros((actual_paths, n_steps + 1))
        eurusd_paths = np.zeros((actual_paths, n_steps + 1))

        gold_paths[:, 0] = self.market.gold_spot
        eurusd_paths[:, 0] = self.market.eurusd_spot

        # Risk-neutral drift terms
        # Gold: μ_S = r_USD - q - ρ × σ_S × σ_X (quanto adjustment for EUR payoff)
        # The quanto adjustment accounts for correlation between gold and EUR/USD
        # when the underlying is in USD but the payoff currency is EUR
        gold_drift = self.market.r_usd - self.market.gold_yield
        if self.use_quanto_adjustment:
            quanto_adjustment = self.market.rho * self.market.sigma_gold * self.market.sigma_eurusd
            gold_drift -= quanto_adjustment  # With ρ < 0, this increases drift
        # EUR/USD: μ_X = r_EUR - r_USD (interest rate parity)
        eurusd_drift = self.market.r_eur - self.market.r_usd

        # Simulate using exact log-normal dynamics
        for t in range(n_steps):
            # Gold price evolution
            gold_paths[:, t+1] = gold_paths[:, t] * np.exp(
                (gold_drift - 0.5 * self.market.sigma_gold**2) * dt +
                self.market.sigma_gold * sqrt_dt * W1[:, t]
            )

            # EUR/USD evolution
            eurusd_paths[:, t+1] = eurusd_paths[:, t] * np.exp(
                (eurusd_drift - 0.5 * self.market.sigma_eurusd**2) * dt +
                self.market.sigma_eurusd * sqrt_dt * W2[:, t]
            )

        times = np.linspace(0, self.contract.tenor, n_steps + 1)

        return {
            'gold_paths': gold_paths,
            'eurusd_paths': eurusd_paths,
            'times': times,
            'dt': dt
        }

    def check_barrier_breach(self,
                             eurusd_paths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check for barrier breaches in EUR/USD paths.

        The product features double knock-out barriers:
        - Lower barrier: EUR/USD < 1.05 → Knock-out
        - Upper barrier: EUR/USD > 1.25 → Knock-out

        Args:
            eurusd_paths: Array of EUR/USD paths

        Returns:
            Tuple of:
            - knocked_out: Boolean array indicating knocked-out paths
            - knockout_idx: Index of first barrier breach (-1 if no breach)
        """
        n_paths, n_steps = eurusd_paths.shape

        # Check barrier conditions
        below_lower = eurusd_paths < self.contract.barrier_lower
        above_upper = eurusd_paths > self.contract.barrier_upper
        breached = below_lower | above_upper

        # Find first breach time for each path
        knocked_out = np.any(breached, axis=1)
        knockout_idx = np.full(n_paths, -1, dtype=int)

        for i in range(n_paths):
            if knocked_out[i]:
                knockout_idx[i] = np.argmax(breached[i])

        return knocked_out, knockout_idx


# =============================================================================
# STRUCTURED FORWARD PRICER
# =============================================================================

class StructuredForwardPricer:
    """
    Prices the structured gold forward with double knock-out barriers.

    Product Payoff Structure:
    ------------------------
    At settlement time τ (maturity T or knock-out time, whichever comes first):

    Z Group's Payoff (EUR): N × (P_τ - K) / K
    A Bank's Payoff (EUR):  N × (K - P_τ) / K

    where:
    - N = Notional Principal (EUR 500M)
    - P_τ = LBMA Gold Spot Price at settlement (USD/oz)
    - K = Strike Benchmark Price ($4,600/oz)

    The knock-out feature terminates the contract if EUR/USD breaches
    either the lower (1.05) or upper (1.25) barrier.

    Pricing Methodology:
    -------------------
    Risk-neutral expectation under the EUR measure:

    V_0 = E^Q[e^{-r_EUR × τ} × Payoff]

    Computed via Monte Carlo simulation with:
    - Antithetic variates
    - Control variate (vanilla forward)
    """

    def __init__(self, market: MarketData, contract: ContractTerms):
        """
        Initialize pricer with market data and contract terms.

        Args:
            market: MarketData object
            contract: ContractTerms object
        """
        self.market = market
        self.contract = contract

    def price_monte_carlo(self, n_paths: int = 100000, n_steps: int = 504,
                          seed: Optional[int] = 42, antithetic: bool = True,
                          control_variate: bool = True,
                          use_quanto_adjustment: bool = True) -> Dict[str, Any]:
        """
        Price the structured product using Monte Carlo simulation.

        Implements variance reduction via:
        1. Antithetic variates: Uses (W, -W) pairs
        2. Control variate: Vanilla gold forward with known analytical price

        Args:
            n_paths: Number of simulation paths
            n_steps: Number of time steps (504 ≈ 252 trading days × 2 years)
            seed: Random seed for reproducibility
            antithetic: Enable antithetic variates
            control_variate: Enable control variate
            use_quanto_adjustment: Apply quanto drift adjustment for EUR payoff

        Returns:
            Dictionary with:
            - price_zgroup: Present value for Z Group (EUR)
            - price_abank: Present value for A Bank (EUR)
            - std_error: Monte Carlo standard error
            - ci_95_lower/upper: 95% confidence interval bounds
            - knockout_rate: Probability of barrier breach
            - avg_knockout_time: Average time to knockout (if breached)
            - lower_breach_rate: Rate of lower barrier breaches
            - upper_breach_rate: Rate of upper barrier breaches
            - Settlement data arrays for further analysis
        """
        simulator = CorrelatedGBMSimulator(self.market, self.contract, seed, use_quanto_adjustment)

        # Generate paths
        sim_result = simulator.simulate_paths(n_paths, n_steps, antithetic)
        gold_paths = sim_result['gold_paths']
        eurusd_paths = sim_result['eurusd_paths']
        times = sim_result['times']

        # Check barrier breaches
        knocked_out, knockout_idx = simulator.check_barrier_breach(eurusd_paths)

        # Determine settlement prices and times
        n_actual_paths = gold_paths.shape[0]
        settlement_prices = np.zeros(n_actual_paths)
        settlement_times = np.zeros(n_actual_paths)

        for i in range(n_actual_paths):
            if knocked_out[i]:
                # Contract knocked out - settle at knockout time
                idx = knockout_idx[i]
                settlement_prices[i] = gold_paths[i, idx]
                settlement_times[i] = times[idx]
            else:
                # No knockout - settle at maturity
                settlement_prices[i] = gold_paths[i, -1]
                settlement_times[i] = self.contract.tenor

        # Calculate payoffs (from Z Group's perspective)
        K = self.contract.strike
        N = self.contract.notional
        payoffs_zgroup = N * (settlement_prices - K) / K
        payoffs_abank = -payoffs_zgroup

        # Discount to present value using EUR rate
        discount_factors = np.exp(-self.market.r_eur * settlement_times)
        pv_zgroup = payoffs_zgroup * discount_factors
        pv_abank = payoffs_abank * discount_factors

        # Control variate adjustment
        cv_adjustment = 0.0
        if control_variate:
            # Vanilla forward price (analytical)
            forward_price = self.market.gold_spot * np.exp(
                (self.market.r_usd - self.market.gold_yield) * self.contract.tenor
            )

            # Vanilla forward payoffs from simulation
            vanilla_payoffs = N * (gold_paths[:, -1] - K) / K
            vanilla_pv = vanilla_payoffs * np.exp(-self.market.r_eur * self.contract.tenor)

            # Analytical vanilla forward PV
            analytical_vanilla = N * (forward_price - K) / K * np.exp(
                -self.market.r_eur * self.contract.tenor
            )

            # Optimal control variate coefficient (minimizes variance)
            cov_matrix = np.cov(pv_zgroup, vanilla_pv)
            if cov_matrix[1, 1] > 0:
                beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                cv_adjustment = beta * (analytical_vanilla - np.mean(vanilla_pv))

        # Compute statistics
        mean_pv_zgroup = np.mean(pv_zgroup) + cv_adjustment
        mean_pv_abank = np.mean(pv_abank) - cv_adjustment
        std_pv = np.std(pv_zgroup)
        se = std_pv / np.sqrt(n_actual_paths)

        # 95% confidence interval
        ci_95 = (mean_pv_zgroup - 1.96 * se, mean_pv_zgroup + 1.96 * se)

        # Knockout statistics
        knockout_rate = np.mean(knocked_out)
        avg_knockout_time = np.mean(settlement_times[knocked_out]) if np.any(knocked_out) else np.nan

        # Barrier breach breakdown
        if np.any(knocked_out):
            ko_eurusd = eurusd_paths[knocked_out, :]
            ko_idx = knockout_idx[knocked_out]
            ko_prices = np.array([ko_eurusd[i, ko_idx[i]] for i in range(len(ko_idx))])
            lower_breach_rate = np.mean(ko_prices < self.contract.barrier_lower)
            upper_breach_rate = np.mean(ko_prices > self.contract.barrier_upper)
        else:
            lower_breach_rate = 0.0
            upper_breach_rate = 0.0

        return {
            'price_zgroup': mean_pv_zgroup,
            'price_abank': mean_pv_abank,
            'std_error': se,
            'ci_95_lower': ci_95[0],
            'ci_95_upper': ci_95[1],
            'knockout_rate': knockout_rate,
            'avg_knockout_time': avg_knockout_time,
            'lower_breach_rate': lower_breach_rate * knockout_rate,
            'upper_breach_rate': upper_breach_rate * knockout_rate,
            'n_paths': n_actual_paths,
            'n_steps': n_steps,
            'gold_paths': gold_paths,
            'eurusd_paths': eurusd_paths,
            'settlement_prices': settlement_prices,
            'settlement_times': settlement_times,
            'knocked_out': knocked_out
        }

    def compute_greeks(self, base_result: Dict, bump_size: float = 0.01,
                       n_paths: int = 50000, seed: int = 42) -> Dict[str, float]:
        """
        Compute option Greeks using finite difference method.

        Greeks Computed:
        ---------------
        - Delta (Gold): ∂V/∂S_gold
        - Gamma (Gold): ∂²V/∂S²_gold
        - Delta (EUR/USD): ∂V/∂X_eurusd
        - Vega (Gold): ∂V/∂σ_gold
        - Rho (EUR): ∂V/∂r_EUR
        - Correlation Sensitivity: ∂V/∂ρ

        All computed via central difference where applicable.

        Args:
            base_result: Pricing result dictionary
            bump_size: Relative bump size for finite difference
            n_paths: Number of paths for Greek calculation
            seed: Random seed

        Returns:
            Dictionary of Greek values
        """
        base_price = base_result['price_zgroup']

        # Delta (Gold) - central difference
        market_up = MarketData(
            gold_spot=self.market.gold_spot * (1 + bump_size),
            eurusd_spot=self.market.eurusd_spot,
            r_eur=self.market.r_eur, r_usd=self.market.r_usd,
            sigma_gold=self.market.sigma_gold, sigma_eurusd=self.market.sigma_eurusd,
            rho=self.market.rho, gold_yield=self.market.gold_yield
        )
        market_down = MarketData(
            gold_spot=self.market.gold_spot * (1 - bump_size),
            eurusd_spot=self.market.eurusd_spot,
            r_eur=self.market.r_eur, r_usd=self.market.r_usd,
            sigma_gold=self.market.sigma_gold, sigma_eurusd=self.market.sigma_eurusd,
            rho=self.market.rho, gold_yield=self.market.gold_yield
        )

        pricer_up = StructuredForwardPricer(market_up, self.contract)
        pricer_down = StructuredForwardPricer(market_down, self.contract)

        price_up = pricer_up.price_monte_carlo(n_paths, seed=seed)['price_zgroup']
        price_down = pricer_down.price_monte_carlo(n_paths, seed=seed)['price_zgroup']

        delta_gold = (price_up - price_down) / (2 * bump_size * self.market.gold_spot)
        gamma_gold = (price_up - 2 * base_price + price_down) / (bump_size * self.market.gold_spot)**2

        # Delta (EUR/USD)
        market_up_fx = MarketData(
            gold_spot=self.market.gold_spot,
            eurusd_spot=self.market.eurusd_spot * (1 + bump_size),
            r_eur=self.market.r_eur, r_usd=self.market.r_usd,
            sigma_gold=self.market.sigma_gold, sigma_eurusd=self.market.sigma_eurusd,
            rho=self.market.rho, gold_yield=self.market.gold_yield
        )
        market_down_fx = MarketData(
            gold_spot=self.market.gold_spot,
            eurusd_spot=self.market.eurusd_spot * (1 - bump_size),
            r_eur=self.market.r_eur, r_usd=self.market.r_usd,
            sigma_gold=self.market.sigma_gold, sigma_eurusd=self.market.sigma_eurusd,
            rho=self.market.rho, gold_yield=self.market.gold_yield
        )

        pricer_up_fx = StructuredForwardPricer(market_up_fx, self.contract)
        pricer_down_fx = StructuredForwardPricer(market_down_fx, self.contract)

        price_up_fx = pricer_up_fx.price_monte_carlo(n_paths, seed=seed)['price_zgroup']
        price_down_fx = pricer_down_fx.price_monte_carlo(n_paths, seed=seed)['price_zgroup']

        delta_eurusd = (price_up_fx - price_down_fx) / (2 * bump_size * self.market.eurusd_spot)

        # Vega (Gold volatility)
        vega_bump = 0.01
        market_vega_up = MarketData(
            gold_spot=self.market.gold_spot, eurusd_spot=self.market.eurusd_spot,
            r_eur=self.market.r_eur, r_usd=self.market.r_usd,
            sigma_gold=self.market.sigma_gold + vega_bump,
            sigma_eurusd=self.market.sigma_eurusd,
            rho=self.market.rho, gold_yield=self.market.gold_yield
        )
        market_vega_down = MarketData(
            gold_spot=self.market.gold_spot, eurusd_spot=self.market.eurusd_spot,
            r_eur=self.market.r_eur, r_usd=self.market.r_usd,
            sigma_gold=self.market.sigma_gold - vega_bump,
            sigma_eurusd=self.market.sigma_eurusd,
            rho=self.market.rho, gold_yield=self.market.gold_yield
        )

        pricer_vega_up = StructuredForwardPricer(market_vega_up, self.contract)
        pricer_vega_down = StructuredForwardPricer(market_vega_down, self.contract)

        price_vega_up = pricer_vega_up.price_monte_carlo(n_paths, seed=seed)['price_zgroup']
        price_vega_down = pricer_vega_down.price_monte_carlo(n_paths, seed=seed)['price_zgroup']

        vega_gold = (price_vega_up - price_vega_down) / (2 * vega_bump)

        # Rho (EUR rate)
        rho_bump = 0.001  # 10 bps
        market_rho_up = MarketData(
            gold_spot=self.market.gold_spot, eurusd_spot=self.market.eurusd_spot,
            r_eur=self.market.r_eur + rho_bump, r_usd=self.market.r_usd,
            sigma_gold=self.market.sigma_gold, sigma_eurusd=self.market.sigma_eurusd,
            rho=self.market.rho, gold_yield=self.market.gold_yield
        )

        pricer_rho_up = StructuredForwardPricer(market_rho_up, self.contract)
        price_rho_up = pricer_rho_up.price_monte_carlo(n_paths, seed=seed)['price_zgroup']

        rho_eur = (price_rho_up - base_price) / rho_bump

        # Correlation sensitivity
        corr_bump = 0.05
        market_corr_up = MarketData(
            gold_spot=self.market.gold_spot, eurusd_spot=self.market.eurusd_spot,
            r_eur=self.market.r_eur, r_usd=self.market.r_usd,
            sigma_gold=self.market.sigma_gold, sigma_eurusd=self.market.sigma_eurusd,
            rho=min(self.market.rho + corr_bump, 0.99), gold_yield=self.market.gold_yield
        )
        market_corr_down = MarketData(
            gold_spot=self.market.gold_spot, eurusd_spot=self.market.eurusd_spot,
            r_eur=self.market.r_eur, r_usd=self.market.r_usd,
            sigma_gold=self.market.sigma_gold, sigma_eurusd=self.market.sigma_eurusd,
            rho=max(self.market.rho - corr_bump, -0.99), gold_yield=self.market.gold_yield
        )

        pricer_corr_up = StructuredForwardPricer(market_corr_up, self.contract)
        pricer_corr_down = StructuredForwardPricer(market_corr_down, self.contract)

        price_corr_up = pricer_corr_up.price_monte_carlo(n_paths, seed=seed)['price_zgroup']
        price_corr_down = pricer_corr_down.price_monte_carlo(n_paths, seed=seed)['price_zgroup']

        corr_sensitivity = (price_corr_up - price_corr_down) / (2 * corr_bump)

        return {
            'delta_gold': delta_gold,
            'gamma_gold': gamma_gold,
            'delta_eurusd': delta_eurusd,
            'vega_gold': vega_gold,
            'rho_eur': rho_eur,
            'correlation_sensitivity': corr_sensitivity
        }


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity_analysis(market: MarketData, contract: ContractTerms,
                             n_paths: int = 50000) -> pd.DataFrame:
    """
    Run comprehensive sensitivity analysis on key model parameters.

    Analyzes sensitivity to:
    - Gold spot price
    - EUR/USD spot rate
    - Gold volatility
    - EUR/USD volatility
    - Asset correlation

    Args:
        market: Base market data
        contract: Contract terms
        n_paths: Number of simulation paths

    Returns:
        DataFrame with sensitivity results
    """
    results = []

    # Gold spot sensitivity
    print("  Analyzing gold spot sensitivity...")
    gold_spots = np.linspace(2400, 3200, 9)
    for gs in gold_spots:
        m = MarketData(gold_spot=gs, eurusd_spot=market.eurusd_spot,
                      r_eur=market.r_eur, r_usd=market.r_usd,
                      sigma_gold=market.sigma_gold, sigma_eurusd=market.sigma_eurusd,
                      rho=market.rho, gold_yield=market.gold_yield)
        pricer = StructuredForwardPricer(m, contract)
        res = pricer.price_monte_carlo(n_paths, seed=42)
        results.append({
            'parameter': 'gold_spot', 'value': gs,
            'price_zgroup': res['price_zgroup'],
            'knockout_rate': res['knockout_rate']
        })

    # EUR/USD spot sensitivity
    print("  Analyzing EUR/USD spot sensitivity...")
    eurusd_spots = np.linspace(1.06, 1.24, 10)
    for fx in eurusd_spots:
        m = MarketData(gold_spot=market.gold_spot, eurusd_spot=fx,
                      r_eur=market.r_eur, r_usd=market.r_usd,
                      sigma_gold=market.sigma_gold, sigma_eurusd=market.sigma_eurusd,
                      rho=market.rho, gold_yield=market.gold_yield)
        pricer = StructuredForwardPricer(m, contract)
        res = pricer.price_monte_carlo(n_paths, seed=42)
        results.append({
            'parameter': 'eurusd_spot', 'value': fx,
            'price_zgroup': res['price_zgroup'],
            'knockout_rate': res['knockout_rate']
        })

    # Gold volatility sensitivity
    print("  Analyzing gold volatility sensitivity...")
    gold_vols = np.linspace(0.10, 0.30, 11)
    for vol in gold_vols:
        m = MarketData(gold_spot=market.gold_spot, eurusd_spot=market.eurusd_spot,
                      r_eur=market.r_eur, r_usd=market.r_usd,
                      sigma_gold=vol, sigma_eurusd=market.sigma_eurusd,
                      rho=market.rho, gold_yield=market.gold_yield)
        pricer = StructuredForwardPricer(m, contract)
        res = pricer.price_monte_carlo(n_paths, seed=42)
        results.append({
            'parameter': 'sigma_gold', 'value': vol,
            'price_zgroup': res['price_zgroup'],
            'knockout_rate': res['knockout_rate']
        })

    # EUR/USD volatility sensitivity
    print("  Analyzing EUR/USD volatility sensitivity...")
    eurusd_vols = np.linspace(0.04, 0.14, 11)
    for vol in eurusd_vols:
        m = MarketData(gold_spot=market.gold_spot, eurusd_spot=market.eurusd_spot,
                      r_eur=market.r_eur, r_usd=market.r_usd,
                      sigma_gold=market.sigma_gold, sigma_eurusd=vol,
                      rho=market.rho, gold_yield=market.gold_yield)
        pricer = StructuredForwardPricer(m, contract)
        res = pricer.price_monte_carlo(n_paths, seed=42)
        results.append({
            'parameter': 'sigma_eurusd', 'value': vol,
            'price_zgroup': res['price_zgroup'],
            'knockout_rate': res['knockout_rate']
        })

    # Correlation sensitivity
    print("  Analyzing correlation sensitivity...")
    correlations = np.linspace(-0.6, 0.4, 11)
    for rho in correlations:
        m = MarketData(gold_spot=market.gold_spot, eurusd_spot=market.eurusd_spot,
                      r_eur=market.r_eur, r_usd=market.r_usd,
                      sigma_gold=market.sigma_gold, sigma_eurusd=market.sigma_eurusd,
                      rho=rho, gold_yield=market.gold_yield)
        pricer = StructuredForwardPricer(m, contract)
        res = pricer.price_monte_carlo(n_paths, seed=42)
        results.append({
            'parameter': 'correlation', 'value': rho,
            'price_zgroup': res['price_zgroup'],
            'knockout_rate': res['knockout_rate']
        })

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(pricing_result: Dict, sensitivity_df: pd.DataFrame,
                          greeks: Dict, contract: ContractTerms,
                          output_dir: str = 'output'):
    """
    Generate all visualizations for the product proposal.

    Creates:
    1. Monte Carlo simulation paths
    2. Payoff distribution
    3. Sensitivity analysis charts
    4. Knockout analysis
    5. Payoff diagram
    6. Greeks summary

    Args:
        pricing_result: Monte Carlo pricing results
        sensitivity_df: Sensitivity analysis DataFrame
        greeks: Greeks dictionary
        contract: Contract terms
        output_dir: Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    times = np.linspace(0, contract.tenor, pricing_result['gold_paths'].shape[1])

    # 1. Monte Carlo Paths
    print("  Generating Monte Carlo paths figure...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    knocked_out = pricing_result['knocked_out']
    gold_paths = pricing_result['gold_paths']
    eurusd_paths = pricing_result['eurusd_paths']

    n_display = 50
    ko_idx = np.where(knocked_out)[0][:n_display//2]
    surv_idx = np.where(~knocked_out)[0][:n_display//2]
    sample_idx = np.concatenate([ko_idx, surv_idx])

    # Gold paths
    ax1 = axes[0]
    for i in sample_idx:
        color = 'red' if knocked_out[i] else 'blue'
        alpha = 0.3 if knocked_out[i] else 0.5
        ax1.plot(times, gold_paths[i], color=color, alpha=alpha, linewidth=0.5)
    ax1.axhline(y=contract.strike, color='green', linestyle='--', linewidth=2)
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Gold Price (USD/oz)')
    ax1.set_title('Gold Price Paths - Monte Carlo Simulation')

    # EUR/USD paths
    ax2 = axes[1]
    for i in sample_idx:
        color = 'red' if knocked_out[i] else 'blue'
        alpha = 0.3 if knocked_out[i] else 0.5
        ax2.plot(times, eurusd_paths[i], color=color, alpha=alpha, linewidth=0.5)
    ax2.axhline(y=contract.barrier_lower, color='darkred', linestyle='--', linewidth=2)
    ax2.axhline(y=contract.barrier_upper, color='darkred', linestyle='--', linewidth=2)
    ax2.fill_between(times, contract.barrier_lower, contract.barrier_upper,
                     color='green', alpha=0.1)
    ax2.set_xlabel('Time (Years)')
    ax2.set_ylabel('EUR/USD Exchange Rate')
    ax2.set_title('EUR/USD Paths with Double Knock-Out Barriers')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/monte_carlo_paths.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Payoff Distribution
    print("  Generating payoff distribution figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    settlement_prices = pricing_result['settlement_prices']
    payoffs = contract.notional * (settlement_prices - contract.strike) / contract.strike

    ax1 = axes[0]
    ax1.hist(payoffs[~knocked_out] / 1e6, bins=50, alpha=0.7, color='blue',
            label='Surviving Paths', density=True)
    ax1.hist(payoffs[knocked_out] / 1e6, bins=50, alpha=0.7, color='red',
            label='Knocked Out Paths', density=True)
    ax1.axvline(x=np.mean(payoffs) / 1e6, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('Payoff (EUR Millions)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Z Group Payoffs')
    ax1.legend()

    ax2 = axes[1]
    ax2.hist(settlement_prices[~knocked_out], bins=50, alpha=0.7, color='blue',
            label='Surviving', density=True)
    ax2.hist(settlement_prices[knocked_out], bins=50, alpha=0.7, color='red',
            label='Knocked Out', density=True)
    ax2.axvline(x=contract.strike, color='green', linestyle='--', linewidth=2)
    ax2.set_xlabel('Gold Price at Settlement (USD/oz)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Settlement Prices')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/payoff_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Payoff Diagram
    print("  Generating payoff diagram...")
    fig, ax = plt.subplots(figsize=(12, 7))

    gold_prices = np.linspace(3500, 6000, 200)
    zgroup_payoff = contract.notional * (gold_prices - contract.strike) / contract.strike
    abank_payoff = -zgroup_payoff

    ax.plot(gold_prices, zgroup_payoff / 1e6, 'b-', linewidth=2.5, label='Z Group Payoff')
    ax.plot(gold_prices, abank_payoff / 1e6, 'r-', linewidth=2.5, label='A Bank Payoff')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=contract.strike, color='green', linestyle='--', linewidth=2,
              label=f'Strike K = ${contract.strike:,.0f}')
    ax.fill_between(gold_prices, 0, zgroup_payoff / 1e6,
                   where=(zgroup_payoff > 0), alpha=0.3, color='blue')
    ax.fill_between(gold_prices, 0, zgroup_payoff / 1e6,
                   where=(zgroup_payoff < 0), alpha=0.3, color='red')

    ax.set_xlabel('Gold Spot Price at Settlement (USD/oz)', fontsize=12)
    ax.set_ylabel('Payoff (EUR Millions)', fontsize=12)
    ax.set_title(f'Payoff Diagram: Structured Gold Forward\n'
                f'Notional = EUR {contract.notional/1e6:.0f}M, Strike = ${contract.strike:,.0f}/oz',
                fontsize=14)
    ax.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/payoff_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Sensitivity Analysis
    print("  Generating sensitivity analysis figure...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    parameters = sensitivity_df['parameter'].unique()
    param_labels = {
        'gold_spot': 'Gold Spot ($/oz)',
        'eurusd_spot': 'EUR/USD Spot',
        'sigma_gold': 'Gold Volatility',
        'sigma_eurusd': 'EUR/USD Volatility',
        'correlation': 'Correlation (rho)'
    }

    for i, param in enumerate(parameters):
        if i >= len(axes) - 1:
            break
        ax = axes[i]
        data = sensitivity_df[sensitivity_df['parameter'] == param]

        ax.plot(data['value'], data['price_zgroup'] / 1e6, 'b-o', linewidth=2, markersize=6)
        ax.set_ylabel('Z Group PV (EUR Millions)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        ax2 = ax.twinx()
        ax2.plot(data['value'], data['knockout_rate'] * 100, 'r--s', linewidth=2, markersize=6)
        ax2.set_ylabel('Knockout Rate (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_xlabel(param_labels.get(param, param))
        ax.set_title(f'Sensitivity to {param_labels.get(param, param)}')
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Greeks Summary
    print("  Generating Greeks summary figure...")
    fig, ax = plt.subplots(figsize=(12, 6))

    greek_names = ['Delta\n(Gold)', 'Gamma\n(Gold)', 'Delta\n(EUR/USD)',
                  'Vega\n(Gold)', 'Rho\n(EUR)', 'Corr\nSensitivity']
    greek_values = [
        greeks['delta_gold'] / 1e6,
        greeks['gamma_gold'] / 1e6,
        greeks['delta_eurusd'] / 1e6,
        greeks['vega_gold'] / 1e6,
        greeks['rho_eur'] / 1e6,
        greeks['correlation_sensitivity'] / 1e6
    ]

    colors = ['blue' if v >= 0 else 'red' for v in greek_values]
    bars = ax.bar(greek_names, greek_values, color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Sensitivity (EUR Millions per unit)', fontsize=12)
    ax.set_title('Risk Sensitivities (Greeks) Summary', fontsize=14)

    for bar, val in zip(bars, greek_values):
        height = bar.get_height()
        ax.annotate(f'EUR {val:.2f}M',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height >= 0 else -15),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/greeks_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  All visualizations generated successfully!")


# =============================================================================
# EXCEL OUTPUT GENERATION
# =============================================================================

def generate_excel_output(pricing_result: Dict, sensitivity_df: pd.DataFrame,
                          greeks: Dict, market: MarketData, contract: ContractTerms,
                          output_path: str):
    """
    Generate comprehensive Excel file with all analysis data.

    Sheets:
    1. Market Data - Input parameters
    2. Contract Terms - Product specifications
    3. Pricing Results - Monte Carlo results
    4. Greeks - Risk sensitivities
    5. Sensitivity Analysis - Parameter sensitivity
    6. Path Statistics - Simulation statistics
    7. Sample Paths - Sample of individual path data

    Args:
        pricing_result: Monte Carlo results
        sensitivity_df: Sensitivity analysis data
        greeks: Greek values
        market: Market parameters
        contract: Contract terms
        output_path: Excel file path
    """
    print("\nGenerating Excel output...")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        # Sheet 1: Market Data
        market_params = pd.DataFrame({
            'Parameter': [
                'Gold Spot Price (USD/oz)',
                'EUR/USD Spot Rate',
                'EUR Risk-Free Rate (%)',
                'USD Risk-Free Rate (%)',
                'Gold Volatility (%)',
                'EUR/USD Volatility (%)',
                'Correlation',
                'Gold Convenience Yield (%)'
            ],
            'Value': [
                market.gold_spot,
                market.eurusd_spot,
                market.r_eur * 100,
                market.r_usd * 100,
                market.sigma_gold * 100,
                market.sigma_eurusd * 100,
                market.rho,
                market.gold_yield * 100
            ]
        })
        market_params.to_excel(writer, sheet_name='Market Data', index=False)

        # Sheet 2: Contract Terms
        contract_terms = pd.DataFrame({
            'Parameter': [
                'Notional Principal (EUR)',
                'Strike Price (USD/oz)',
                'Tenor (Years)',
                'Lower Barrier (EUR/USD)',
                'Upper Barrier (EUR/USD)',
                'Start Date',
                'End Date'
            ],
            'Value': [
                contract.notional,
                contract.strike,
                contract.tenor,
                contract.barrier_lower,
                contract.barrier_upper,
                'March 1, 2026',
                'February 28, 2028'
            ]
        })
        contract_terms.to_excel(writer, sheet_name='Contract Terms', index=False)

        # Sheet 3: Pricing Results
        pricing_summary = pd.DataFrame({
            'Metric': [
                'Z Group Present Value (EUR)',
                'A Bank Present Value (EUR)',
                'Standard Error (EUR)',
                '95% CI Lower (EUR)',
                '95% CI Upper (EUR)',
                'Knockout Rate (%)',
                'Average Knockout Time (Years)',
                'Lower Barrier Breach Rate (%)',
                'Upper Barrier Breach Rate (%)',
                'Number of Simulation Paths',
                'Number of Time Steps'
            ],
            'Value': [
                pricing_result['price_zgroup'],
                pricing_result['price_abank'],
                pricing_result['std_error'],
                pricing_result['ci_95_lower'],
                pricing_result['ci_95_upper'],
                pricing_result['knockout_rate'] * 100,
                pricing_result['avg_knockout_time'] if not np.isnan(pricing_result['avg_knockout_time']) else 'N/A',
                pricing_result['lower_breach_rate'] * 100,
                pricing_result['upper_breach_rate'] * 100,
                pricing_result['n_paths'],
                pricing_result['n_steps']
            ]
        })
        pricing_summary.to_excel(writer, sheet_name='Pricing Results', index=False)

        # Sheet 4: Greeks
        greeks_df = pd.DataFrame({
            'Greek': [
                'Delta (Gold)',
                'Gamma (Gold)',
                'Delta (EUR/USD)',
                'Vega (Gold)',
                'Rho (EUR Rate)',
                'Correlation Sensitivity'
            ],
            'Value (EUR)': [
                greeks['delta_gold'],
                greeks['gamma_gold'],
                greeks['delta_eurusd'],
                greeks['vega_gold'],
                greeks['rho_eur'],
                greeks['correlation_sensitivity']
            ],
            'Description': [
                'Change in value per $1 change in gold price',
                'Second derivative with respect to gold price',
                'Change in value per 0.01 change in EUR/USD',
                'Change in value per 1% change in gold volatility',
                'Change in value per 1bp change in EUR rate',
                'Change in value per 0.05 change in correlation'
            ]
        })
        greeks_df.to_excel(writer, sheet_name='Greeks', index=False)

        # Sheet 5: Sensitivity Analysis
        sensitivity_df.to_excel(writer, sheet_name='Sensitivity Analysis', index=False)

        # Sheet 6: Path Statistics
        settlement_prices = pricing_result['settlement_prices']
        settlement_times = pricing_result['settlement_times']
        knocked_out = pricing_result['knocked_out']

        path_stats = pd.DataFrame({
            'Statistic': [
                'Mean Settlement Price (USD/oz)',
                'Std Dev Settlement Price',
                'Min Settlement Price',
                'Max Settlement Price',
                'Mean Settlement Time (Years)',
                'Total Paths',
                'Knocked Out Paths',
                'Surviving Paths'
            ],
            'Value': [
                np.mean(settlement_prices),
                np.std(settlement_prices),
                np.min(settlement_prices),
                np.max(settlement_prices),
                np.mean(settlement_times),
                len(knocked_out),
                np.sum(knocked_out),
                np.sum(~knocked_out)
            ]
        })
        path_stats.to_excel(writer, sheet_name='Path Statistics', index=False)

        # Sheet 7: Sample Paths
        n_sample = min(1000, len(settlement_prices))
        np.random.seed(42)
        sample_idx = np.random.choice(len(settlement_prices), n_sample, replace=False)
        sample_data = pd.DataFrame({
            'Path_ID': sample_idx,
            'Settlement_Price_USD': settlement_prices[sample_idx],
            'Settlement_Time_Years': settlement_times[sample_idx],
            'Knocked_Out': knocked_out[sample_idx],
            'Payoff_ZGroup_EUR': contract.notional * (settlement_prices[sample_idx] - contract.strike) / contract.strike
        })
        sample_data.to_excel(writer, sheet_name='Sample Paths', index=False)

    print(f"Excel file saved to: {output_path}")


# =============================================================================
# VALIDATION AND CONVERGENCE ANALYSIS
# =============================================================================

def validate_inputs(market: MarketData, contract: ContractTerms) -> list:
    """
    Validate market data and contract terms for reasonableness.

    Returns list of warnings/errors. Empty list means all inputs valid.
    """
    issues = []

    # Spot prices must be positive
    if market.gold_spot <= 0:
        issues.append(f"ERROR: Gold spot must be positive, got {market.gold_spot}")
    if market.eurusd_spot <= 0:
        issues.append(f"ERROR: EUR/USD spot must be positive, got {market.eurusd_spot}")

    # Volatilities: positive and reasonable
    if not 0 < market.sigma_gold < 1.0:
        issues.append(f"WARNING: Gold volatility {market.sigma_gold*100:.1f}% outside typical range")
    if not 0 < market.sigma_eurusd < 0.5:
        issues.append(f"WARNING: EUR/USD volatility {market.sigma_eurusd*100:.1f}% outside typical range")

    # Correlation in [-1, 1]
    if not -1 <= market.rho <= 1:
        issues.append(f"ERROR: Correlation must be in [-1,1], got {market.rho}")

    # Check strike vs forward price
    forward_price = market.gold_spot * np.exp(
        (market.r_usd - market.gold_yield) * contract.tenor
    )
    strike_ratio = contract.strike / forward_price
    if strike_ratio > 1.3:
        issues.append(f"NOTE: Strike ${contract.strike:,.0f} is {(strike_ratio-1)*100:.0f}% above "
                     f"forward ${forward_price:,.0f} (deep OTM for Z Group)")

    # Check barrier proximity
    if not contract.barrier_lower < market.eurusd_spot < contract.barrier_upper:
        issues.append(f"ERROR: Spot {market.eurusd_spot} outside barrier range!")

    lower_dist = (market.eurusd_spot - contract.barrier_lower) / market.eurusd_spot
    if lower_dist < 0.05:
        issues.append(f"NOTE: Lower barrier only {lower_dist*100:.1f}% from spot (high KO probability)")

    return issues


def compute_analytical_benchmarks(market: MarketData, contract: ContractTerms) -> dict:
    """
    Compute analytical benchmarks for model validation.

    The vanilla forward (no barriers) has a known price - useful for sanity checks.
    Includes quanto adjustment calculation.
    """
    # Quanto adjustment: -ρ × σ_S × σ_X
    # With negative correlation, this is positive (increases gold drift under EUR measure)
    quanto_adjustment = market.rho * market.sigma_gold * market.sigma_eurusd
    drift_without_quanto = market.r_usd - market.gold_yield
    drift_with_quanto = drift_without_quanto - quanto_adjustment

    # Gold forward price (standard, without quanto)
    gold_forward = market.gold_spot * np.exp(
        drift_without_quanto * contract.tenor
    )

    # Gold forward under EUR measure (with quanto adjustment)
    gold_forward_quanto = market.gold_spot * np.exp(
        drift_with_quanto * contract.tenor
    )

    # EUR/USD forward (interest rate parity)
    eurusd_forward = market.eurusd_spot * np.exp(
        (market.r_eur - market.r_usd) * contract.tenor
    )

    # Vanilla forward PV (no barriers) - with quanto adjustment
    vanilla_pv = np.exp(-market.r_eur * contract.tenor) * \
                 contract.notional * (gold_forward_quanto - contract.strike) / contract.strike

    return {
        'gold_forward': gold_forward,
        'gold_forward_quanto': gold_forward_quanto,
        'eurusd_forward': eurusd_forward,
        'vanilla_forward_pv': vanilla_pv,
        'moneyness': gold_forward / contract.strike,
        'quanto_adjustment': quanto_adjustment,
        'drift_without_quanto': drift_without_quanto,
        'drift_with_quanto': drift_with_quanto
    }


def run_convergence_test(pricer, path_counts: list = None) -> pd.DataFrame:
    """
    Test Monte Carlo convergence across different path counts.

    Validates that estimates stabilize as paths increase.
    """
    if path_counts is None:
        path_counts = [5000, 10000, 25000, 50000, 100000]

    results = []
    for n in path_counts:
        res = pricer.price_monte_carlo(n_paths=n, n_steps=504, seed=42)
        results.append({
            'paths': n,
            'price': res['price_zgroup'],
            'std_error': res['std_error'],
            'ko_rate': res['knockout_rate']
        })

    return pd.DataFrame(results)


def plot_convergence_analysis(conv_df: pd.DataFrame, output_dir: str):
    """Generate convergence analysis plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Price convergence with CI
    ax1 = axes[0]
    ax1.errorbar(conv_df['paths'], conv_df['price'] / 1e6,
                yerr=1.96 * conv_df['std_error'] / 1e6,
                fmt='b-o', linewidth=2, markersize=6, capsize=4)
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Paths')
    ax1.set_ylabel('Price (EUR Millions)')
    ax1.set_title('Monte Carlo Convergence')
    ax1.grid(True, alpha=0.3)

    # Standard error decay
    ax2 = axes[1]
    ax2.plot(conv_df['paths'], conv_df['std_error'] / 1e6, 'r-o', linewidth=2, markersize=6)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Paths')
    ax2.set_ylabel('Standard Error (EUR Millions)')
    ax2.set_title('Standard Error Decay (should be ~1/√n)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for GAAIF Challenge submission.

    Executes complete analysis pipeline:
    1. Initialize market data and contract terms
    2. Run Monte Carlo pricing
    3. Compute Greeks
    4. Run sensitivity analysis
    5. Generate visualizations
    6. Export Excel data file
    """
    print("=" * 70)
    print("GAAIF CHALLENGE - STRUCTURED FORWARD PRICING MODEL")
    print("=" * 70)
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize market data
    print("\n" + "-" * 70)
    print("INITIALIZING MARKET DATA")
    print("-" * 70)

    market = MarketData(
        gold_spot=2750.0,        # Current gold price
        eurusd_spot=1.08,        # Current EUR/USD
        r_eur=0.025,             # ECB rate
        r_usd=0.045,             # Fed rate
        sigma_gold=0.18,         # Gold volatility
        sigma_eurusd=0.08,       # EUR/USD volatility
        rho=-0.25,               # Negative correlation
        gold_yield=0.005         # Convenience yield
    )

    contract = ContractTerms(
        notional=500_000_000,
        strike=4600.0,
        tenor=2.0,
        barrier_lower=1.05,
        barrier_upper=1.25
    )

    print(f"\nMarket Parameters:")
    print(f"  Gold Spot:        ${market.gold_spot:,.2f}/oz")
    print(f"  EUR/USD Spot:     {market.eurusd_spot:.4f}")
    print(f"  EUR Rate:         {market.r_eur*100:.2f}%")
    print(f"  USD Rate:         {market.r_usd*100:.2f}%")
    print(f"  Gold Vol:         {market.sigma_gold*100:.1f}%")
    print(f"  EUR/USD Vol:      {market.sigma_eurusd*100:.1f}%")
    print(f"  Correlation:      {market.rho:.2f}")

    print(f"\nContract Terms:")
    print(f"  Notional:         EUR {contract.notional/1e6:,.0f}M")
    print(f"  Strike:           ${contract.strike:,.2f}/oz")
    print(f"  Tenor:            {contract.tenor} years")
    print(f"  Barriers:         [{contract.barrier_lower}, {contract.barrier_upper}]")

    # Input Validation
    print("\n" + "-" * 70)
    print("INPUT VALIDATION")
    print("-" * 70)

    issues = validate_inputs(market, contract)
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  All inputs validated successfully.")

    # Analytical Benchmarks
    print("\n" + "-" * 70)
    print("ANALYTICAL BENCHMARKS")
    print("-" * 70)

    benchmarks = compute_analytical_benchmarks(market, contract)
    print(f"  Gold 2Y Forward:       ${benchmarks['gold_forward']:,.0f}/oz")
    print(f"  Gold Forward (Quanto): ${benchmarks['gold_forward_quanto']:,.0f}/oz")
    print(f"  EUR/USD 2Y Forward:    {benchmarks['eurusd_forward']:.4f}")
    print(f"  Vanilla Forward PV:    EUR {benchmarks['vanilla_forward_pv']:,.0f}")
    print(f"  Moneyness (F/K):       {benchmarks['moneyness']:.1%}")
    print(f"\nQuanto Adjustment Details:")
    print(f"  Adjustment term:       {benchmarks['quanto_adjustment']*100:.3f}% (ρ × σ_S × σ_X)")
    print(f"  Drift w/o quanto:      {benchmarks['drift_without_quanto']*100:.2f}%")
    print(f"  Drift w/ quanto:       {benchmarks['drift_with_quanto']*100:.2f}%")
    print(f"  Impact on forward:     ${benchmarks['gold_forward_quanto'] - benchmarks['gold_forward']:,.0f}/oz")

    # Monte Carlo Pricing
    print("\n" + "-" * 70)
    print("MONTE CARLO PRICING")
    print("-" * 70)

    pricer = StructuredForwardPricer(market, contract)

    print("\nRunning Monte Carlo simulation (100,000 paths, 504 steps)...")
    pricing_result = pricer.price_monte_carlo(
        n_paths=100000,
        n_steps=504,
        seed=42,
        antithetic=True,
        control_variate=True
    )

    print(f"\nPricing Results:")
    print(f"  Z Group PV:       EUR {pricing_result['price_zgroup']:,.2f}")
    print(f"  A Bank PV:        EUR {pricing_result['price_abank']:,.2f}")
    print(f"  Std Error:        EUR {pricing_result['std_error']:,.2f}")
    print(f"  95% CI:           [{pricing_result['ci_95_lower']:,.2f}, {pricing_result['ci_95_upper']:,.2f}]")
    print(f"\nBarrier Analysis:")
    print(f"  Knockout Rate:    {pricing_result['knockout_rate']*100:.2f}%")
    if not np.isnan(pricing_result['avg_knockout_time']):
        print(f"  Avg KO Time:      {pricing_result['avg_knockout_time']:.2f} years")
    print(f"  Lower Breach:     {pricing_result['lower_breach_rate']*100:.2f}%")
    print(f"  Upper Breach:     {pricing_result['upper_breach_rate']*100:.2f}%")

    # Convergence Analysis
    print("\n" + "-" * 70)
    print("CONVERGENCE ANALYSIS")
    print("-" * 70)

    print("\nTesting Monte Carlo convergence...")
    conv_df = run_convergence_test(pricer, path_counts=[5000, 10000, 25000, 50000, 100000])
    print(f"\n  {'Paths':>10} {'Price (EUR)':>18} {'Std Error':>14} {'KO Rate':>10}")
    print("  " + "-" * 56)
    for _, row in conv_df.iterrows():
        print(f"  {row['paths']:>10,} {row['price']:>18,.0f} {row['std_error']:>14,.0f} {row['ko_rate']*100:>9.1f}%")

    print("\n  Generating convergence plot...")
    plot_convergence_analysis(conv_df, output_dir)

    # Greeks
    print("\n" + "-" * 70)
    print("COMPUTING GREEKS")
    print("-" * 70)

    greeks = pricer.compute_greeks(pricing_result, n_paths=50000, seed=42)

    print(f"\nRisk Sensitivities:")
    print(f"  Delta (Gold):     EUR {greeks['delta_gold']:,.2f} per $1")
    print(f"  Gamma (Gold):     EUR {greeks['gamma_gold']:,.2f}")
    print(f"  Delta (EUR/USD):  EUR {greeks['delta_eurusd']:,.2f} per 0.01")
    print(f"  Vega (Gold):      EUR {greeks['vega_gold']:,.2f} per 1% vol")
    print(f"  Rho (EUR):        EUR {greeks['rho_eur']:,.2f} per 1bp")
    print(f"  Corr Sens:        EUR {greeks['correlation_sensitivity']:,.2f} per 0.05")

    # Sensitivity Analysis
    print("\n" + "-" * 70)
    print("SENSITIVITY ANALYSIS")
    print("-" * 70)

    sensitivity_df = run_sensitivity_analysis(market, contract, n_paths=50000)

    # Visualizations
    print("\n" + "-" * 70)
    print("GENERATING VISUALIZATIONS")
    print("-" * 70)

    create_visualizations(pricing_result, sensitivity_df, greeks, contract, output_dir)

    # Excel Output
    print("\n" + "-" * 70)
    print("GENERATING EXCEL OUTPUT")
    print("-" * 70)

    generate_excel_output(
        pricing_result, sensitivity_df, greeks,
        market, contract,
        f'{output_dir}/GAAIF_Analysis_Data.xlsx'
    )

    # Quanto Impact Analysis
    print("\n" + "-" * 70)
    print("QUANTO ADJUSTMENT IMPACT")
    print("-" * 70)

    print("\nComparing pricing with vs without quanto adjustment...")
    result_no_quanto = pricer.price_monte_carlo(
        n_paths=100000, n_steps=504, seed=42,
        use_quanto_adjustment=False
    )
    quanto_impact = pricing_result['price_zgroup'] - result_no_quanto['price_zgroup']
    print(f"  Price WITH quanto:     EUR {pricing_result['price_zgroup']:,.0f}")
    print(f"  Price WITHOUT quanto:  EUR {result_no_quanto['price_zgroup']:,.0f}")
    print(f"  Quanto Impact:         EUR {quanto_impact:,.0f}")
    print(f"  Impact as % of PV:     {quanto_impact/abs(pricing_result['price_zgroup'])*100:.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nOutput files generated:")
    print(f"  - {output_dir}/GAAIF_Analysis_Data.xlsx")
    print(f"  - {output_dir}/convergence_analysis.png")
    print(f"  - {output_dir}/monte_carlo_paths.png")
    print(f"  - {output_dir}/payoff_distribution.png")
    print(f"  - {output_dir}/payoff_diagram.png")
    print(f"  - {output_dir}/sensitivity_analysis.png")
    print(f"  - {output_dir}/greeks_summary.png")

    return {
        'pricing_result': pricing_result,
        'greeks': greeks,
        'sensitivity_df': sensitivity_df,
        'benchmarks': benchmarks,
        'quanto_impact': quanto_impact
    }


if __name__ == "__main__":
    results = main()
