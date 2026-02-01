#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GAAIF Challenge 2026 - Structured Gold Forward Pricing Model
================================================================================

Product: Gold Forward Contract with EUR/USD Double Knock-Out Barriers

Contract Specifications:
    - Notional Principal: EUR 500 Million
    - Strike Price (K): USD 4,600/oz
    - Tenor: 2 years (March 2026 - February 2028)
    - Knock-Out Barriers: EUR/USD < 1.05 or EUR/USD > 1.25

Settlement Formulas:
    - Z Group Payoff: N × (P - K) / K
    - A Bank Payoff:  N × (K - P) / K
    where P = LBMA Gold Spot at settlement, K = Strike

Mathematical Framework:
    Two-factor correlated Geometric Brownian Motion under risk-neutral measure:

    Gold:     dS/S = (r_USD - q) dt + σ_S dW^S
    EUR/USD:  dX/X = (r_EUR - r_USD) dt + σ_X dW^X

    Correlation: dW^S · dW^X = ρ dt

Author: GAAIF Challenge Submission
Date: February 2026
================================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA CLASSES FOR MODEL INPUTS
# =============================================================================

@dataclass
class MarketData:
    """
    Container for market data parameters.

    All rates are expressed as continuous compounding (annualized).
    Volatilities are annualized standard deviations.

    Attributes:
        gold_spot: Current LBMA gold spot price (USD/oz)
        eurusd_spot: Current EUR/USD exchange rate
        r_eur: EUR risk-free rate (ECB deposit rate)
        r_usd: USD risk-free rate (Fed funds rate)
        sigma_gold: Gold price volatility (annualized)
        sigma_eurusd: EUR/USD volatility (annualized)
        rho: Correlation between gold and EUR/USD returns
        gold_yield: Gold convenience yield / lease rate
    """
    # Spot prices (as of February 1, 2026)
    gold_spot: float = 4900.0       # USD/oz - elevated after rally to $5,608
    eurusd_spot: float = 1.19       # Near 4-year high

    # Interest rates (continuous compounding)
    r_eur: float = 0.025            # 2.5% ECB deposit rate
    r_usd: float = 0.0425           # 4.25% Fed funds rate

    # Volatilities (annualized)
    sigma_gold: float = 0.28        # 28% - elevated due to recent volatility
    sigma_eurusd: float = 0.10      # 10% - also elevated

    # Correlation and yield
    rho: float = -0.30              # Negative correlation (typical)
    gold_yield: float = 0.003       # 0.3% convenience yield


@dataclass
class ContractTerms:
    """
    Container for contract specifications.

    Attributes:
        notional: Notional principal in EUR
        strike: Strike price in USD/oz
        tenor: Contract tenor in years
        barrier_lower: Lower EUR/USD knock-out barrier
        barrier_upper: Upper EUR/USD knock-out barrier
    """
    notional: float = 500_000_000   # EUR 500 Million
    strike: float = 4600.0          # USD 4,600/oz
    tenor: float = 2.0              # 2 years
    barrier_lower: float = 1.05     # Lower knock-out barrier
    barrier_upper: float = 1.25     # Upper knock-out barrier


# =============================================================================
# MONTE CARLO SIMULATION ENGINE
# =============================================================================

class CorrelatedGBMSimulator:
    """
    Simulates correlated Geometric Brownian Motion paths for Gold and EUR/USD.

    Uses Cholesky decomposition to generate correlated random variables:
        W^X = ρ·W^S + √(1-ρ²)·Z
    where Z is independent standard normal.

    Implements exact log-normal solution for discretization:
        S(t+dt) = S(t) × exp[(μ - σ²/2)dt + σ√dt·W]
    """

    def __init__(self, market: MarketData, contract: ContractTerms,
                 seed: Optional[int] = None):
        """
        Initialize simulator with market data and contract terms.

        Args:
            market: MarketData instance with current market parameters
            contract: ContractTerms instance with product specifications
            seed: Random seed for reproducibility
        """
        self.market = market
        self.contract = contract
        self.rng = np.random.default_rng(seed)

    def _generate_correlated_normals(self, n_paths: int,
                                      n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate correlated standard normal random variables.

        Uses Cholesky decomposition of correlation matrix:
            [1   ρ  ]   [1      0        ] [1  ρ]
            [ρ   1  ] = [ρ  √(1-ρ²)] [0  1]

        Args:
            n_paths: Number of simulation paths
            n_steps: Number of time steps per path

        Returns:
            Tuple of (W_gold, W_eurusd) correlated normal arrays
        """
        # Generate independent standard normals
        Z1 = self.rng.standard_normal((n_paths, n_steps))
        Z2 = self.rng.standard_normal((n_paths, n_steps))

        # Apply Cholesky decomposition for correlation
        rho = self.market.rho
        W_gold = Z1
        W_eurusd = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        return W_gold, W_eurusd

    def simulate_paths(self, n_paths: int, n_steps: int,
                       antithetic: bool = True) -> Dict:
        """
        Simulate correlated Gold and EUR/USD price paths.

        Args:
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps (e.g., 504 for daily over 2 years)
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

        # Generate half paths if using antithetic variates
        actual_paths = n_paths // 2 if antithetic else n_paths
        W_gold, W_eurusd = self._generate_correlated_normals(actual_paths, n_steps)

        # Mirror paths for antithetic variates
        if antithetic:
            W_gold = np.vstack([W_gold, -W_gold])
            W_eurusd = np.vstack([W_eurusd, -W_eurusd])
            actual_paths = n_paths

        # Initialize price arrays
        gold_paths = np.zeros((actual_paths, n_steps + 1))
        eurusd_paths = np.zeros((actual_paths, n_steps + 1))

        gold_paths[:, 0] = self.market.gold_spot
        eurusd_paths[:, 0] = self.market.eurusd_spot

        # Risk-neutral drift terms
        gold_drift = self.market.r_usd - self.market.gold_yield
        eurusd_drift = self.market.r_eur - self.market.r_usd

        # Simulate using exact log-normal solution
        for t in range(n_steps):
            # Gold: dS/S = (r_USD - q)dt + σ_S dW^S
            gold_paths[:, t+1] = gold_paths[:, t] * np.exp(
                (gold_drift - 0.5 * self.market.sigma_gold**2) * dt +
                self.market.sigma_gold * sqrt_dt * W_gold[:, t]
            )

            # EUR/USD: dX/X = (r_EUR - r_USD)dt + σ_X dW^X
            eurusd_paths[:, t+1] = eurusd_paths[:, t] * np.exp(
                (eurusd_drift - 0.5 * self.market.sigma_eurusd**2) * dt +
                self.market.sigma_eurusd * sqrt_dt * W_eurusd[:, t]
            )

        return {
            'gold_paths': gold_paths,
            'eurusd_paths': eurusd_paths,
            'times': np.linspace(0, self.contract.tenor, n_steps + 1),
            'dt': dt
        }

    def check_barrier_breach(self, eurusd_paths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check for knock-out barrier breaches in EUR/USD paths.

        Args:
            eurusd_paths: Array of EUR/USD price paths

        Returns:
            Tuple of:
                - knocked_out: Boolean array indicating knocked-out paths
                - knockout_idx: Index of first breach for each path (-1 if none)
        """
        n_paths = eurusd_paths.shape[0]

        # Check barrier conditions
        below_lower = eurusd_paths < self.contract.barrier_lower
        above_upper = eurusd_paths > self.contract.barrier_upper
        breached = below_lower | above_upper

        # Find first breach for each path
        knocked_out = np.any(breached, axis=1)
        knockout_idx = np.full(n_paths, -1, dtype=int)

        for i in range(n_paths):
            if knocked_out[i]:
                knockout_idx[i] = np.argmax(breached[i])

        return knocked_out, knockout_idx


# =============================================================================
# PRICING ENGINE
# =============================================================================

class StructuredForwardPricer:
    """
    Prices the structured gold forward with double knock-out barriers.

    The product pays at maturity (or early termination):
        Z Group: N × (P_τ - K) / K
        A Bank:  N × (K - P_τ) / K

    where τ = min(T, first barrier breach time).

    Implements variance reduction techniques:
        1. Antithetic variates
        2. Control variate (vanilla forward as control)
    """

    def __init__(self, market: MarketData, contract: ContractTerms):
        """
        Initialize pricer with market data and contract terms.

        Args:
            market: MarketData instance
            contract: ContractTerms instance
        """
        self.market = market
        self.contract = contract

    def price_monte_carlo(self, n_paths: int = 100000, n_steps: int = 504,
                          seed: Optional[int] = 42, antithetic: bool = True,
                          control_variate: bool = True) -> Dict:
        """
        Price the structured product using Monte Carlo simulation.

        Args:
            n_paths: Number of simulation paths (default: 100,000)
            n_steps: Number of time steps (default: 504 ≈ daily for 2 years)
            seed: Random seed for reproducibility
            antithetic: Use antithetic variates
            control_variate: Use control variate adjustment

        Returns:
            Dictionary with comprehensive pricing results
        """
        # Initialize simulator and generate paths
        simulator = CorrelatedGBMSimulator(self.market, self.contract, seed)
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
                # Early termination at knock-out
                idx = knockout_idx[i]
                settlement_prices[i] = gold_paths[i, idx]
                settlement_times[i] = times[idx]
            else:
                # Settlement at maturity
                settlement_prices[i] = gold_paths[i, -1]
                settlement_times[i] = self.contract.tenor

        # Calculate payoffs (from Z Group perspective)
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
            # Vanilla forward as control variate
            forward_price = self.market.gold_spot * np.exp(
                (self.market.r_usd - self.market.gold_yield) * self.contract.tenor
            )
            vanilla_payoffs = N * (gold_paths[:, -1] - K) / K
            vanilla_pv = vanilla_payoffs * np.exp(-self.market.r_eur * self.contract.tenor)

            # Analytical vanilla forward price
            analytical_vanilla = N * (forward_price - K) / K * \
                                 np.exp(-self.market.r_eur * self.contract.tenor)

            # Optimal control variate coefficient
            cov_matrix = np.cov(pv_zgroup, vanilla_pv)
            if cov_matrix[1, 1] > 0:
                beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                cv_adjustment = beta * (analytical_vanilla - np.mean(vanilla_pv))

        # Calculate statistics
        mean_pv_zgroup = np.mean(pv_zgroup) + cv_adjustment
        mean_pv_abank = np.mean(pv_abank) - cv_adjustment
        std_pv = np.std(pv_zgroup)
        se = std_pv / np.sqrt(n_actual_paths)

        # Knockout statistics
        knockout_rate = np.mean(knocked_out)
        avg_ko_time = np.mean(settlement_times[knocked_out]) if np.any(knocked_out) else np.nan

        # Barrier breach breakdown
        if np.any(knocked_out):
            ko_eurusd = eurusd_paths[knocked_out, :]
            ko_idx = knockout_idx[knocked_out]
            ko_prices = np.array([ko_eurusd[i, ko_idx[i]] for i in range(len(ko_idx))])
            lower_breach_rate = np.mean(ko_prices < self.contract.barrier_lower) * knockout_rate
            upper_breach_rate = np.mean(ko_prices > self.contract.barrier_upper) * knockout_rate
        else:
            lower_breach_rate = 0.0
            upper_breach_rate = 0.0

        return {
            'price_zgroup': mean_pv_zgroup,
            'price_abank': mean_pv_abank,
            'std_error': se,
            'ci_95_lower': mean_pv_zgroup - 1.96 * se,
            'ci_95_upper': mean_pv_zgroup + 1.96 * se,
            'knockout_rate': knockout_rate,
            'avg_knockout_time': avg_ko_time,
            'lower_breach_rate': lower_breach_rate,
            'upper_breach_rate': upper_breach_rate,
            'mean_settlement_price': np.mean(settlement_prices),
            'n_paths': n_actual_paths,
            'n_steps': n_steps
        }

    def compute_greeks(self, bump_size: float = 0.01,
                       n_paths: int = 50000, seed: int = 42) -> Dict:
        """
        Compute option Greeks using finite difference method.

        Args:
            bump_size: Relative bump size for finite differences
            n_paths: Number of paths for Greek computation
            seed: Random seed

        Returns:
            Dictionary with Delta, Gamma, Vega, Rho, and correlation sensitivity
        """
        base_result = self.price_monte_carlo(n_paths, 252, seed)
        base_price = base_result['price_zgroup']

        greeks = {}

        # Delta (Gold) - central difference
        for direction, mult in [('up', 1), ('down', -1)]:
            m = MarketData(
                gold_spot=self.market.gold_spot * (1 + mult * bump_size),
                eurusd_spot=self.market.eurusd_spot,
                r_eur=self.market.r_eur, r_usd=self.market.r_usd,
                sigma_gold=self.market.sigma_gold,
                sigma_eurusd=self.market.sigma_eurusd,
                rho=self.market.rho, gold_yield=self.market.gold_yield
            )
            p = StructuredForwardPricer(m, self.contract)
            greeks[f'gold_{direction}'] = p.price_monte_carlo(n_paths, 252, seed)['price_zgroup']

        greeks['delta_gold'] = (greeks['gold_up'] - greeks['gold_down']) / \
                               (2 * bump_size * self.market.gold_spot)
        greeks['gamma_gold'] = (greeks['gold_up'] - 2*base_price + greeks['gold_down']) / \
                               (bump_size * self.market.gold_spot)**2

        # Delta (EUR/USD)
        for direction, mult in [('up', 1), ('down', -1)]:
            m = MarketData(
                gold_spot=self.market.gold_spot,
                eurusd_spot=self.market.eurusd_spot * (1 + mult * bump_size),
                r_eur=self.market.r_eur, r_usd=self.market.r_usd,
                sigma_gold=self.market.sigma_gold,
                sigma_eurusd=self.market.sigma_eurusd,
                rho=self.market.rho, gold_yield=self.market.gold_yield
            )
            p = StructuredForwardPricer(m, self.contract)
            greeks[f'fx_{direction}'] = p.price_monte_carlo(n_paths, 252, seed)['price_zgroup']

        greeks['delta_eurusd'] = (greeks['fx_up'] - greeks['fx_down']) / \
                                 (2 * bump_size * self.market.eurusd_spot)

        # Vega (Gold volatility)
        vega_bump = 0.01
        m_vega = MarketData(
            gold_spot=self.market.gold_spot, eurusd_spot=self.market.eurusd_spot,
            r_eur=self.market.r_eur, r_usd=self.market.r_usd,
            sigma_gold=self.market.sigma_gold + vega_bump,
            sigma_eurusd=self.market.sigma_eurusd,
            rho=self.market.rho, gold_yield=self.market.gold_yield
        )
        p_vega = StructuredForwardPricer(m_vega, self.contract)
        greeks['vega_gold'] = (p_vega.price_monte_carlo(n_paths, 252, seed)['price_zgroup'] -
                              base_price) / vega_bump

        # Rho (EUR rate)
        rho_bump = 0.001
        m_rho = MarketData(
            gold_spot=self.market.gold_spot, eurusd_spot=self.market.eurusd_spot,
            r_eur=self.market.r_eur + rho_bump, r_usd=self.market.r_usd,
            sigma_gold=self.market.sigma_gold, sigma_eurusd=self.market.sigma_eurusd,
            rho=self.market.rho, gold_yield=self.market.gold_yield
        )
        p_rho = StructuredForwardPricer(m_rho, self.contract)
        greeks['rho_eur'] = (p_rho.price_monte_carlo(n_paths, 252, seed)['price_zgroup'] -
                            base_price) / rho_bump

        return {
            'delta_gold': greeks['delta_gold'],
            'gamma_gold': greeks['gamma_gold'],
            'delta_eurusd': greeks['delta_eurusd'],
            'vega_gold': greeks['vega_gold'],
            'rho_eur': greeks['rho_eur']
        }


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity_analysis(market: MarketData, contract: ContractTerms,
                             n_paths: int = 50000) -> pd.DataFrame:
    """
    Run comprehensive sensitivity analysis on key parameters.

    Analyzes sensitivity to:
        - Gold spot price
        - EUR/USD spot rate
        - Gold volatility
        - EUR/USD volatility
        - Correlation

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
    for gs in np.linspace(4000, 5600, 9):
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
    for fx in np.linspace(1.06, 1.24, 10):
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
    for vol in np.linspace(0.15, 0.40, 6):
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

    # Correlation sensitivity
    print("  Analyzing correlation sensitivity...")
    for rho in np.linspace(-0.6, 0.4, 6):
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
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function - runs complete pricing analysis.
    """
    print("=" * 70)
    print("GAAIF CHALLENGE - STRUCTURED GOLD FORWARD PRICING MODEL")
    print("=" * 70)

    # Initialize market data and contract
    market = MarketData()
    contract = ContractTerms()

    # Display parameters
    print("\n" + "-" * 70)
    print("MARKET DATA (February 1, 2026)")
    print("-" * 70)
    print(f"  Gold Spot Price:     ${market.gold_spot:,.2f}/oz")
    print(f"  EUR/USD Spot:        {market.eurusd_spot:.4f}")
    print(f"  EUR Risk-Free Rate:  {market.r_eur*100:.2f}%")
    print(f"  USD Risk-Free Rate:  {market.r_usd*100:.2f}%")
    print(f"  Gold Volatility:     {market.sigma_gold*100:.1f}%")
    print(f"  EUR/USD Volatility:  {market.sigma_eurusd*100:.1f}%")
    print(f"  Correlation:         {market.rho:.2f}")

    print("\n" + "-" * 70)
    print("CONTRACT TERMS")
    print("-" * 70)
    print(f"  Notional:            EUR {contract.notional/1e6:,.0f} Million")
    print(f"  Strike Price (K):    ${contract.strike:,.2f}/oz")
    print(f"  Tenor:               {contract.tenor} years")
    print(f"  Lower Barrier:       EUR/USD {contract.barrier_lower}")
    print(f"  Upper Barrier:       EUR/USD {contract.barrier_upper}")

    # Current intrinsic value
    intrinsic = contract.notional * (market.gold_spot - contract.strike) / contract.strike
    print(f"\n  Current Intrinsic:   EUR {intrinsic/1e6:,.2f} Million")

    # Monte Carlo Pricing
    print("\n" + "-" * 70)
    print("MONTE CARLO PRICING")
    print("-" * 70)
    print("Running simulation (100,000 paths, 504 time steps)...")

    pricer = StructuredForwardPricer(market, contract)
    result = pricer.price_monte_carlo(n_paths=100000, n_steps=504, seed=42)

    print(f"\n  Z Group Present Value:  EUR {result['price_zgroup']/1e6:,.2f} Million")
    print(f"  A Bank Present Value:   EUR {result['price_abank']/1e6:,.2f} Million")
    print(f"  Standard Error:         EUR {result['std_error']/1e6:,.4f} Million")
    print(f"  95% Confidence Interval: [{result['ci_95_lower']/1e6:,.2f}M, "
          f"{result['ci_95_upper']/1e6:,.2f}M]")

    print(f"\n  Knockout Rate:          {result['knockout_rate']*100:.2f}%")
    print(f"  Avg Knockout Time:      {result['avg_knockout_time']:.2f} years")
    print(f"  Lower Barrier Breaches: {result['lower_breach_rate']*100:.2f}%")
    print(f"  Upper Barrier Breaches: {result['upper_breach_rate']*100:.2f}%")

    # Greeks
    print("\n" + "-" * 70)
    print("RISK SENSITIVITIES (GREEKS)")
    print("-" * 70)
    print("Computing Greeks (50,000 paths)...")

    greeks = pricer.compute_greeks(n_paths=50000, seed=42)

    print(f"\n  Delta (Gold):        EUR {greeks['delta_gold']:,.0f} per $1 gold")
    print(f"  Gamma (Gold):        EUR {greeks['gamma_gold']:,.2f}")
    print(f"  Delta (EUR/USD):     EUR {greeks['delta_eurusd']/1e6:,.2f}M per 0.01 FX")
    print(f"  Vega (Gold Vol):     EUR {greeks['vega_gold']/1e6:,.2f}M per 1% vol")
    print(f"  Rho (EUR Rate):      EUR {greeks['rho_eur']/1e6:,.2f}M per 1bp")

    # Sensitivity Analysis
    print("\n" + "-" * 70)
    print("SENSITIVITY ANALYSIS")
    print("-" * 70)

    sensitivity_df = run_sensitivity_analysis(market, contract, n_paths=30000)

    print("\n  Gold Spot Sensitivity:")
    gold_sens = sensitivity_df[sensitivity_df['parameter'] == 'gold_spot']
    for _, row in gold_sens.iterrows():
        print(f"    ${row['value']:,.0f}: EUR {row['price_zgroup']/1e6:+,.1f}M")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        'market': market,
        'contract': contract,
        'pricing_result': result,
        'greeks': greeks,
        'sensitivity': sensitivity_df
    }


if __name__ == "__main__":
    results = main()
