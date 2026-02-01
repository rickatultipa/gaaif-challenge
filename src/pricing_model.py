"""
GAAIF Challenge - Structured Forward with Double Knock-Out Barrier Pricing Model
================================================================================

Product: Gold Forward Contract with EUR/USD Double Knock-Out Barriers
- Notional: EUR 500 Million
- Strike (K): $4,600/oz
- Tenor: 2 years
- Barriers: EUR/USD lower = 1.05, upper = 1.25
- Settlement: N × (P - K) / K for Z Group; N × (K - P) / K for A Bank

Mathematical Framework:
- Two-factor correlated GBM under risk-neutral measure
- Gold: dS/S = (r_USD - q) dt + σ_S dW^S
- EUR/USD: dX/X = (r_EUR - r_USD) dt + σ_X dW^X
- Correlation: dW^S × dW^X = ρ dt

Author: GAAIF Challenge Submission
Date: January 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MarketData:
    """Container for market data parameters."""
    # Spot prices - UPDATED Feb 1, 2026
    # Gold crashed from $5,608 high on Jan 28 to ~$4,900, still well above strike
    gold_spot: float = 4900.0  # Current gold price (USD/oz) - Feb 2026
    eurusd_spot: float = 1.19  # Current EUR/USD rate - near 4-year high

    # Interest rates (annualized, continuous compounding)
    r_eur: float = 0.025  # EUR risk-free rate (ECB)
    r_usd: float = 0.0425  # USD risk-free rate (Fed has been cutting)

    # Volatilities (annualized) - INCREASED due to recent extreme moves
    # Gold dropped 7%+ in one week - realized vol is spiking
    sigma_gold: float = 0.28  # Gold volatility - elevated (~28% given recent crash)
    sigma_eurusd: float = 0.10  # EUR/USD volatility (~10% - also elevated)

    # Correlation - may have shifted during crisis
    rho: float = -0.30  # Gold-EURUSD correlation (stronger negative in crisis)

    # Gold convenience yield
    gold_yield: float = 0.003  # Reduced lease rate in high-price environment


@dataclass
class ContractTerms:
    """Container for contract specifications."""
    notional: float = 500_000_000  # EUR 500 Million
    strike: float = 4600.0  # Gold strike price (USD/oz)
    tenor: float = 2.0  # 2 years
    barrier_lower: float = 1.05  # Lower EUR/USD barrier
    barrier_upper: float = 1.25  # Upper EUR/USD barrier


class CorrelatedGBMSimulator:
    """
    Simulates correlated Geometric Brownian Motion paths for Gold and EUR/USD.
    Uses Cholesky decomposition for correlation structure.
    """

    def __init__(self, market: MarketData, contract: ContractTerms, seed: Optional[int] = None):
        self.market = market
        self.contract = contract
        self.rng = np.random.default_rng(seed)

    def _generate_correlated_normals(self, n_paths: int, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate correlated standard normal random variables using Cholesky decomposition."""
        # Independent standard normals
        Z1 = self.rng.standard_normal((n_paths, n_steps))
        Z2 = self.rng.standard_normal((n_paths, n_steps))

        # Apply Cholesky decomposition for correlation
        # [W1]   [1    0         ] [Z1]
        # [W2] = [rho  sqrt(1-rho^2)] [Z2]
        rho = self.market.rho
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        return W1, W2

    def simulate_paths(self, n_paths: int, n_steps: int,
                       antithetic: bool = True) -> dict:
        """
        Simulate correlated Gold and EUR/USD paths.

        Args:
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps
            antithetic: Use antithetic variates for variance reduction

        Returns:
            Dictionary with gold_paths, eurusd_paths, times, and barrier info
        """
        dt = self.contract.tenor / n_steps
        sqrt_dt = np.sqrt(dt)

        # If using antithetic variates, generate half the paths and mirror
        actual_paths = n_paths // 2 if antithetic else n_paths

        W1, W2 = self._generate_correlated_normals(actual_paths, n_steps)

        if antithetic:
            W1 = np.vstack([W1, -W1])
            W2 = np.vstack([W2, -W2])
            actual_paths = n_paths

        # Initialize paths
        gold_paths = np.zeros((actual_paths, n_steps + 1))
        eurusd_paths = np.zeros((actual_paths, n_steps + 1))

        gold_paths[:, 0] = self.market.gold_spot
        eurusd_paths[:, 0] = self.market.eurusd_spot

        # Drift terms under risk-neutral measure
        # Gold: drift = r_USD - convenience_yield
        gold_drift = self.market.r_usd - self.market.gold_yield
        # EUR/USD: drift = r_EUR - r_USD (interest rate differential)
        eurusd_drift = self.market.r_eur - self.market.r_usd

        # Simulate paths using log-normal dynamics
        for t in range(n_steps):
            # Gold: dS/S = (r_USD - q) dt + σ_S dW^S
            gold_paths[:, t+1] = gold_paths[:, t] * np.exp(
                (gold_drift - 0.5 * self.market.sigma_gold**2) * dt +
                self.market.sigma_gold * sqrt_dt * W1[:, t]
            )

            # EUR/USD: dX/X = (r_EUR - r_USD) dt + σ_X dW^X
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

    def check_barrier_breach(self, eurusd_paths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check for barrier breaches in EUR/USD paths.

        Returns:
            knocked_out: Boolean array indicating if path was knocked out
            knockout_idx: Index of first barrier breach (-1 if no breach)
        """
        n_paths, n_steps = eurusd_paths.shape

        # Check barrier conditions
        below_lower = eurusd_paths < self.contract.barrier_lower
        above_upper = eurusd_paths > self.contract.barrier_upper
        breached = below_lower | above_upper

        # Find first breach index for each path
        knocked_out = np.any(breached, axis=1)
        knockout_idx = np.full(n_paths, -1, dtype=int)

        for i in range(n_paths):
            if knocked_out[i]:
                knockout_idx[i] = np.argmax(breached[i])

        return knocked_out, knockout_idx


class StructuredForwardPricer:
    """
    Prices the structured gold forward with double knock-out barriers.

    The product pays:
    - Z Group: N × (P_τ - K) / K
    - A Bank: N × (K - P_τ) / K

    where τ is maturity T or knock-out time (whichever comes first).
    """

    def __init__(self, market: MarketData, contract: ContractTerms):
        self.market = market
        self.contract = contract

    def price_monte_carlo(self, n_paths: int = 100000, n_steps: int = 504,
                          seed: Optional[int] = 42, antithetic: bool = True,
                          control_variate: bool = True) -> dict:
        """
        Price the structured product using Monte Carlo simulation.

        Args:
            n_paths: Number of simulation paths
            n_steps: Number of time steps (504 ≈ 252 trading days × 2 years)
            seed: Random seed for reproducibility
            antithetic: Use antithetic variates
            control_variate: Use control variate (vanilla forward)

        Returns:
            Dictionary with pricing results and statistics
        """
        simulator = CorrelatedGBMSimulator(self.market, self.contract, seed)

        # Simulate paths
        sim_result = simulator.simulate_paths(n_paths, n_steps, antithetic)
        gold_paths = sim_result['gold_paths']
        eurusd_paths = sim_result['eurusd_paths']
        times = sim_result['times']
        dt = sim_result['dt']

        # Check for barrier breaches
        knocked_out, knockout_idx = simulator.check_barrier_breach(eurusd_paths)

        # Calculate settlement prices and times
        n_actual_paths = gold_paths.shape[0]
        settlement_prices = np.zeros(n_actual_paths)
        settlement_times = np.zeros(n_actual_paths)

        for i in range(n_actual_paths):
            if knocked_out[i]:
                # Knocked out - settle at knockout time
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

        # Discount to present value (using EUR rate for EUR-denominated payoff)
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
            analytical_vanilla = N * (forward_price - K) / K * np.exp(-self.market.r_eur * self.contract.tenor)

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

        # Confidence interval
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

    def compute_greeks(self, base_result: dict, bump_size: float = 0.01,
                       n_paths: int = 50000, seed: int = 42) -> dict:
        """
        Compute option Greeks using finite difference method.

        Returns:
            Dictionary with Delta, Gamma, Vega, Rho, and correlation sensitivity
        """
        base_price = base_result['price_zgroup']

        # Delta (gold)
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

        # Vega (gold volatility)
        vega_bump = 0.01  # 1% vol bump
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

        # Rho (EUR interest rate sensitivity)
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


def run_sensitivity_analysis(market: MarketData, contract: ContractTerms,
                            n_paths: int = 50000) -> pd.DataFrame:
    """
    Run sensitivity analysis on key parameters.

    Returns:
        DataFrame with sensitivity results
    """
    results = []

    # Gold spot sensitivity
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


if __name__ == "__main__":
    print("=" * 70)
    print("GAAIF Challenge - Structured Forward Pricing Model")
    print("=" * 70)

    # Initialize market data and contract terms
    market = MarketData()
    contract = ContractTerms()

    print("\n" + "=" * 70)
    print("MARKET DATA")
    print("=" * 70)
    print(f"Gold Spot Price:     ${market.gold_spot:,.2f}/oz")
    print(f"EUR/USD Spot:        {market.eurusd_spot:.4f}")
    print(f"EUR Risk-Free Rate:  {market.r_eur*100:.2f}%")
    print(f"USD Risk-Free Rate:  {market.r_usd*100:.2f}%")
    print(f"Gold Volatility:     {market.sigma_gold*100:.1f}%")
    print(f"EUR/USD Volatility:  {market.sigma_eurusd*100:.1f}%")
    print(f"Correlation:         {market.rho:.2f}")

    print("\n" + "=" * 70)
    print("CONTRACT TERMS")
    print("=" * 70)
    print(f"Notional:            EUR {contract.notional/1e6:,.0f} Million")
    print(f"Strike Price (K):    ${contract.strike:,.2f}/oz")
    print(f"Tenor:               {contract.tenor} years")
    print(f"Lower Barrier:       EUR/USD {contract.barrier_lower}")
    print(f"Upper Barrier:       EUR/USD {contract.barrier_upper}")

    # Price the product
    print("\n" + "=" * 70)
    print("MONTE CARLO PRICING")
    print("=" * 70)

    pricer = StructuredForwardPricer(market, contract)
    result = pricer.price_monte_carlo(n_paths=100000, n_steps=504, seed=42)

    print(f"\nSimulation Parameters:")
    print(f"  Number of Paths:   {result['n_paths']:,}")
    print(f"  Number of Steps:   {result['n_steps']}")
    print(f"  Variance Reduction: Antithetic + Control Variate")

    print(f"\nPricing Results:")
    print(f"  Z Group PV:        EUR {result['price_zgroup']:,.2f}")
    print(f"  A Bank PV:         EUR {result['price_abank']:,.2f}")
    print(f"  Standard Error:    EUR {result['std_error']:,.2f}")
    print(f"  95% CI:            [{result['ci_95_lower']:,.2f}, {result['ci_95_upper']:,.2f}]")

    print(f"\nBarrier Analysis:")
    print(f"  Knockout Rate:     {result['knockout_rate']*100:.2f}%")
    print(f"  Avg KO Time:       {result['avg_knockout_time']:.2f} years" if not np.isnan(result['avg_knockout_time']) else "  Avg KO Time:       N/A")
    print(f"  Lower Breach:      {result['lower_breach_rate']*100:.2f}%")
    print(f"  Upper Breach:      {result['upper_breach_rate']*100:.2f}%")

    # Compute Greeks
    print("\n" + "=" * 70)
    print("RISK SENSITIVITIES (GREEKS)")
    print("=" * 70)

    greeks = pricer.compute_greeks(result, n_paths=50000)

    print(f"  Delta (Gold):      EUR {greeks['delta_gold']:,.2f} per $1 gold move")
    print(f"  Gamma (Gold):      EUR {greeks['gamma_gold']:,.2f}")
    print(f"  Delta (EUR/USD):   EUR {greeks['delta_eurusd']:,.2f} per 0.01 FX move")
    print(f"  Vega (Gold):       EUR {greeks['vega_gold']:,.2f} per 1% vol")
    print(f"  Rho (EUR Rate):    EUR {greeks['rho_eur']:,.2f} per 1bp")
    print(f"  Corr Sensitivity:  EUR {greeks['correlation_sensitivity']:,.2f} per 0.05 corr")

    print("\n" + "=" * 70)
    print("Pricing Complete")
    print("=" * 70)
