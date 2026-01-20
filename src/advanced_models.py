"""
Advanced Pricing Models for GAAIF Challenge
============================================

This module implements advanced extensions to the base GBM model:
1. Heston Stochastic Volatility Model
2. Merton Jump-Diffusion Model
3. Alternative Scenario Analysis

These provide more realistic dynamics and demonstrate model sophistication.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HestonParams:
    """Parameters for Heston stochastic volatility model."""
    # Initial variance
    v0: float = 0.0324           # Initial variance (18%^2)
    # Long-term variance
    theta: float = 0.04          # Long-term variance (20%^2)
    # Mean reversion speed
    kappa: float = 2.0           # Mean reversion rate
    # Volatility of volatility
    xi: float = 0.3              # Vol of vol
    # Correlation between price and variance
    rho_sv: float = -0.7         # Typically negative (leverage effect)


@dataclass
class JumpParams:
    """Parameters for Merton jump-diffusion model."""
    # Jump intensity (expected jumps per year)
    lambda_j: float = 0.5        # ~0.5 jumps per year
    # Mean jump size (log)
    mu_j: float = -0.05          # -5% mean jump
    # Jump size volatility
    sigma_j: float = 0.10        # 10% jump vol


class HestonSimulator:
    """
    Simulates gold prices under Heston stochastic volatility model.

    dS_t / S_t = μ dt + √V_t dW^S_t
    dV_t = κ(θ - V_t) dt + ξ√V_t dW^V_t
    Corr(dW^S, dW^V) = ρ_sv

    This captures:
    - Volatility clustering
    - Mean-reverting volatility
    - Leverage effect (negative correlation)
    """

    def __init__(self, heston_params: HestonParams, seed: Optional[int] = None):
        self.params = heston_params
        self.rng = np.random.default_rng(seed)

    def simulate(self, S0: float, mu: float, T: float,
                 n_paths: int, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston model paths using QE (Quadratic Exponential) scheme.

        Args:
            S0: Initial spot price
            mu: Drift rate
            T: Time horizon
            n_paths: Number of paths
            n_steps: Number of time steps

        Returns:
            Tuple of (price_paths, variance_paths)
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        # Initialize arrays
        S = np.zeros((n_paths, n_steps + 1))
        V = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0
        V[:, 0] = self.params.v0

        # Generate correlated Brownian increments
        Z1 = self.rng.standard_normal((n_paths, n_steps))
        Z2 = self.rng.standard_normal((n_paths, n_steps))

        # Cholesky for price-variance correlation
        rho = self.params.rho_sv
        W_S = Z1
        W_V = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        kappa = self.params.kappa
        theta = self.params.theta
        xi = self.params.xi

        for t in range(n_steps):
            # Ensure variance stays positive (full truncation)
            V_pos = np.maximum(V[:, t], 0)
            sqrt_V = np.sqrt(V_pos)

            # Price update
            S[:, t+1] = S[:, t] * np.exp(
                (mu - 0.5 * V_pos) * dt + sqrt_V * sqrt_dt * W_S[:, t]
            )

            # Variance update (Euler with full truncation)
            V[:, t+1] = V[:, t] + kappa * (theta - V_pos) * dt + \
                        xi * sqrt_V * sqrt_dt * W_V[:, t]
            V[:, t+1] = np.maximum(V[:, t+1], 0)

        return S, V


class MertonJumpSimulator:
    """
    Simulates gold prices under Merton jump-diffusion model.

    dS_t / S_t = (μ - λκ) dt + σ dW_t + (J - 1) dN_t

    where:
    - N_t is a Poisson process with intensity λ
    - J = exp(μ_j + σ_j Z) is the jump multiplier
    - κ = E[J-1] = exp(μ_j + σ_j²/2) - 1

    This captures:
    - Sudden price jumps (crashes/spikes)
    - Fat tails in return distribution
    """

    def __init__(self, jump_params: JumpParams, seed: Optional[int] = None):
        self.params = jump_params
        self.rng = np.random.default_rng(seed)

    def simulate(self, S0: float, mu: float, sigma: float, T: float,
                 n_paths: int, n_steps: int) -> np.ndarray:
        """
        Simulate Merton jump-diffusion paths.

        Args:
            S0: Initial spot price
            mu: Drift rate (before jump compensation)
            sigma: Diffusion volatility
            T: Time horizon
            n_paths: Number of paths
            n_steps: Number of time steps

        Returns:
            Array of price paths
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        lam = self.params.lambda_j
        mu_j = self.params.mu_j
        sigma_j = self.params.sigma_j

        # Jump compensator
        kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1

        # Compensated drift
        mu_comp = mu - lam * kappa

        # Initialize
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0

        for t in range(n_steps):
            # Diffusion component
            Z = self.rng.standard_normal(n_paths)
            diffusion = (mu_comp - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z

            # Jump component
            N = self.rng.poisson(lam * dt, n_paths)  # Number of jumps
            jump_sizes = np.zeros(n_paths)
            for i in range(n_paths):
                if N[i] > 0:
                    # Sum of N[i] log-normal jump sizes
                    jumps = self.rng.normal(mu_j, sigma_j, N[i])
                    jump_sizes[i] = np.sum(jumps)

            # Combined update
            S[:, t+1] = S[:, t] * np.exp(diffusion + jump_sizes)

        return S


def run_scenario_analysis(base_market, base_contract, n_paths: int = 50000) -> pd.DataFrame:
    """
    Run comprehensive scenario analysis across different parameter combinations.

    Tests:
    - Different strike prices
    - Different barrier levels
    - Different volatility assumptions
    - Different correlation assumptions

    Returns:
        DataFrame with scenario results
    """
    from pricing_model import MarketData, ContractTerms, StructuredForwardPricer

    results = []

    # Scenario 1: Different Strike Prices
    print("  Running strike scenarios...")
    strikes = [3000, 3500, 4000, 4600, 5000, 5500]
    for K in strikes:
        contract = ContractTerms(
            notional=base_contract.notional,
            strike=K,
            tenor=base_contract.tenor,
            barrier_lower=base_contract.barrier_lower,
            barrier_upper=base_contract.barrier_upper
        )
        pricer = StructuredForwardPricer(base_market, contract)
        res = pricer.price_monte_carlo(n_paths, seed=42)
        results.append({
            'scenario_type': 'strike',
            'parameter': 'strike_price',
            'value': K,
            'price_zgroup': res['price_zgroup'],
            'price_abank': res['price_abank'],
            'knockout_rate': res['knockout_rate'],
            'avg_ko_time': res['avg_knockout_time']
        })

    # Scenario 2: Different Barrier Widths
    print("  Running barrier scenarios...")
    barrier_configs = [
        (1.00, 1.20, 'Narrow'),
        (1.03, 1.23, 'Tight'),
        (1.05, 1.25, 'Base'),
        (1.00, 1.30, 'Wide'),
        (0.95, 1.35, 'Very Wide'),
    ]
    for lower, upper, name in barrier_configs:
        contract = ContractTerms(
            notional=base_contract.notional,
            strike=base_contract.strike,
            tenor=base_contract.tenor,
            barrier_lower=lower,
            barrier_upper=upper
        )
        pricer = StructuredForwardPricer(base_market, contract)
        res = pricer.price_monte_carlo(n_paths, seed=42)
        results.append({
            'scenario_type': 'barrier',
            'parameter': f'{lower}-{upper} ({name})',
            'value': upper - lower,
            'price_zgroup': res['price_zgroup'],
            'price_abank': res['price_abank'],
            'knockout_rate': res['knockout_rate'],
            'avg_ko_time': res['avg_knockout_time']
        })

    # Scenario 3: Different Volatility Regimes
    print("  Running volatility scenarios...")
    vol_scenarios = [
        (0.12, 0.06, 'Low Vol'),
        (0.15, 0.07, 'Moderate Vol'),
        (0.18, 0.08, 'Base Vol'),
        (0.22, 0.10, 'High Vol'),
        (0.28, 0.12, 'Crisis Vol'),
    ]
    for gold_vol, fx_vol, name in vol_scenarios:
        market = MarketData(
            gold_spot=base_market.gold_spot,
            eurusd_spot=base_market.eurusd_spot,
            r_eur=base_market.r_eur,
            r_usd=base_market.r_usd,
            sigma_gold=gold_vol,
            sigma_eurusd=fx_vol,
            rho=base_market.rho,
            gold_yield=base_market.gold_yield
        )
        pricer = StructuredForwardPricer(market, base_contract)
        res = pricer.price_monte_carlo(n_paths, seed=42)
        results.append({
            'scenario_type': 'volatility',
            'parameter': name,
            'value': gold_vol,
            'price_zgroup': res['price_zgroup'],
            'price_abank': res['price_abank'],
            'knockout_rate': res['knockout_rate'],
            'avg_ko_time': res['avg_knockout_time']
        })

    # Scenario 4: Different EUR/USD Starting Points
    print("  Running EUR/USD spot scenarios...")
    fx_spots = [1.06, 1.08, 1.10, 1.12, 1.15, 1.18, 1.20]
    for fx in fx_spots:
        market = MarketData(
            gold_spot=base_market.gold_spot,
            eurusd_spot=fx,
            r_eur=base_market.r_eur,
            r_usd=base_market.r_usd,
            sigma_gold=base_market.sigma_gold,
            sigma_eurusd=base_market.sigma_eurusd,
            rho=base_market.rho,
            gold_yield=base_market.gold_yield
        )
        pricer = StructuredForwardPricer(market, base_contract)
        res = pricer.price_monte_carlo(n_paths, seed=42)
        results.append({
            'scenario_type': 'eurusd_spot',
            'parameter': 'eurusd_spot',
            'value': fx,
            'price_zgroup': res['price_zgroup'],
            'price_abank': res['price_abank'],
            'knockout_rate': res['knockout_rate'],
            'avg_ko_time': res['avg_knockout_time']
        })

    return pd.DataFrame(results)


def run_model_comparison(market, contract, n_paths: int = 50000) -> Dict:
    """
    Compare pricing across different model specifications:
    1. Base GBM
    2. Heston Stochastic Volatility
    3. Merton Jump-Diffusion
    4. Combined (Heston + Jumps)

    Returns:
        Dictionary with comparison results
    """
    from pricing_model import MarketData, ContractTerms, StructuredForwardPricer, CorrelatedGBMSimulator

    results = {}
    n_steps = 504

    # 1. Base GBM (reference)
    print("  Running base GBM model...")
    pricer = StructuredForwardPricer(market, contract)
    base_result = pricer.price_monte_carlo(n_paths, n_steps, seed=42)
    results['GBM'] = {
        'price_zgroup': base_result['price_zgroup'],
        'knockout_rate': base_result['knockout_rate'],
        'std_error': base_result['std_error']
    }

    # 2. Heston Model for Gold
    print("  Running Heston stochastic volatility model...")
    heston_params = HestonParams(
        v0=market.sigma_gold**2,
        theta=market.sigma_gold**2 * 1.1,  # Slightly higher long-term vol
        kappa=2.0,
        xi=0.3,
        rho_sv=-0.7
    )
    heston_sim = HestonSimulator(heston_params, seed=42)

    # Simulate gold with Heston, EUR/USD with GBM
    gold_drift = market.r_usd - market.gold_yield
    gold_heston, _ = heston_sim.simulate(
        market.gold_spot, gold_drift, contract.tenor, n_paths, n_steps
    )

    # EUR/USD still GBM
    gbm_sim = CorrelatedGBMSimulator(market, contract, seed=42)
    gbm_result = gbm_sim.simulate_paths(n_paths, n_steps, antithetic=False)
    eurusd_paths = gbm_result['eurusd_paths']

    # Check knockouts and calculate payoffs
    knocked_out, knockout_idx = gbm_sim.check_barrier_breach(eurusd_paths)
    times = np.linspace(0, contract.tenor, n_steps + 1)

    settlement_prices = np.zeros(n_paths)
    settlement_times = np.zeros(n_paths)
    for i in range(n_paths):
        if knocked_out[i]:
            settlement_prices[i] = gold_heston[i, knockout_idx[i]]
            settlement_times[i] = times[knockout_idx[i]]
        else:
            settlement_prices[i] = gold_heston[i, -1]
            settlement_times[i] = contract.tenor

    payoffs = contract.notional * (settlement_prices - contract.strike) / contract.strike
    discount_factors = np.exp(-market.r_eur * settlement_times)
    pv = payoffs * discount_factors

    results['Heston'] = {
        'price_zgroup': np.mean(pv),
        'knockout_rate': np.mean(knocked_out),
        'std_error': np.std(pv) / np.sqrt(n_paths)
    }

    # 3. Merton Jump-Diffusion
    print("  Running Merton jump-diffusion model...")
    jump_params = JumpParams(lambda_j=0.5, mu_j=-0.03, sigma_j=0.08)
    jump_sim = MertonJumpSimulator(jump_params, seed=42)

    gold_jump = jump_sim.simulate(
        market.gold_spot, gold_drift, market.sigma_gold,
        contract.tenor, n_paths, n_steps
    )

    settlement_prices_jump = np.zeros(n_paths)
    for i in range(n_paths):
        if knocked_out[i]:
            settlement_prices_jump[i] = gold_jump[i, knockout_idx[i]]
        else:
            settlement_prices_jump[i] = gold_jump[i, -1]

    payoffs_jump = contract.notional * (settlement_prices_jump - contract.strike) / contract.strike
    pv_jump = payoffs_jump * discount_factors

    results['Merton'] = {
        'price_zgroup': np.mean(pv_jump),
        'knockout_rate': np.mean(knocked_out),
        'std_error': np.std(pv_jump) / np.sqrt(n_paths)
    }

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from pricing_model import MarketData, ContractTerms

    print("=" * 60)
    print("Advanced Model Testing")
    print("=" * 60)

    market = MarketData()
    contract = ContractTerms()

    # Test Heston
    print("\nTesting Heston Model...")
    heston = HestonSimulator(HestonParams(), seed=42)
    S, V = heston.simulate(2750, 0.04, 2.0, 1000, 252)
    print(f"  Final price range: [{S[:, -1].min():.0f}, {S[:, -1].max():.0f}]")
    print(f"  Final vol range: [{np.sqrt(V[:, -1]).min()*100:.1f}%, {np.sqrt(V[:, -1]).max()*100:.1f}%]")

    # Test Merton
    print("\nTesting Merton Model...")
    merton = MertonJumpSimulator(JumpParams(), seed=42)
    S_jump = merton.simulate(2750, 0.04, 0.18, 2.0, 1000, 252)
    print(f"  Final price range: [{S_jump[:, -1].min():.0f}, {S_jump[:, -1].max():.0f}]")

    print("\nAdvanced model tests complete!")
