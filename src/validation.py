"""
Validation and Testing Module for GAAIF Challenge
==================================================

This module provides:
1. Input validation for market data and contract terms
2. Convergence analysis for Monte Carlo
3. Analytical benchmarks for validation
4. Stress testing scenarios
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


def validate_market_data(market) -> List[str]:
    """
    Validate market data inputs for reasonableness.

    Returns:
        List of warning/error messages (empty if all valid)
    """
    errors = []

    # Spot prices must be positive
    if market.gold_spot <= 0:
        errors.append(f"ERROR: Gold spot must be positive, got {market.gold_spot}")
    if market.eurusd_spot <= 0:
        errors.append(f"ERROR: EUR/USD spot must be positive, got {market.eurusd_spot}")

    # Volatilities must be positive and reasonable (0-100%)
    if not 0 < market.sigma_gold < 1.0:
        errors.append(f"WARNING: Gold volatility {market.sigma_gold*100:.1f}% seems unusual")
    if not 0 < market.sigma_eurusd < 0.5:
        errors.append(f"WARNING: EUR/USD volatility {market.sigma_eurusd*100:.1f}% seems unusual")

    # Correlation must be in [-1, 1]
    if not -1 <= market.rho <= 1:
        errors.append(f"ERROR: Correlation must be in [-1,1], got {market.rho}")

    # Interest rates should be reasonable (-5% to 20%)
    if not -0.05 <= market.r_eur <= 0.20:
        errors.append(f"WARNING: EUR rate {market.r_eur*100:.1f}% seems unusual")
    if not -0.05 <= market.r_usd <= 0.20:
        errors.append(f"WARNING: USD rate {market.r_usd*100:.1f}% seems unusual")

    return errors


def validate_contract_terms(contract, market) -> List[str]:
    """
    Validate contract terms for reasonableness.

    Returns:
        List of warning/error messages
    """
    errors = []

    # Notional must be positive
    if contract.notional <= 0:
        errors.append(f"ERROR: Notional must be positive")

    # Strike must be positive
    if contract.strike <= 0:
        errors.append(f"ERROR: Strike must be positive")

    # Check strike vs forward price
    forward_price = market.gold_spot * np.exp(
        (market.r_usd - market.gold_yield) * contract.tenor
    )
    strike_ratio = contract.strike / forward_price
    if strike_ratio > 1.5:
        errors.append(f"WARNING: Strike ${contract.strike} is {(strike_ratio-1)*100:.0f}% above "
                     f"forward price ${forward_price:.0f} - deeply OTM for Z Group")
    elif strike_ratio < 0.7:
        errors.append(f"WARNING: Strike ${contract.strike} is {(1-strike_ratio)*100:.0f}% below "
                     f"forward price ${forward_price:.0f} - deeply ITM for Z Group")

    # Barriers must make sense
    if contract.barrier_lower >= contract.barrier_upper:
        errors.append(f"ERROR: Lower barrier must be < upper barrier")

    if not contract.barrier_lower < market.eurusd_spot < contract.barrier_upper:
        errors.append(f"ERROR: Current EUR/USD {market.eurusd_spot} is outside barriers!")

    # Check barrier distance from spot
    lower_dist = (market.eurusd_spot - contract.barrier_lower) / market.eurusd_spot
    upper_dist = (contract.barrier_upper - market.eurusd_spot) / market.eurusd_spot

    if lower_dist < 0.05:
        errors.append(f"WARNING: Lower barrier is only {lower_dist*100:.1f}% from spot - "
                     f"high knockout probability expected")
    if upper_dist < 0.05:
        errors.append(f"WARNING: Upper barrier is only {upper_dist*100:.1f}% from spot - "
                     f"high knockout probability expected")

    # Tenor must be positive
    if contract.tenor <= 0:
        errors.append(f"ERROR: Tenor must be positive")

    return errors


def run_convergence_analysis(pricer, path_counts: List[int] = None,
                             n_steps: int = 504, seed: int = 42) -> pd.DataFrame:
    """
    Run Monte Carlo convergence analysis.

    Shows that the price estimate converges as number of paths increases.

    Args:
        pricer: StructuredForwardPricer instance
        path_counts: List of path counts to test
        n_steps: Number of time steps
        seed: Random seed

    Returns:
        DataFrame with convergence results
    """
    if path_counts is None:
        path_counts = [1000, 2500, 5000, 10000, 25000, 50000, 100000, 200000]

    results = []

    print("Running convergence analysis...")
    for n_paths in path_counts:
        print(f"  Testing {n_paths:,} paths...")
        result = pricer.price_monte_carlo(n_paths=n_paths, n_steps=n_steps, seed=seed)

        results.append({
            'n_paths': n_paths,
            'price_zgroup': result['price_zgroup'],
            'std_error': result['std_error'],
            'ci_width': result['ci_95_upper'] - result['ci_95_lower'],
            'knockout_rate': result['knockout_rate']
        })

    df = pd.DataFrame(results)

    # Calculate convergence metrics
    df['price_change'] = df['price_zgroup'].diff().abs()
    df['relative_se'] = df['std_error'] / df['price_zgroup'].abs() * 100

    return df


def plot_convergence(convergence_df: pd.DataFrame, output_path: str = None):
    """
    Plot Monte Carlo convergence analysis.

    Args:
        convergence_df: DataFrame from run_convergence_analysis
        output_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Price convergence
    ax1 = axes[0, 0]
    ax1.plot(convergence_df['n_paths'], convergence_df['price_zgroup'] / 1e6,
            'b-o', linewidth=2, markersize=6)
    ax1.fill_between(convergence_df['n_paths'],
                     (convergence_df['price_zgroup'] - 1.96*convergence_df['std_error']) / 1e6,
                     (convergence_df['price_zgroup'] + 1.96*convergence_df['std_error']) / 1e6,
                     alpha=0.3, color='blue')
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Paths')
    ax1.set_ylabel('Price (EUR Millions)')
    ax1.set_title('Price Convergence with 95% CI')
    ax1.grid(True, alpha=0.3)

    # Standard error decay
    ax2 = axes[0, 1]
    ax2.plot(convergence_df['n_paths'], convergence_df['std_error'] / 1e6,
            'r-o', linewidth=2, markersize=6, label='Actual SE')
    # Theoretical SE decay (1/sqrt(n))
    theoretical_se = convergence_df['std_error'].iloc[0] * np.sqrt(convergence_df['n_paths'].iloc[0]) / np.sqrt(convergence_df['n_paths'])
    ax2.plot(convergence_df['n_paths'], theoretical_se / 1e6,
            'k--', linewidth=1, label='Theoretical (1/√n)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Paths')
    ax2.set_ylabel('Standard Error (EUR Millions)')
    ax2.set_title('Standard Error Decay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Knockout rate stability
    ax3 = axes[1, 0]
    ax3.plot(convergence_df['n_paths'], convergence_df['knockout_rate'] * 100,
            'g-o', linewidth=2, markersize=6)
    ax3.set_xscale('log')
    ax3.set_xlabel('Number of Paths')
    ax3.set_ylabel('Knockout Rate (%)')
    ax3.set_title('Knockout Rate Stability')
    ax3.grid(True, alpha=0.3)

    # Relative standard error
    ax4 = axes[1, 1]
    ax4.bar(range(len(convergence_df)), convergence_df['relative_se'].abs(),
           color='purple', alpha=0.7)
    ax4.set_xticks(range(len(convergence_df)))
    ax4.set_xticklabels([f'{x//1000}K' for x in convergence_df['n_paths']], rotation=45)
    ax4.set_xlabel('Number of Paths')
    ax4.set_ylabel('Relative Std Error (%)')
    ax4.set_title('Relative Precision')
    ax4.axhline(y=1, color='red', linestyle='--', label='1% threshold')
    ax4.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to: {output_path}")

    plt.close()
    return fig


def analytical_forward_price(market, contract) -> Dict[str, float]:
    """
    Calculate analytical forward price for validation.

    The vanilla gold forward (without barriers) has a known analytical price.
    This serves as a sanity check for the Monte Carlo.

    Returns:
        Dictionary with analytical benchmarks
    """
    # Gold forward price under risk-neutral measure
    F = market.gold_spot * np.exp(
        (market.r_usd - market.gold_yield) * contract.tenor
    )

    # Vanilla forward payoff (no barriers)
    # E[e^{-rT} * N * (P_T - K) / K] = e^{-rT} * N * (F - K) / K
    vanilla_pv = np.exp(-market.r_eur * contract.tenor) * \
                 contract.notional * (F - contract.strike) / contract.strike

    # Expected EUR/USD at maturity (under risk-neutral)
    F_fx = market.eurusd_spot * np.exp(
        (market.r_eur - market.r_usd) * contract.tenor
    )

    return {
        'gold_forward': F,
        'eurusd_forward': F_fx,
        'vanilla_forward_pv': vanilla_pv,
        'gold_spot': market.gold_spot,
        'strike': contract.strike,
        'moneyness': F / contract.strike  # < 1 means OTM for Z Group
    }


def run_stress_tests(market, contract, pricer_class, n_paths: int = 30000) -> pd.DataFrame:
    """
    Run stress tests under extreme market scenarios.

    Tests:
    - Volatility spike (2x normal)
    - Correlation breakdown
    - Rate shock scenarios
    - Spot price moves

    Returns:
        DataFrame with stress test results
    """
    from pricing_model import MarketData, ContractTerms

    results = []

    scenarios = [
        ('Base Case', {}),
        ('Gold Vol +50%', {'sigma_gold': market.sigma_gold * 1.5}),
        ('Gold Vol +100%', {'sigma_gold': market.sigma_gold * 2.0}),
        ('FX Vol +50%', {'sigma_eurusd': market.sigma_eurusd * 1.5}),
        ('FX Vol +100%', {'sigma_eurusd': market.sigma_eurusd * 2.0}),
        ('Correlation = 0', {'rho': 0.0}),
        ('Correlation = -0.5', {'rho': -0.5}),
        ('Correlation = +0.3', {'rho': 0.3}),
        ('EUR Rate +100bp', {'r_eur': market.r_eur + 0.01}),
        ('USD Rate +100bp', {'r_usd': market.r_usd + 0.01}),
        ('Gold Spot +10%', {'gold_spot': market.gold_spot * 1.1}),
        ('Gold Spot -10%', {'gold_spot': market.gold_spot * 0.9}),
        ('EUR/USD at 1.06', {'eurusd_spot': 1.06}),
        ('EUR/USD at 1.15', {'eurusd_spot': 1.15}),
    ]

    print("Running stress tests...")
    for name, overrides in scenarios:
        print(f"  {name}...")

        # Create modified market data
        params = {
            'gold_spot': market.gold_spot,
            'eurusd_spot': market.eurusd_spot,
            'r_eur': market.r_eur,
            'r_usd': market.r_usd,
            'sigma_gold': market.sigma_gold,
            'sigma_eurusd': market.sigma_eurusd,
            'rho': market.rho,
            'gold_yield': market.gold_yield
        }
        params.update(overrides)

        stressed_market = MarketData(**params)
        pricer = pricer_class(stressed_market, contract)
        result = pricer.price_monte_carlo(n_paths=n_paths, seed=42)

        results.append({
            'scenario': name,
            'price_zgroup': result['price_zgroup'],
            'price_change_pct': 0 if name == 'Base Case' else None,
            'knockout_rate': result['knockout_rate'],
            'std_error': result['std_error']
        })

    df = pd.DataFrame(results)

    # Calculate changes from base case
    base_price = df.loc[df['scenario'] == 'Base Case', 'price_zgroup'].values[0]
    df['price_change_pct'] = (df['price_zgroup'] - base_price) / abs(base_price) * 100

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from pricing_model import MarketData, ContractTerms, StructuredForwardPricer

    print("=" * 60)
    print("Validation Module Test")
    print("=" * 60)

    market = MarketData()
    contract = ContractTerms()

    # Validate inputs
    print("\n1. Input Validation:")
    print("-" * 40)
    errors = validate_market_data(market)
    errors.extend(validate_contract_terms(contract, market))
    if errors:
        for e in errors:
            print(f"  {e}")
    else:
        print("  All inputs valid!")

    # Analytical benchmarks
    print("\n2. Analytical Benchmarks:")
    print("-" * 40)
    benchmarks = analytical_forward_price(market, contract)
    print(f"  Gold Forward (2Y):     ${benchmarks['gold_forward']:,.0f}")
    print(f"  EUR/USD Forward (2Y):  {benchmarks['eurusd_forward']:.4f}")
    print(f"  Vanilla Forward PV:    EUR {benchmarks['vanilla_forward_pv']:,.0f}")
    print(f"  Moneyness (F/K):       {benchmarks['moneyness']:.2%}")

    # Convergence (quick test)
    print("\n3. Quick Convergence Test:")
    print("-" * 40)
    pricer = StructuredForwardPricer(market, contract)
    conv_df = run_convergence_analysis(pricer, path_counts=[5000, 20000, 50000])
    print(conv_df[['n_paths', 'price_zgroup', 'std_error', 'knockout_rate']].to_string(index=False))

    print("\nValidation complete!")
