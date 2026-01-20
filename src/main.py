"""
GAAIF Challenge - Main Execution Script
=======================================

This script runs the complete analysis and generates all deliverables:
1. Monte Carlo pricing
2. Sensitivity analysis
3. Greeks computation
4. Visualizations
5. Excel data output

Author: GAAIF Challenge Submission
Date: January 2026
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# Ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pricing_model import (
    MarketData, ContractTerms, StructuredForwardPricer,
    run_sensitivity_analysis
)
from market_data import (
    MarketDataFetcher, VolatilityEstimator, create_sample_market_data
)
from visualization import ProductVisualizer


def create_output_directories():
    """Create necessary output directories."""
    dirs = ['../output', '../data']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def generate_excel_output(pricing_result: dict, sensitivity_df: pd.DataFrame,
                          greeks: dict, market: MarketData, contract: ContractTerms,
                          output_path: str):
    """
    Generate Excel file with all analysis data.

    Args:
        pricing_result: Monte Carlo pricing results
        sensitivity_df: Sensitivity analysis results
        greeks: Greeks values
        market: Market data parameters
        contract: Contract terms
        output_path: Path for Excel file
    """
    print("\nGenerating Excel output...")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        # Sheet 1: Market Data Parameters
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

        # Sheet 6: Sample Paths Statistics
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

        # Sheet 7: Sample of Individual Path Data
        n_sample = min(1000, len(settlement_prices))
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


def run_convergence_analysis(market: MarketData, contract: ContractTerms) -> pd.DataFrame:
    """
    Run convergence analysis with different numbers of paths.

    Returns:
        DataFrame with convergence results
    """
    print("\nRunning convergence analysis...")

    path_counts = [1000, 5000, 10000, 25000, 50000, 100000, 200000]
    results = []

    pricer = StructuredForwardPricer(market, contract)

    for n_paths in path_counts:
        print(f"  Testing {n_paths:,} paths...")
        res = pricer.price_monte_carlo(n_paths=n_paths, n_steps=504, seed=42)
        results.append({
            'n_paths': n_paths,
            'price_zgroup': res['price_zgroup'],
            'std_error': res['std_error'],
            'knockout_rate': res['knockout_rate']
        })

    return pd.DataFrame(results)


def main():
    """Main execution function."""
    print("=" * 70)
    print("GAAIF CHALLENGE - STRUCTURED PRODUCT PRICING")
    print("=" * 70)
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directories
    create_output_directories()

    # Initialize market data with realistic parameters
    print("\n" + "-" * 70)
    print("INITIALIZING MARKET DATA")
    print("-" * 70)

    market = MarketData(
        gold_spot=2750.0,        # Current gold price
        eurusd_spot=1.08,        # Current EUR/USD
        r_eur=0.025,             # ECB rate ~2.5%
        r_usd=0.045,             # Fed rate ~4.5%
        sigma_gold=0.18,         # Gold volatility ~18%
        sigma_eurusd=0.08,       # EUR/USD volatility ~8%
        rho=-0.25,               # Typical negative correlation
        gold_yield=0.005         # Gold convenience yield
    )

    contract = ContractTerms(
        notional=500_000_000,    # EUR 500M
        strike=4600.0,           # Strike $4,600/oz
        tenor=2.0,               # 2 years
        barrier_lower=1.05,      # Lower barrier
        barrier_upper=1.25       # Upper barrier
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

    # Main pricing
    print("\n" + "-" * 70)
    print("MONTE CARLO PRICING")
    print("-" * 70)

    pricer = StructuredForwardPricer(market, contract)

    print("\nRunning Monte Carlo simulation (100,000 paths, 504 steps)...")
    pricing_result = pricer.price_monte_carlo(
        n_paths=100000,
        n_steps=504,  # ~252 trading days × 2 years
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

    # Convergence Analysis
    convergence_df = run_convergence_analysis(market, contract)

    # Generate visualizations
    print("\n" + "-" * 70)
    print("GENERATING VISUALIZATIONS")
    print("-" * 70)

    viz = ProductVisualizer(output_dir='../output')
    viz.create_all_visualizations(
        pricing_result,
        sensitivity_df,
        greeks,
        contract.strike,
        contract.notional,
        (contract.barrier_lower, contract.barrier_upper)
    )

    # Generate Excel output
    print("\n" + "-" * 70)
    print("GENERATING EXCEL OUTPUT")
    print("-" * 70)

    generate_excel_output(
        pricing_result,
        sensitivity_df,
        greeks,
        market,
        contract,
        '../output/GAAIF_Analysis_Data.xlsx'
    )

    # Save convergence data
    convergence_df.to_excel('../output/convergence_analysis.xlsx', index=False)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nOutput files generated:")
    print("  - output/GAAIF_Analysis_Data.xlsx")
    print("  - output/convergence_analysis.xlsx")
    print("  - output/monte_carlo_paths.png")
    print("  - output/payoff_distribution.png")
    print("  - output/sensitivity_analysis.png")
    print("  - output/knockout_analysis.png")
    print("  - output/payoff_diagram.png")
    print("  - output/greeks_summary.png")

    return {
        'pricing_result': pricing_result,
        'greeks': greeks,
        'sensitivity_df': sensitivity_df,
        'convergence_df': convergence_df
    }


if __name__ == "__main__":
    results = main()
