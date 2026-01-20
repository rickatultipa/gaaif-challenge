"""
Enhanced Analysis Runner for GAAIF Challenge
=============================================

Runs comprehensive analysis including:
1. Base case Monte Carlo pricing
2. Scenario analysis (strikes, barriers, volatility)
3. Model comparison (GBM vs Heston vs Merton)
4. Enhanced visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pricing_model import (
    MarketData, ContractTerms, StructuredForwardPricer,
    run_sensitivity_analysis
)
from advanced_models import (
    run_scenario_analysis, run_model_comparison,
    HestonParams, JumpParams
)
from visualization import ProductVisualizer


def create_scenario_visualizations(scenario_df: pd.DataFrame, output_dir: str):
    """Create visualizations for scenario analysis."""

    # Strike analysis
    strike_data = scenario_df[scenario_df['scenario_type'] == 'strike']
    if len(strike_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        ax1.bar(strike_data['value'].astype(str), strike_data['price_zgroup'] / 1e6,
               color='steelblue', alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Strike Price (USD/oz)')
        ax1.set_ylabel('Z Group PV (EUR Millions)')
        ax1.set_title('Impact of Strike Price on Valuation')

        ax2 = axes[1]
        ax2.bar(strike_data['value'].astype(str), strike_data['knockout_rate'] * 100,
               color='coral', alpha=0.8)
        ax2.set_xlabel('Strike Price (USD/oz)')
        ax2.set_ylabel('Knockout Rate (%)')
        ax2.set_title('Strike Price vs Knockout Probability')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/scenario_strike.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Barrier analysis
    barrier_data = scenario_df[scenario_df['scenario_type'] == 'barrier']
    if len(barrier_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        ax1.bar(range(len(barrier_data)), barrier_data['price_zgroup'] / 1e6,
               color='steelblue', alpha=0.8)
        ax1.set_xticks(range(len(barrier_data)))
        ax1.set_xticklabels(barrier_data['parameter'], rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Z Group PV (EUR Millions)')
        ax1.set_title('Impact of Barrier Width on Valuation')

        ax2 = axes[1]
        ax2.bar(range(len(barrier_data)), barrier_data['knockout_rate'] * 100,
               color='coral', alpha=0.8)
        ax2.set_xticks(range(len(barrier_data)))
        ax2.set_xticklabels(barrier_data['parameter'], rotation=45, ha='right')
        ax2.set_ylabel('Knockout Rate (%)')
        ax2.set_title('Barrier Width vs Knockout Probability')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/scenario_barrier.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Volatility analysis
    vol_data = scenario_df[scenario_df['scenario_type'] == 'volatility']
    if len(vol_data) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(vol_data['value'] * 100, vol_data['price_zgroup'] / 1e6,
               'b-o', linewidth=2, markersize=8, label='Z Group PV')
        ax.set_xlabel('Gold Volatility (%)')
        ax.set_ylabel('Z Group PV (EUR Millions)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        ax2 = ax.twinx()
        ax2.plot(vol_data['value'] * 100, vol_data['knockout_rate'] * 100,
                'r--s', linewidth=2, markersize=8, label='KO Rate')
        ax2.set_ylabel('Knockout Rate (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_title('Impact of Volatility Regime')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/scenario_volatility.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_model_comparison_chart(model_results: dict, output_dir: str):
    """Create chart comparing different model specifications."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(model_results.keys())
    prices = [model_results[m]['price_zgroup'] / 1e6 for m in models]
    ko_rates = [model_results[m]['knockout_rate'] * 100 for m in models]

    # Price comparison
    ax1 = axes[0]
    colors = ['steelblue', 'seagreen', 'coral']
    bars = ax1.bar(models, prices, color=colors[:len(models)], alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Z Group PV (EUR Millions)')
    ax1.set_title('Model Comparison: Valuation')

    for bar, price in zip(bars, prices):
        ax1.annotate(f'EUR {price:.1f}M',
                    xy=(bar.get_x() + bar.get_width()/2, price),
                    xytext=(0, 3 if price >= 0 else -15),
                    textcoords='offset points',
                    ha='center', va='bottom' if price >= 0 else 'top',
                    fontsize=10)

    # Knockout rate comparison
    ax2 = axes[1]
    ax2.bar(models, ko_rates, color=colors[:len(models)], alpha=0.8)
    ax2.set_ylabel('Knockout Rate (%)')
    ax2.set_title('Model Comparison: Knockout Probability')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_enhanced_excel(pricing_result, scenario_df, model_results,
                           greeks, market, contract, output_path):
    """Generate comprehensive Excel with all analysis."""

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Executive Summary
        summary = pd.DataFrame({
            'Metric': [
                'Product Type',
                'Notional (EUR)',
                'Strike (USD/oz)',
                'Tenor (Years)',
                'Barrier Range',
                '',
                'Z Group PV (EUR)',
                'A Bank PV (EUR)',
                'Knockout Probability',
                'Expected Duration',
                '',
                'Model Used',
                'Simulation Paths',
                'Confidence Level'
            ],
            'Value': [
                'Structured Gold Forward with Double KO',
                f'{contract.notional:,.0f}',
                f'${contract.strike:,.0f}',
                f'{contract.tenor}',
                f'[{contract.barrier_lower}, {contract.barrier_upper}]',
                '',
                f'{pricing_result["price_zgroup"]:,.0f}',
                f'{pricing_result["price_abank"]:,.0f}',
                f'{pricing_result["knockout_rate"]*100:.1f}%',
                f'{pricing_result["avg_knockout_time"]:.2f} years' if not np.isnan(pricing_result["avg_knockout_time"]) else 'N/A',
                '',
                'Correlated GBM + Variance Reduction',
                f'{pricing_result["n_paths"]:,}',
                '95%'
            ]
        })
        summary.to_excel(writer, sheet_name='Executive Summary', index=False)

        # Sheet 2: Market Data
        market_df = pd.DataFrame({
            'Parameter': [
                'Gold Spot (USD/oz)', 'EUR/USD Spot',
                'EUR Risk-Free Rate', 'USD Risk-Free Rate',
                'Gold Volatility', 'EUR/USD Volatility',
                'Correlation', 'Gold Convenience Yield'
            ],
            'Value': [
                market.gold_spot, market.eurusd_spot,
                f'{market.r_eur*100:.2f}%', f'{market.r_usd*100:.2f}%',
                f'{market.sigma_gold*100:.1f}%', f'{market.sigma_eurusd*100:.1f}%',
                market.rho, f'{market.gold_yield*100:.2f}%'
            ]
        })
        market_df.to_excel(writer, sheet_name='Market Data', index=False)

        # Sheet 3: Pricing Results
        pricing_df = pd.DataFrame({
            'Metric': [
                'Z Group Present Value (EUR)',
                'A Bank Present Value (EUR)',
                'Standard Error',
                '95% CI Lower',
                '95% CI Upper',
                'Knockout Rate',
                'Avg Knockout Time',
                'Lower Barrier Breach Rate',
                'Upper Barrier Breach Rate'
            ],
            'Value': [
                pricing_result['price_zgroup'],
                pricing_result['price_abank'],
                pricing_result['std_error'],
                pricing_result['ci_95_lower'],
                pricing_result['ci_95_upper'],
                pricing_result['knockout_rate'],
                pricing_result['avg_knockout_time'],
                pricing_result['lower_breach_rate'],
                pricing_result['upper_breach_rate']
            ]
        })
        pricing_df.to_excel(writer, sheet_name='Pricing Results', index=False)

        # Sheet 4: Greeks
        greeks_df = pd.DataFrame({
            'Greek': list(greeks.keys()),
            'Value (EUR)': list(greeks.values())
        })
        greeks_df.to_excel(writer, sheet_name='Greeks', index=False)

        # Sheet 5: Scenario Analysis
        scenario_df.to_excel(writer, sheet_name='Scenario Analysis', index=False)

        # Sheet 6: Model Comparison
        model_df = pd.DataFrame([
            {'Model': k, 'Price_ZGroup_EUR': v['price_zgroup'],
             'Knockout_Rate': v['knockout_rate'], 'Std_Error': v['std_error']}
            for k, v in model_results.items()
        ])
        model_df.to_excel(writer, sheet_name='Model Comparison', index=False)

    print(f"Enhanced Excel saved to: {output_path}")


def main():
    """Run enhanced analysis."""

    print("=" * 70)
    print("GAAIF CHALLENGE - ENHANCED ANALYSIS")
    print("=" * 70)
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    output_dir = '../output'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize
    market = MarketData()
    contract = ContractTerms()

    print("\n" + "-" * 70)
    print("1. BASE CASE PRICING")
    print("-" * 70)

    pricer = StructuredForwardPricer(market, contract)
    pricing_result = pricer.price_monte_carlo(n_paths=100000, n_steps=504, seed=42)

    print(f"  Z Group PV:     EUR {pricing_result['price_zgroup']:,.0f}")
    print(f"  Knockout Rate:  {pricing_result['knockout_rate']*100:.1f}%")

    # Greeks
    print("\n  Computing Greeks...")
    greeks = pricer.compute_greeks(pricing_result, n_paths=50000)

    print("\n" + "-" * 70)
    print("2. SCENARIO ANALYSIS")
    print("-" * 70)

    scenario_df = run_scenario_analysis(market, contract, n_paths=30000)
    print(f"  Completed {len(scenario_df)} scenarios")

    print("\n" + "-" * 70)
    print("3. MODEL COMPARISON")
    print("-" * 70)

    model_results = run_model_comparison(market, contract, n_paths=50000)
    for model, res in model_results.items():
        print(f"  {model}: EUR {res['price_zgroup']:,.0f} (KO: {res['knockout_rate']*100:.1f}%)")

    print("\n" + "-" * 70)
    print("4. GENERATING VISUALIZATIONS")
    print("-" * 70)

    # Base visualizations
    viz = ProductVisualizer(output_dir=output_dir)
    sensitivity_df = run_sensitivity_analysis(market, contract, n_paths=30000)
    viz.create_all_visualizations(
        pricing_result, sensitivity_df, greeks,
        contract.strike, contract.notional,
        (contract.barrier_lower, contract.barrier_upper)
    )

    # Scenario visualizations
    print("  Creating scenario charts...")
    create_scenario_visualizations(scenario_df, output_dir)

    # Model comparison chart
    print("  Creating model comparison chart...")
    create_model_comparison_chart(model_results, output_dir)

    print("\n" + "-" * 70)
    print("5. GENERATING EXCEL OUTPUT")
    print("-" * 70)

    generate_enhanced_excel(
        pricing_result, scenario_df, model_results,
        greeks, market, contract,
        f'{output_dir}/GAAIF_Enhanced_Analysis.xlsx'
    )

    print("\n" + "=" * 70)
    print("ENHANCED ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nAdditional outputs generated:")
    print("  - scenario_strike.png")
    print("  - scenario_barrier.png")
    print("  - scenario_volatility.png")
    print("  - model_comparison.png")
    print("  - GAAIF_Enhanced_Analysis.xlsx")

    return {
        'pricing_result': pricing_result,
        'scenario_df': scenario_df,
        'model_results': model_results,
        'greeks': greeks
    }


if __name__ == "__main__":
    results = main()
