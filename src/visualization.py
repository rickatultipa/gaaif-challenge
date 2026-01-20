"""
Visualization Module for GAAIF Challenge
========================================

Generates charts and graphs for the Product Proposal:
- Monte Carlo path simulations
- Barrier breach analysis
- Sensitivity analysis
- Payoff diagrams
- Risk metrics visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def format_millions(x, p):
    """Format axis tick as millions."""
    return f'€{x/1e6:.0f}M'


def format_currency(x, p):
    """Format axis tick as currency."""
    return f'${x:,.0f}'


class ProductVisualizer:
    """Generates visualizations for the structured product analysis."""

    def __init__(self, output_dir: str = '../output'):
        self.output_dir = output_dir

    def plot_sample_paths(self, gold_paths: np.ndarray, eurusd_paths: np.ndarray,
                          times: np.ndarray, knocked_out: np.ndarray,
                          contract_strike: float, barriers: tuple,
                          n_display: int = 50, save: bool = True) -> plt.Figure:
        """
        Plot sample Monte Carlo paths with barrier overlay.

        Args:
            gold_paths: Array of gold price paths
            eurusd_paths: Array of EUR/USD paths
            times: Time array
            knocked_out: Boolean array of knockout status
            contract_strike: Gold strike price
            barriers: Tuple of (lower_barrier, upper_barrier)
            n_display: Number of paths to display
            save: Whether to save the figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Select sample paths (mix of knocked out and surviving)
        ko_idx = np.where(knocked_out)[0][:n_display//2]
        surv_idx = np.where(~knocked_out)[0][:n_display//2]
        sample_idx = np.concatenate([ko_idx, surv_idx])

        # Plot Gold Paths
        ax1 = axes[0]
        for i in sample_idx:
            color = 'red' if knocked_out[i] else 'blue'
            alpha = 0.3 if knocked_out[i] else 0.5
            ax1.plot(times, gold_paths[i], color=color, alpha=alpha, linewidth=0.5)

        ax1.axhline(y=contract_strike, color='green', linestyle='--', linewidth=2,
                   label=f'Strike K = ${contract_strike:,.0f}')
        ax1.set_xlabel('Time (Years)')
        ax1.set_ylabel('Gold Price (USD/oz)')
        ax1.set_title('Gold Price Paths - Monte Carlo Simulation')
        ax1.legend(loc='upper left')
        ax1.yaxis.set_major_formatter(FuncFormatter(format_currency))

        # Add legend for path types
        ko_patch = mpatches.Patch(color='red', alpha=0.5, label='Knocked Out')
        surv_patch = mpatches.Patch(color='blue', alpha=0.5, label='Surviving')
        ax1.legend(handles=[ko_patch, surv_patch,
                           plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2,
                                     label=f'Strike K = ${contract_strike:,.0f}')],
                  loc='upper left')

        # Plot EUR/USD Paths
        ax2 = axes[1]
        for i in sample_idx:
            color = 'red' if knocked_out[i] else 'blue'
            alpha = 0.3 if knocked_out[i] else 0.5
            ax2.plot(times, eurusd_paths[i], color=color, alpha=alpha, linewidth=0.5)

        ax2.axhline(y=barriers[0], color='darkred', linestyle='--', linewidth=2,
                   label=f'Lower Barrier = {barriers[0]}')
        ax2.axhline(y=barriers[1], color='darkred', linestyle='--', linewidth=2,
                   label=f'Upper Barrier = {barriers[1]}')
        ax2.fill_between(times, barriers[0], barriers[1], color='green', alpha=0.1)
        ax2.set_xlabel('Time (Years)')
        ax2.set_ylabel('EUR/USD Exchange Rate')
        ax2.set_title('EUR/USD Paths with Double Knock-Out Barriers')
        ax2.legend(loc='upper left')

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/monte_carlo_paths.png', dpi=300, bbox_inches='tight')

        return fig

    def plot_payoff_distribution(self, settlement_prices: np.ndarray,
                                 contract_strike: float, notional: float,
                                 knocked_out: np.ndarray, save: bool = True) -> plt.Figure:
        """
        Plot distribution of payoffs.

        Args:
            settlement_prices: Array of gold prices at settlement
            contract_strike: Gold strike price
            notional: Notional principal
            knocked_out: Boolean array of knockout status
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Calculate payoffs
        payoffs = notional * (settlement_prices - contract_strike) / contract_strike

        # Left: Histogram of payoffs
        ax1 = axes[0]
        ax1.hist(payoffs[~knocked_out] / 1e6, bins=50, alpha=0.7, color='blue',
                label='Surviving Paths', density=True)
        ax1.hist(payoffs[knocked_out] / 1e6, bins=50, alpha=0.7, color='red',
                label='Knocked Out Paths', density=True)
        ax1.axvline(x=np.mean(payoffs) / 1e6, color='black', linestyle='--', linewidth=2,
                   label=f'Mean = €{np.mean(payoffs)/1e6:.1f}M')
        ax1.set_xlabel('Payoff (EUR Millions)')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Z Group Payoffs')
        ax1.legend()

        # Right: Settlement price distribution
        ax2 = axes[1]
        ax2.hist(settlement_prices[~knocked_out], bins=50, alpha=0.7, color='blue',
                label='Surviving', density=True)
        ax2.hist(settlement_prices[knocked_out], bins=50, alpha=0.7, color='red',
                label='Knocked Out', density=True)
        ax2.axvline(x=contract_strike, color='green', linestyle='--', linewidth=2,
                   label=f'Strike = ${contract_strike:,.0f}')
        ax2.set_xlabel('Gold Price at Settlement (USD/oz)')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Settlement Prices')
        ax2.legend()
        ax2.xaxis.set_major_formatter(FuncFormatter(format_currency))

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/payoff_distribution.png', dpi=300, bbox_inches='tight')

        return fig

    def plot_sensitivity_analysis(self, sensitivity_df: pd.DataFrame,
                                  save: bool = True) -> plt.Figure:
        """
        Plot sensitivity analysis results.

        Args:
            sensitivity_df: DataFrame with sensitivity analysis results
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        parameters = sensitivity_df['parameter'].unique()

        for i, param in enumerate(parameters):
            if i >= len(axes):
                break

            ax = axes[i]
            data = sensitivity_df[sensitivity_df['parameter'] == param]

            # Price on primary axis
            ax.plot(data['value'], data['price_zgroup'] / 1e6, 'b-o', linewidth=2,
                   markersize=6, label='Price (€M)')
            ax.set_ylabel('Z Group PV (€ Millions)', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

            # Knockout rate on secondary axis
            ax2 = ax.twinx()
            ax2.plot(data['value'], data['knockout_rate'] * 100, 'r--s', linewidth=2,
                    markersize=6, label='KO Rate (%)')
            ax2.set_ylabel('Knockout Rate (%)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            # Format x-axis label
            param_labels = {
                'gold_spot': 'Gold Spot ($/oz)',
                'eurusd_spot': 'EUR/USD Spot',
                'sigma_gold': 'Gold Volatility',
                'sigma_eurusd': 'EUR/USD Volatility',
                'correlation': 'Correlation (ρ)'
            }
            ax.set_xlabel(param_labels.get(param, param))
            ax.set_title(f'Sensitivity to {param_labels.get(param, param)}')

            # Add grid
            ax.grid(True, alpha=0.3)

        # Hide unused subplot
        if len(parameters) < len(axes):
            axes[-1].set_visible(False)

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/sensitivity_analysis.png', dpi=300, bbox_inches='tight')

        return fig

    def plot_knockout_analysis(self, eurusd_paths: np.ndarray, times: np.ndarray,
                               knocked_out: np.ndarray, knockout_idx: np.ndarray,
                               barriers: tuple, save: bool = True) -> plt.Figure:
        """
        Detailed analysis of barrier breaches.

        Args:
            eurusd_paths: Array of EUR/USD paths
            times: Time array
            knocked_out: Boolean array of knockout status
            knockout_idx: Array of knockout time indices
            barriers: Tuple of (lower_barrier, upper_barrier)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Get knockout data
        ko_mask = knocked_out
        ko_times = times[knockout_idx[ko_mask]]
        ko_prices = np.array([eurusd_paths[i, knockout_idx[i]] for i in np.where(ko_mask)[0]])

        # Top Left: Knockout timing distribution
        ax1 = axes[0, 0]
        ax1.hist(ko_times, bins=30, color='red', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Time of Knockout (Years)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Knockout Times')
        ax1.axvline(x=np.mean(ko_times), color='black', linestyle='--',
                   label=f'Mean = {np.mean(ko_times):.2f}y')
        ax1.legend()

        # Top Right: Knockout price distribution
        ax2 = axes[0, 1]
        lower_breaches = ko_prices < barriers[0]
        upper_breaches = ko_prices > barriers[1]

        ax2.hist(ko_prices[lower_breaches], bins=20, color='blue', alpha=0.7,
                label=f'Lower Barrier ({np.sum(lower_breaches)} paths)')
        ax2.hist(ko_prices[upper_breaches], bins=20, color='orange', alpha=0.7,
                label=f'Upper Barrier ({np.sum(upper_breaches)} paths)')
        ax2.axvline(x=barriers[0], color='darkred', linestyle='--', linewidth=2)
        ax2.axvline(x=barriers[1], color='darkred', linestyle='--', linewidth=2)
        ax2.set_xlabel('EUR/USD at Knockout')
        ax2.set_ylabel('Frequency')
        ax2.set_title('EUR/USD Level at Barrier Breach')
        ax2.legend()

        # Bottom Left: Cumulative knockout probability over time
        ax3 = axes[1, 0]
        sorted_times = np.sort(ko_times)
        cumulative = np.arange(1, len(sorted_times) + 1) / len(knocked_out) * 100
        ax3.plot(sorted_times, cumulative, 'r-', linewidth=2)
        ax3.fill_between(sorted_times, 0, cumulative, alpha=0.3, color='red')
        ax3.set_xlabel('Time (Years)')
        ax3.set_ylabel('Cumulative Knockout Probability (%)')
        ax3.set_title('Cumulative Knockout Probability Over Time')
        ax3.set_xlim([0, times[-1]])
        ax3.set_ylim([0, max(cumulative) * 1.1])

        # Bottom Right: Survival probability
        ax4 = axes[1, 1]
        survival = 100 - cumulative
        ax4.plot(sorted_times, survival, 'g-', linewidth=2)
        ax4.fill_between(sorted_times, survival, 100, alpha=0.3, color='green')
        ax4.set_xlabel('Time (Years)')
        ax4.set_ylabel('Survival Probability (%)')
        ax4.set_title('Contract Survival Probability')
        ax4.set_xlim([0, times[-1]])
        ax4.set_ylim([0, 105])

        # Add final survival rate annotation
        final_survival = 100 * (1 - np.mean(knocked_out))
        ax4.annotate(f'Final: {final_survival:.1f}%',
                    xy=(times[-1], final_survival),
                    xytext=(times[-1] * 0.7, final_survival + 10),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=12)

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/knockout_analysis.png', dpi=300, bbox_inches='tight')

        return fig

    def plot_payoff_diagram(self, contract_strike: float, notional: float,
                            gold_range: tuple = (3500, 6000), save: bool = True) -> plt.Figure:
        """
        Plot theoretical payoff diagram at maturity.

        Args:
            contract_strike: Strike price
            notional: Notional amount
            gold_range: Range of gold prices to plot
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        gold_prices = np.linspace(gold_range[0], gold_range[1], 200)
        zgroup_payoff = notional * (gold_prices - contract_strike) / contract_strike
        abank_payoff = -zgroup_payoff

        ax.plot(gold_prices, zgroup_payoff / 1e6, 'b-', linewidth=2.5, label='Z Group Payoff')
        ax.plot(gold_prices, abank_payoff / 1e6, 'r-', linewidth=2.5, label='A Bank Payoff')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=contract_strike, color='green', linestyle='--', linewidth=2,
                  label=f'Strike K = ${contract_strike:,.0f}')

        ax.fill_between(gold_prices, 0, zgroup_payoff / 1e6,
                       where=(zgroup_payoff > 0), alpha=0.3, color='blue')
        ax.fill_between(gold_prices, 0, zgroup_payoff / 1e6,
                       where=(zgroup_payoff < 0), alpha=0.3, color='red')

        ax.set_xlabel('Gold Spot Price at Settlement (USD/oz)', fontsize=12)
        ax.set_ylabel('Payoff (EUR Millions)', fontsize=12)
        ax.set_title('Payoff Diagram: Structured Gold Forward\n'
                    f'Notional = €{notional/1e6:.0f}M, Strike = ${contract_strike:,.0f}/oz',
                    fontsize=14)
        ax.legend(loc='upper left', fontsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(format_currency))

        # Add breakeven annotation
        ax.annotate('Breakeven\n(Gold = Strike)',
                   xy=(contract_strike, 0),
                   xytext=(contract_strike + 300, 30),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   fontsize=11)

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/payoff_diagram.png', dpi=300, bbox_inches='tight')

        return fig

    def plot_greeks_summary(self, greeks: dict, save: bool = True) -> plt.Figure:
        """
        Visualize option Greeks.

        Args:
            greeks: Dictionary of Greek values
        """
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
        ax.set_ylabel('Sensitivity (€ Millions per unit)', fontsize=12)
        ax.set_title('Risk Sensitivities (Greeks) Summary', fontsize=14)

        # Add value labels on bars
        for bar, val in zip(bars, greek_values):
            height = bar.get_height()
            ax.annotate(f'€{val:.2f}M',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=10)

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/greeks_summary.png', dpi=300, bbox_inches='tight')

        return fig

    def create_all_visualizations(self, pricing_result: dict, sensitivity_df: pd.DataFrame,
                                  greeks: dict, contract_strike: float, notional: float,
                                  barriers: tuple):
        """
        Generate all visualizations for the proposal.

        Args:
            pricing_result: Dictionary from Monte Carlo pricing
            sensitivity_df: Sensitivity analysis DataFrame
            greeks: Greeks dictionary
            contract_strike: Strike price
            notional: Notional principal
            barriers: (lower, upper) barriers
        """
        print("Generating visualizations...")

        times = np.linspace(0, 2.0, pricing_result['gold_paths'].shape[1])

        # 1. Monte Carlo paths
        print("  - Monte Carlo paths...")
        self.plot_sample_paths(
            pricing_result['gold_paths'],
            pricing_result['eurusd_paths'],
            times,
            pricing_result['knocked_out'],
            contract_strike,
            barriers
        )

        # 2. Payoff distribution
        print("  - Payoff distribution...")
        self.plot_payoff_distribution(
            pricing_result['settlement_prices'],
            contract_strike,
            notional,
            pricing_result['knocked_out']
        )

        # 3. Sensitivity analysis
        print("  - Sensitivity analysis...")
        self.plot_sensitivity_analysis(sensitivity_df)

        # 4. Knockout analysis
        print("  - Knockout analysis...")
        knocked_out = pricing_result['knocked_out']
        knockout_idx = np.zeros(len(knocked_out), dtype=int)
        for i in range(len(knocked_out)):
            if knocked_out[i]:
                breached = (pricing_result['eurusd_paths'][i] < barriers[0]) | \
                          (pricing_result['eurusd_paths'][i] > barriers[1])
                knockout_idx[i] = np.argmax(breached)

        self.plot_knockout_analysis(
            pricing_result['eurusd_paths'],
            times,
            knocked_out,
            knockout_idx,
            barriers
        )

        # 5. Payoff diagram
        print("  - Payoff diagram...")
        self.plot_payoff_diagram(contract_strike, notional)

        # 6. Greeks summary
        print("  - Greeks summary...")
        self.plot_greeks_summary(greeks)

        print("All visualizations generated successfully!")


if __name__ == "__main__":
    # Test visualization with dummy data
    import sys
    sys.path.insert(0, '.')
    from pricing_model import MarketData, ContractTerms, StructuredForwardPricer, run_sensitivity_analysis

    print("Testing visualization module...")

    market = MarketData()
    contract = ContractTerms()
    pricer = StructuredForwardPricer(market, contract)

    # Run pricing
    result = pricer.price_monte_carlo(n_paths=10000, n_steps=252)

    # Run sensitivity
    sens_df = run_sensitivity_analysis(market, contract, n_paths=5000)

    # Compute Greeks
    greeks = pricer.compute_greeks(result, n_paths=5000)

    # Create visualizer
    viz = ProductVisualizer(output_dir='../output')

    # Generate all plots
    viz.create_all_visualizations(
        result, sens_df, greeks,
        contract.strike, contract.notional,
        (contract.barrier_lower, contract.barrier_upper)
    )
