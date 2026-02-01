#!/usr/bin/env python3
"""
Script to update charts with enriched data matching the comprehensive Excel file.
Regenerates PDF from the updated HTML report.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pricing_model import MarketData, ContractTerms, StructuredForwardPricer, CorrelatedGBMSimulator

# Create output directory
os.makedirs('output/pdf_charts', exist_ok=True)

# Initialize market data and contract
market = MarketData()
contract = ContractTerms()
pricer = StructuredForwardPricer(market, contract)

print("=" * 60)
print("UPDATING CHARTS WITH ENRICHED DATA")
print("=" * 60)

# Chart styling
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10

colors = {
    'primary': '#1a365d',
    'secondary': '#2c5282',
    'gold': '#d69e2e',
    'positive': '#38a169',
    'negative': '#e53e3e',
    'neutral': '#718096'
}

# =============================================================================
# 1. GOLD SENSITIVITY - 20 data points (matching Excel)
# =============================================================================
print("\n1. Generating Gold Sensitivity Chart (20 points)...")
gold_spots = np.linspace(4000, 5800, 20)
gold_pvs = []
gold_ko_rates = []

for gs in gold_spots:
    m = MarketData(gold_spot=gs, eurusd_spot=market.eurusd_spot,
                  r_eur=market.r_eur, r_usd=market.r_usd,
                  sigma_gold=market.sigma_gold, sigma_eurusd=market.sigma_eurusd,
                  rho=market.rho, gold_yield=market.gold_yield)
    p = StructuredForwardPricer(m, contract)
    res = p.price_monte_carlo(n_paths=30000, seed=42)
    gold_pvs.append(res['price_zgroup'] / 1e6)
    gold_ko_rates.append(res['knockout_rate'] * 100)
    print(f"   Gold ${gs:.0f}: PV = EUR {res['price_zgroup']/1e6:+.1f}M")

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.fill_between(gold_spots, gold_pvs, alpha=0.3, color=colors['positive'])
ax1.plot(gold_spots, gold_pvs, color=colors['positive'], linewidth=2.5, label='Z Group PV')
ax1.axhline(y=0, color=colors['neutral'], linestyle='--', alpha=0.5)
ax1.axvline(x=contract.strike, color=colors['gold'], linestyle='--', linewidth=2, label=f'Strike ${contract.strike:,.0f}')
ax1.axvline(x=market.gold_spot, color=colors['secondary'], linestyle='-', linewidth=2, label=f'Current ${market.gold_spot:,.0f}')

ax2.plot(gold_spots, gold_ko_rates, color=colors['negative'], linewidth=2, linestyle=':', label='Knockout Rate')

ax1.set_xlabel('Gold Spot Price ($/oz)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Z Group Present Value (EUR Million)', fontsize=12, color=colors['positive'])
ax2.set_ylabel('Knockout Rate (%)', fontsize=12, color=colors['negative'])
ax1.set_title('Gold Price Sensitivity Analysis (20 Data Points)', fontsize=14, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
plt.savefig('output/pdf_charts/gold_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 2. EUR/USD SENSITIVITY - 20 data points (matching Excel)
# =============================================================================
print("\n2. Generating EUR/USD Sensitivity Chart (20 points)...")
fx_spots = np.linspace(1.06, 1.24, 20)
fx_pvs = []
fx_ko_rates = []
fx_upper_rates = []

for fx in fx_spots:
    m = MarketData(gold_spot=market.gold_spot, eurusd_spot=fx,
                  r_eur=market.r_eur, r_usd=market.r_usd,
                  sigma_gold=market.sigma_gold, sigma_eurusd=market.sigma_eurusd,
                  rho=market.rho, gold_yield=market.gold_yield)
    p = StructuredForwardPricer(m, contract)
    res = p.price_monte_carlo(n_paths=30000, seed=42)
    fx_pvs.append(res['price_zgroup'] / 1e6)
    fx_ko_rates.append(res['knockout_rate'] * 100)
    fx_upper_rates.append(res['upper_breach_rate'] * 100)
    print(f"   EUR/USD {fx:.3f}: PV = EUR {res['price_zgroup']/1e6:+.1f}M, KO={res['knockout_rate']*100:.1f}%")

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.fill_between(fx_spots, fx_pvs, alpha=0.3, color=colors['primary'])
ax1.plot(fx_spots, fx_pvs, color=colors['primary'], linewidth=2.5, label='Z Group PV')
ax1.axhline(y=0, color=colors['neutral'], linestyle='--', alpha=0.5)
ax1.axvline(x=contract.barrier_lower, color=colors['negative'], linestyle='--', linewidth=2, label='Lower Barrier (1.05)')
ax1.axvline(x=contract.barrier_upper, color=colors['negative'], linestyle='--', linewidth=2, label='Upper Barrier (1.25)')
ax1.axvline(x=market.eurusd_spot, color=colors['gold'], linestyle='-', linewidth=2, label=f'Current {market.eurusd_spot:.2f}')

# Shade barrier proximity zone
ax1.axvspan(1.20, 1.25, alpha=0.1, color=colors['negative'], label='Near Upper Barrier')

ax2.plot(fx_spots, fx_ko_rates, color=colors['negative'], linewidth=2, linestyle=':', label='Total KO Rate')
ax2.plot(fx_spots, fx_upper_rates, color=colors['gold'], linewidth=2, linestyle='-.', label='Upper Breach Rate')

ax1.set_xlabel('EUR/USD Spot Rate', fontsize=12, fontweight='bold')
ax1.set_ylabel('Z Group Present Value (EUR Million)', fontsize=12, color=colors['primary'])
ax2.set_ylabel('Rate (%)', fontsize=12, color=colors['negative'])
ax1.set_title('EUR/USD Sensitivity Analysis (20 Data Points)', fontsize=14, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('output/pdf_charts/fx_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 3. VOLATILITY SURFACE (2D Grid) - matching Excel
# =============================================================================
print("\n3. Generating Volatility Surface (2D Grid)...")
gold_vols = np.linspace(0.15, 0.40, 6)
fx_vols = np.linspace(0.06, 0.14, 5)

vol_grid = np.zeros((len(fx_vols), len(gold_vols)))

for i, fv in enumerate(fx_vols):
    for j, gv in enumerate(gold_vols):
        m = MarketData(gold_spot=market.gold_spot, eurusd_spot=market.eurusd_spot,
                      r_eur=market.r_eur, r_usd=market.r_usd,
                      sigma_gold=gv, sigma_eurusd=fv,
                      rho=market.rho, gold_yield=market.gold_yield)
        p = StructuredForwardPricer(m, contract)
        res = p.price_monte_carlo(n_paths=20000, seed=42)
        vol_grid[i, j] = res['price_zgroup'] / 1e6
    print(f"   FX Vol {fv*100:.0f}% row complete")

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(vol_grid, cmap='RdYlGn', aspect='auto',
               extent=[gold_vols[0]*100, gold_vols[-1]*100, fx_vols[0]*100, fx_vols[-1]*100],
               origin='lower')
ax.set_xlabel('Gold Volatility (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('EUR/USD Volatility (%)', fontsize=12, fontweight='bold')
ax.set_title('Z Group PV Volatility Surface (EUR Million)', fontsize=14, fontweight='bold')
plt.colorbar(im, label='EUR Million')

# Mark current volatilities
ax.plot(market.sigma_gold*100, market.sigma_eurusd*100, 'ko', markersize=12, markerfacecolor='white')
ax.annotate('Current', (market.sigma_gold*100, market.sigma_eurusd*100),
            textcoords='offset points', xytext=(10, 10), fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('output/pdf_charts/volatility_surface.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 4. CORRELATION SENSITIVITY - 15 points (matching Excel)
# =============================================================================
print("\n4. Generating Correlation Sensitivity Chart (15 points)...")
correlations = np.linspace(-0.6, 0.4, 15)
corr_pvs = []
corr_ko_rates = []

for rho in correlations:
    m = MarketData(gold_spot=market.gold_spot, eurusd_spot=market.eurusd_spot,
                  r_eur=market.r_eur, r_usd=market.r_usd,
                  sigma_gold=market.sigma_gold, sigma_eurusd=market.sigma_eurusd,
                  rho=rho, gold_yield=market.gold_yield)
    p = StructuredForwardPricer(m, contract)
    res = p.price_monte_carlo(n_paths=30000, seed=42)
    corr_pvs.append(res['price_zgroup'] / 1e6)
    corr_ko_rates.append(res['knockout_rate'] * 100)
    print(f"   Correlation {rho:.2f}: PV = EUR {res['price_zgroup']/1e6:+.1f}M")

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.fill_between(correlations, corr_pvs, alpha=0.3, color=colors['secondary'])
ax1.plot(correlations, corr_pvs, color=colors['secondary'], linewidth=2.5, marker='o', markersize=4, label='Z Group PV')
ax1.axvline(x=market.rho, color=colors['gold'], linestyle='-', linewidth=2, label=f'Current ρ={market.rho:.2f}')
ax1.axhline(y=0, color=colors['neutral'], linestyle='--', alpha=0.5)

ax2.plot(correlations, corr_ko_rates, color=colors['negative'], linewidth=2, linestyle=':', marker='s', markersize=4, label='Knockout Rate')

ax1.set_xlabel('Gold/EUR-USD Correlation (ρ)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Z Group Present Value (EUR Million)', fontsize=12, color=colors['secondary'])
ax2.set_ylabel('Knockout Rate (%)', fontsize=12, color=colors['negative'])
ax1.set_title('Correlation Sensitivity Analysis (15 Data Points)', fontsize=14, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
plt.savefig('output/pdf_charts/correlation_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 5. CONVERGENCE ANALYSIS
# =============================================================================
print("\n5. Generating Convergence Analysis Chart...")
path_counts = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
convergence_pvs = []
convergence_ses = []

for n in path_counts:
    res = pricer.price_monte_carlo(n_paths=n, seed=42)
    convergence_pvs.append(res['price_zgroup'] / 1e6)
    convergence_ses.append(res['std_error'] / 1e6)
    print(f"   {n:,} paths: PV = EUR {res['price_zgroup']/1e6:.2f}M ± {res['std_error']/1e6:.3f}M")

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.plot(path_counts, convergence_pvs, color=colors['primary'], linewidth=2.5, marker='o', label='Z Group PV')
ax1.axhline(y=convergence_pvs[-1], color=colors['positive'], linestyle='--', alpha=0.7, label=f'Final: EUR {convergence_pvs[-1]:.1f}M')
ax1.fill_between(path_counts,
                 [p - 1.96*s for p, s in zip(convergence_pvs, convergence_ses)],
                 [p + 1.96*s for p, s in zip(convergence_pvs, convergence_ses)],
                 alpha=0.2, color=colors['primary'], label='95% CI')

ax2.plot(path_counts, convergence_ses, color=colors['gold'], linewidth=2, linestyle='--', marker='s', label='Std Error')

ax1.set_xlabel('Number of Simulation Paths', fontsize=12, fontweight='bold')
ax1.set_ylabel('Z Group Present Value (EUR Million)', fontsize=12, color=colors['primary'])
ax2.set_ylabel('Standard Error (EUR Million)', fontsize=12, color=colors['gold'])
ax1.set_title('Monte Carlo Convergence Analysis', fontsize=14, fontweight='bold')
ax1.set_xscale('log')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.tight_layout()
plt.savefig('output/pdf_charts/convergence.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 6. SCENARIO ANALYSIS (18 scenarios matching Excel)
# =============================================================================
print("\n6. Generating Scenario Analysis Chart (18 scenarios)...")
scenarios = [
    ("Base Case", market.gold_spot, market.eurusd_spot, market.sigma_gold),
    ("Gold +10%", market.gold_spot * 1.10, market.eurusd_spot, market.sigma_gold),
    ("Gold -10%", market.gold_spot * 0.90, market.eurusd_spot, market.sigma_gold),
    ("Gold +20%", market.gold_spot * 1.20, market.eurusd_spot, market.sigma_gold),
    ("Gold -20%", market.gold_spot * 0.80, market.eurusd_spot, market.sigma_gold),
    ("EUR Strong +5%", market.gold_spot, market.eurusd_spot * 1.05, market.sigma_gold),
    ("EUR Weak -5%", market.gold_spot, market.eurusd_spot * 0.95, market.sigma_gold),
    ("EUR at Upper", market.gold_spot, 1.24, market.sigma_gold),
    ("EUR at Lower", market.gold_spot, 1.06, market.sigma_gold),
    ("High Vol", market.gold_spot, market.eurusd_spot, 0.35),
    ("Low Vol", market.gold_spot, market.eurusd_spot, 0.20),
    ("Gold Crash -30%", market.gold_spot * 0.70, market.eurusd_spot, market.sigma_gold),
    ("Gold Rally +30%", market.gold_spot * 1.30, market.eurusd_spot, market.sigma_gold),
    ("Combined Stress+", market.gold_spot * 1.15, 1.22, 0.32),
    ("Combined Stress-", market.gold_spot * 0.85, 1.08, 0.25),
    ("All Tail Risk", market.gold_spot * 0.75, 1.06, 0.40),
    ("Best Case", market.gold_spot * 1.25, 1.23, 0.22),
    ("Worst Case", market.gold_spot * 0.70, 1.07, 0.35),
]

scenario_names = []
scenario_pvs = []
scenario_colors = []

for name, gs, fx, vol in scenarios:
    m = MarketData(gold_spot=gs, eurusd_spot=fx,
                  r_eur=market.r_eur, r_usd=market.r_usd,
                  sigma_gold=vol, sigma_eurusd=market.sigma_eurusd,
                  rho=market.rho, gold_yield=market.gold_yield)
    p = StructuredForwardPricer(m, contract)
    res = p.price_monte_carlo(n_paths=25000, seed=42)
    pv = res['price_zgroup'] / 1e6
    scenario_names.append(name)
    scenario_pvs.append(pv)
    scenario_colors.append(colors['positive'] if pv >= 0 else colors['negative'])
    print(f"   {name}: PV = EUR {pv:+.1f}M")

fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(scenario_names))
bars = ax.barh(y_pos, scenario_pvs, color=scenario_colors, alpha=0.8, edgecolor='white', linewidth=0.5)

ax.axvline(x=0, color='black', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(scenario_names, fontsize=9)
ax.set_xlabel('Z Group Present Value (EUR Million)', fontsize=12, fontweight='bold')
ax.set_title('Scenario Analysis - Z Group PV Impact (18 Scenarios)', fontsize=14, fontweight='bold')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, scenario_pvs)):
    ax.text(val + (2 if val >= 0 else -2), i, f'{val:+.0f}M',
            va='center', ha='left' if val >= 0 else 'right', fontsize=8)

plt.tight_layout()
plt.savefig('output/pdf_charts/scenario_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 7. SAMPLE PATHS VISUALIZATION
# =============================================================================
print("\n7. Generating Sample Paths Visualizations...")
simulator = CorrelatedGBMSimulator(market, contract, seed=42)
sim_result = simulator.simulate_paths(n_paths=1000, n_steps=252)
gold_paths = sim_result['gold_paths']
eurusd_paths = sim_result['eurusd_paths']
times = sim_result['times']

# Gold paths
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(min(50, gold_paths.shape[0])):
    ax.plot(times, gold_paths[i, :], alpha=0.3, linewidth=0.5, color=colors['secondary'])
ax.axhline(y=contract.strike, color=colors['gold'], linestyle='--', linewidth=2, label=f'Strike ${contract.strike:,.0f}')
ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Gold Price ($/oz)', fontsize=12)
ax.set_title('Monte Carlo Simulation: Gold Price Paths (50 sample paths)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('output/pdf_charts/gold_paths.png', dpi=150, bbox_inches='tight')
plt.close()

# EUR/USD paths
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(min(50, eurusd_paths.shape[0])):
    ax.plot(times, eurusd_paths[i, :], alpha=0.3, linewidth=0.5, color=colors['secondary'])
ax.axhline(y=contract.barrier_lower, color=colors['negative'], linestyle='--', linewidth=2, label=f'Lower Barrier {contract.barrier_lower}')
ax.axhline(y=contract.barrier_upper, color=colors['negative'], linestyle='--', linewidth=2, label=f'Upper Barrier {contract.barrier_upper}')
ax.fill_between(times, contract.barrier_lower, contract.barrier_upper, alpha=0.1, color=colors['positive'], label='Valid Range')
ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
ax.set_ylabel('EUR/USD', fontsize=12)
ax.set_title('Monte Carlo Simulation: EUR/USD Paths (50 sample paths)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('output/pdf_charts/eurusd_paths.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 8. UPDATED VALUATION COMPARISON (with enriched data context)
# =============================================================================
print("\n8. Generating Valuation Comparison Chart...")
comparison_data = {
    'Metric': ['Gold Spot', 'EUR/USD', 'Volatility', 'Z Group PV', 'Knockout Rate'],
    'January 2026': ['$2,750', '1.08', '18%', 'EUR -192M', '92.6%'],
    'February 2026': ['$4,900', '1.19', '28%', 'EUR +46M', '94.8%'],
    'Change': ['+78%', '+10%', '+56%', '+EUR 238M', '+2.2pp']
}

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Gold Price\n($/oz)', 'EUR/USD\nSpot', 'Z Group PV\n(EUR M)', 'Knockout\nRate (%)']
old_values = [2750/4900*100, 1.08/1.19*100, -192/46*-1, 92.6]  # Normalized
new_values = [100, 100, 100, 94.8/92.6*100]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, [56, 91, 0, 98], width, label='January 2026', color=colors['neutral'], alpha=0.7)
bars2 = ax.bar(x + width/2, [100, 100, 100, 100], width, label='February 2026', color=colors['positive'], alpha=0.8)

ax.set_ylabel('Normalized Value (Current = 100)', fontsize=12)
ax.set_title('Market Conditions & Valuation Change (Jan vs Feb 2026)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Add annotations
ax.annotate('$2,750 → $4,900\n(+78%)', xy=(0, 110), ha='center', fontsize=9, color=colors['positive'], fontweight='bold')
ax.annotate('1.08 → 1.19\n(+10%)', xy=(1, 110), ha='center', fontsize=9, color=colors['positive'], fontweight='bold')
ax.annotate('-EUR 192M → +EUR 46M\n(+EUR 238M swing!)', xy=(2, 110), ha='center', fontsize=9, color=colors['positive'], fontweight='bold')
ax.annotate('92.6% → 94.8%', xy=(3, 110), ha='center', fontsize=9)

ax.set_ylim(0, 130)
plt.tight_layout()
plt.savefig('output/pdf_charts/valuation_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 9. GREEKS CHART
# =============================================================================
print("\n9. Generating Greeks Chart...")
base_result = pricer.price_monte_carlo(n_paths=50000, seed=42)
greeks = pricer.compute_greeks(base_result=base_result, n_paths=50000, seed=42)

fig, ax = plt.subplots(figsize=(10, 6))

greek_names = ['Delta\n(Gold)', 'Delta\n(FX)', 'Vega\n(Gold)', 'Rho\n(EUR)']
greek_values = [
    greeks['delta_gold'] / 1000,  # Scale to thousands
    greeks['delta_eurusd'] / 1e6,  # Scale to millions
    greeks['vega_gold'] / 1e6,
    greeks['rho_eur'] / 1e6
]
greek_colors = [colors['positive'] if v > 0 else colors['negative'] for v in greek_values]

bars = ax.bar(greek_names, greek_values, color=greek_colors, alpha=0.8, edgecolor='white', linewidth=2)
ax.axhline(y=0, color='black', linewidth=1)

ax.set_ylabel('Sensitivity Value', fontsize=12, fontweight='bold')
ax.set_title('Risk Sensitivities (Greeks)', fontsize=14, fontweight='bold')

# Add value labels
for bar, val, name in zip(bars, greek_values, greek_names):
    if 'Gold' in name and 'Delta' in name:
        label = f'EUR {val:.0f}K\nper $1'
    elif 'FX' in name:
        label = f'EUR {val:.1f}M\nper 0.01'
    else:
        label = f'EUR {val:.1f}M'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (5 if val > 0 else -10),
            label, ha='center', va='bottom' if val > 0 else 'top', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('output/pdf_charts/greeks_chart.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 10. PAYOFF DISTRIBUTION
# =============================================================================
print("\n10. Generating Payoff Distribution Chart...")
result = pricer.price_monte_carlo(n_paths=100000, seed=42)
simulator = CorrelatedGBMSimulator(market, contract, seed=42)
sim_result = simulator.simulate_paths(n_paths=100000, n_steps=504)

knocked_out, knockout_idx = simulator.check_barrier_breach(sim_result['eurusd_paths'])
settlement_prices = np.where(
    knocked_out,
    [sim_result['gold_paths'][i, knockout_idx[i]] for i in range(len(knocked_out))],
    sim_result['gold_paths'][:, -1]
)
payoffs = contract.notional * (settlement_prices - contract.strike) / contract.strike / 1e6

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(payoffs, bins=80, color=colors['secondary'], alpha=0.7, edgecolor='white', density=True)
ax.axvline(x=np.mean(payoffs), color=colors['positive'], linewidth=2.5, label=f'Mean: EUR {np.mean(payoffs):.1f}M')
ax.axvline(x=np.percentile(payoffs, 5), color=colors['negative'], linestyle='--', linewidth=2, label=f'5th %ile: EUR {np.percentile(payoffs, 5):.1f}M')
ax.axvline(x=np.percentile(payoffs, 95), color=colors['gold'], linestyle='--', linewidth=2, label=f'95th %ile: EUR {np.percentile(payoffs, 95):.1f}M')
ax.axvline(x=0, color='black', linewidth=1)

ax.set_xlabel('Z Group Payoff (EUR Million)', fontsize=12, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Payoff Distribution (100,000 Paths)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('output/pdf_charts/payoff_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 11. KNOCKOUT ANALYSIS
# =============================================================================
print("\n11. Generating Knockout Analysis Chart...")
knockout_times = np.array([sim_result['times'][knockout_idx[i]] for i in range(len(knocked_out)) if knocked_out[i]])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Knockout time distribution
ax1 = axes[0]
ax1.hist(knockout_times, bins=40, color=colors['secondary'], alpha=0.7, edgecolor='white')
ax1.axvline(x=np.mean(knockout_times), color=colors['gold'], linewidth=2.5, label=f'Mean: {np.mean(knockout_times):.2f} years')
ax1.set_xlabel('Knockout Time (Years)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Knockout Time Distribution', fontsize=12, fontweight='bold')
ax1.legend()

# Barrier breach breakdown
ax2 = axes[1]
ko_eurusd = sim_result['eurusd_paths'][knocked_out, :]
ko_idx = knockout_idx[knocked_out]
ko_prices = np.array([ko_eurusd[i, ko_idx[i]] for i in range(len(ko_idx))])
lower_breaches = np.sum(ko_prices < contract.barrier_lower)
upper_breaches = np.sum(ko_prices > contract.barrier_upper)

breach_labels = ['Lower Barrier\n(EUR/USD < 1.05)', 'Upper Barrier\n(EUR/USD > 1.25)']
breach_counts = [lower_breaches, upper_breaches]
breach_colors = [colors['negative'], colors['gold']]

bars = ax2.bar(breach_labels, breach_counts, color=breach_colors, alpha=0.8, edgecolor='white', linewidth=2)
for bar, count in zip(bars, breach_counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{count:,}\n({count/len(knocked_out)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')

ax2.set_ylabel('Number of Paths', fontsize=11, fontweight='bold')
ax2.set_title('Barrier Breach Analysis', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('output/pdf_charts/knockout_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 12. GOLD PRICE HISTORY (Stylized)
# =============================================================================
print("\n12. Generating Gold History Chart...")
# Create stylized historical chart
months = ['Jan 25', 'Apr 25', 'Jul 25', 'Oct 25', 'Jan 26', 'Feb 26']
gold_history = [2750, 3100, 3800, 4200, 5608, 4900]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(months, gold_history, color=colors['gold'], linewidth=3, marker='o', markersize=8)
ax.fill_between(months, gold_history, alpha=0.2, color=colors['gold'])
ax.axhline(y=contract.strike, color=colors['negative'], linestyle='--', linewidth=2, label=f'Strike ${contract.strike:,.0f}')
ax.annotate('Record High\n$5,608', xy=(4, 5608), xytext=(3.5, 5700), fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=colors['positive']))
ax.annotate('7%+ Crash', xy=(4.5, 5250), xytext=(4.2, 5400), fontsize=9, color=colors['negative'],
            arrowprops=dict(arrowstyle='->', color=colors['negative']))

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Gold Price ($/oz)', fontsize=12)
ax.set_title('Gold Price Evolution (Stylized)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('output/pdf_charts/gold_history.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("ALL CHARTS UPDATED SUCCESSFULLY!")
print("=" * 60)

# =============================================================================
# REGENERATE PDF
# =============================================================================
print("\nRegenerating PDF from updated HTML...")
try:
    from playwright.sync_api import sync_playwright

    html_path = os.path.abspath('output/GAAIF_Report_PDF.html')
    pdf_path = os.path.abspath('submission/GAAIF_Product_Proposal.pdf')

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f'file://{html_path}')
        page.wait_for_timeout(2000)
        page.pdf(path=pdf_path, format='A4', print_background=True, margin={'top': '1cm', 'bottom': '1cm', 'left': '1cm', 'right': '1cm'})
        browser.close()

    print(f"PDF regenerated: {pdf_path}")
except Exception as e:
    print(f"PDF generation error: {e}")
    print("Please regenerate PDF manually using browser print.")

print("\n" + "=" * 60)
print("UPDATE COMPLETE!")
print("=" * 60)
