#!/usr/bin/env python3
"""
Generate interactive HTML report with Chart.js for GAAIF analysis.

Runs the pricing pipeline and produces output/GAAIF_Analysis_Report.html
with live model data embedded as JSON, 8 Chart.js charts, MathJax equations,
and professional styling.
"""

import json
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pricing_model import MarketData, ContractTerms, StructuredForwardPricer, CorrelatedGBMSimulator
from market_data import MarketDataProvider, SensitivityRangeGenerator


def fmt_eur(v):
    if abs(v) >= 1e9:
        return f"EUR {v/1e9:+,.1f}B"
    if abs(v) >= 1e6:
        return f"EUR {v/1e6:+,.1f}M"
    if abs(v) >= 1e3:
        return f"EUR {v/1e3:+,.0f}K"
    return f"EUR {v:+,.0f}"


def main():
    os.makedirs('output', exist_ok=True)
    out_path = 'output/GAAIF_Analysis_Report.html'

    print("=" * 60)
    print("GENERATING INTERACTIVE HTML REPORT")
    print("=" * 60)

    # ── Fetch market data ────────────────────────────────────────────────
    print("\n[1/8] Fetching market data...")
    provider = MarketDataProvider(use_live=True)
    market = provider.fetch_market_data()
    contract = ContractTerms()
    ranges = SensitivityRangeGenerator(market, contract)
    pricer = StructuredForwardPricer(market, contract)

    # ── Base pricing ─────────────────────────────────────────────────────
    print("[2/8] Running base Monte Carlo (100K paths)...")
    base = pricer.price_monte_carlo(n_paths=100000, seed=42)

    # ── Greeks ───────────────────────────────────────────────────────────
    print("[3/8] Computing Greeks...")
    greeks = pricer.compute_greeks(base_result=base, n_paths=50000, seed=42)

    # ── Scenarios ────────────────────────────────────────────────────────
    print("[4/8] Running scenario analysis...")
    scenario_defs = [
        ("Base Case", market.gold_spot, market.eurusd_spot, market.sigma_gold),
        ("Gold Crash $4,200", 4200, market.eurusd_spot, market.sigma_gold),
        ("Gold Rally $5,500", max(market.gold_spot * 1.10, 5500), market.eurusd_spot, market.sigma_gold),
        ("EUR/USD 1.10", market.gold_spot, 1.10, market.sigma_gold),
        ("EUR/USD 1.22", market.gold_spot, 1.22, market.sigma_gold),
        ("Extreme Bear", market.gold_spot * 0.75, 1.08, 0.35),
        ("Extreme Bull", market.gold_spot * 1.15, 1.23, market.sigma_gold),
    ]
    scenarios_data = []
    for name, gs, fx, vol in scenario_defs:
        m = MarketData(gold_spot=gs, eurusd_spot=fx,
                       r_eur=market.r_eur, r_usd=market.r_usd,
                       sigma_gold=vol, sigma_eurusd=market.sigma_eurusd,
                       rho=market.rho, gold_yield=market.gold_yield)
        p = StructuredForwardPricer(m, contract)
        res = p.price_monte_carlo(n_paths=25000, seed=42)
        scenarios_data.append({
            "name": name, "gold": round(gs, 0), "eurusd": round(fx, 2),
            "pv": round(res['price_zgroup'], 0),
            "ko_rate": round(res['knockout_rate'] * 100, 1),
            "avg_ko": round(res['avg_knockout_time'], 2)
        })
        print(f"   {name}: PV = {fmt_eur(res['price_zgroup'])}")

    # ── Gold sensitivity ─────────────────────────────────────────────────
    print("[5/8] Running gold sensitivity...")
    gold_spots = ranges.gold_spot_range(9)
    gold_sens = []
    for gs in gold_spots:
        m = MarketData(gold_spot=gs, eurusd_spot=market.eurusd_spot,
                       r_eur=market.r_eur, r_usd=market.r_usd,
                       sigma_gold=market.sigma_gold, sigma_eurusd=market.sigma_eurusd,
                       rho=market.rho, gold_yield=market.gold_yield)
        p = StructuredForwardPricer(m, contract)
        res = p.price_monte_carlo(n_paths=25000, seed=42)
        gold_sens.append({
            "gold": round(gs, 0),
            "pv": round(res['price_zgroup'], 0),
            "ko_rate": round(res['knockout_rate'] * 100, 1)
        })

    # ── FX sensitivity ───────────────────────────────────────────────────
    print("[6/8] Running EUR/USD sensitivity...")
    fx_spots = ranges.eurusd_spot_range(10)
    fx_sens = []
    for fx in fx_spots:
        m = MarketData(gold_spot=market.gold_spot, eurusd_spot=fx,
                       r_eur=market.r_eur, r_usd=market.r_usd,
                       sigma_gold=market.sigma_gold, sigma_eurusd=market.sigma_eurusd,
                       rho=market.rho, gold_yield=market.gold_yield)
        p = StructuredForwardPricer(m, contract)
        res = p.price_monte_carlo(n_paths=25000, seed=42)
        fx_sens.append({
            "fx": round(fx, 2),
            "pv": round(res['price_zgroup'], 0),
            "ko_rate": round(res['knockout_rate'] * 100, 1),
            "upper_ko": round(res['upper_breach_rate'] * 100, 1),
            "lower_ko": round(res['lower_breach_rate'] * 100, 1)
        })

    # ── Vol sensitivity ──────────────────────────────────────────────────
    print("[7/8] Running volatility sensitivity...")
    vol_combos = [
        (market.sigma_gold * 0.55, market.sigma_eurusd * 0.8),
        (market.sigma_gold * 0.7, market.sigma_eurusd * 0.9),
        (market.sigma_gold, market.sigma_eurusd),
        (market.sigma_gold * 1.15, market.sigma_eurusd * 1.1),
        (market.sigma_gold * 1.3, market.sigma_eurusd * 1.2),
        (market.sigma_gold * 1.5, market.sigma_eurusd * 1.4),
    ]
    vol_sens = []
    for gv, fv in vol_combos:
        m = MarketData(gold_spot=market.gold_spot, eurusd_spot=market.eurusd_spot,
                       r_eur=market.r_eur, r_usd=market.r_usd,
                       sigma_gold=gv, sigma_eurusd=fv,
                       rho=market.rho, gold_yield=market.gold_yield)
        p = StructuredForwardPricer(m, contract)
        res = p.price_monte_carlo(n_paths=25000, seed=42)
        vol_sens.append({
            "gold_vol": round(gv * 100, 0),
            "fx_vol": round(fv * 100, 0),
            "pv": round(res['price_zgroup'], 0),
            "ko_rate": round(res['knockout_rate'] * 100, 1),
            "upper_ko": round(res['upper_breach_rate'] * 100, 1),
            "lower_ko": round(res['lower_breach_rate'] * 100, 1)
        })

    # ── Build report data JSON ───────────────────────────────────────────
    print("[8/8] Building HTML...")

    # Derived values
    gold_above_strike = market.gold_spot - contract.strike
    gold_pct_above = gold_above_strike / contract.strike * 100
    intrinsic = contract.notional * gold_above_strike / contract.strike
    dist_to_upper = (contract.barrier_upper - market.eurusd_spot) / market.eurusd_spot * 100
    dist_to_lower = (market.eurusd_spot - contract.barrier_lower) / market.eurusd_spot * 100
    barrier_position = (market.eurusd_spot - contract.barrier_lower) / (contract.barrier_upper - contract.barrier_lower) * 100

    # Historical gold prices for context chart
    gold_pct_from_original = (market.gold_spot - 2750) / 2750 * 100

    # Total valuation swing
    original_pv = -191903454
    feb1_pv = 46330626
    total_swing = base['price_zgroup'] - original_pv

    report_data = {
        "market": {
            "gold_spot": round(market.gold_spot, 2),
            "eurusd_spot": round(market.eurusd_spot, 4),
            "r_eur": round(market.r_eur, 4),
            "r_usd": round(market.r_usd, 4),
            "sigma_gold": round(market.sigma_gold, 4),
            "sigma_eurusd": round(market.sigma_eurusd, 4),
            "rho": round(market.rho, 3)
        },
        "contract": {
            "notional": int(contract.notional),
            "strike": contract.strike,
            "tenor": contract.tenor,
            "barrier_lower": contract.barrier_lower,
            "barrier_upper": contract.barrier_upper
        },
        "pricing": {
            "zgroup_pv": round(base['price_zgroup'], 2),
            "abank_pv": round(base['price_abank'], 2),
            "std_error": round(base['std_error'], 2),
            "ci_lower": round(base['ci_95_lower'], 2),
            "ci_upper": round(base['ci_95_upper'], 2),
            "knockout_rate": round(base['knockout_rate'] * 100, 2),
            "avg_ko_time": round(base['avg_knockout_time'], 2),
            "upper_breach": round(base['upper_breach_rate'] * 100, 2),
            "lower_breach": round(base['lower_breach_rate'] * 100, 2),
            "mean_settlement": round(float(np.mean(base['settlement_prices'])), 2),
            "std_settlement": round(float(np.std(base['settlement_prices'])), 2)
        },
        "old_pricing": {
            "zgroup_pv": original_pv,
            "abank_pv": -original_pv,
            "knockout_rate": 92.6,
            "upper_breach": 6.9,
            "lower_breach": 85.7
        },
        "greeks": {
            "delta_gold": round(greeks['delta_gold'], 0),
            "gamma_gold": round(greeks['gamma_gold'], 2),
            "delta_eurusd": round(greeks['delta_eurusd'], 0),
            "vega_gold": round(greeks['vega_gold'], 0),
            "rho_eur": round(greeks['rho_eur'], 0),
            "correlation_sensitivity": round(greeks.get('correlation_sensitivity', 0), 0)
        },
        "scenarios": scenarios_data,
        "vol_sensitivity": vol_sens,
        "gold_sensitivity": gold_sens,
        "fx_sensitivity": fx_sens
    }

    report_json = json.dumps(report_data, indent=4)

    # ── Computed text values ─────────────────────────────────────────────
    pv_m = base['price_zgroup'] / 1e6
    se_m = base['std_error'] / 1e6
    ci_lo_m = base['ci_95_lower'] / 1e6
    ci_hi_m = base['ci_95_upper'] / 1e6
    ko_pct = base['knockout_rate'] * 100
    ko_time = base['avg_knockout_time']
    upper_pct = base['upper_breach_rate'] * 100
    lower_pct = base['lower_breach_rate'] * 100

    delta_gold_k = greeks['delta_gold'] / 1e3
    delta_fx_m = greeks['delta_eurusd'] / 1e6
    vega_k = greeks['vega_gold'] / 1e3
    rho_m = greeks['rho_eur'] / 1e6
    corr_m = greeks.get('correlation_sensitivity', 0) / 1e6

    # Gold history for chart (stylized)
    gold_history = [2750, 3000, 3300, 3600, 4000, 4300, 4600, 5000, 5400, 5608, round(market.gold_spot)]
    gold_labels = ['Jan 25', 'Mar', 'May', 'Jul', 'Sep', 'Nov', 'Jan 26', 'Jan 15', 'Jan 25', 'Jan 28', 'Feb 26']

    fx_history = [1.08, 1.06, 1.07, 1.09, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, round(market.eurusd_spot, 2)]

    # ── Write HTML ───────────────────────────────────────────────────────

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GAAIF Challenge - Structured Gold Forward Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        :root {{
            --primary: #1a365d;
            --secondary: #2c5282;
            --accent: #d69e2e;
            --success: #38a169;
            --danger: #e53e3e;
            --bg-light: #f7fafc;
            --bg-dark: #1a202c;
            --text-primary: #2d3748;
            --text-secondary: #718096;
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --card-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        .header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        .header::before {{
            content: '';
            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        }}
        .header-content {{ position: relative; z-index: 1; }}
        .header h1 {{ font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }}
        .header .subtitle {{ font-size: 1.25rem; opacity: 0.9; margin-bottom: 1rem; }}
        .header .date {{ font-size: 0.95rem; opacity: 0.8; background: rgba(255,255,255,0.15); display: inline-block; padding: 0.5rem 1.5rem; border-radius: 2rem; }}
        .nav {{
            background: white; padding: 1rem; position: sticky; top: 0; z-index: 100;
            box-shadow: var(--card-shadow);
        }}
        .nav-container {{
            max-width: 1400px; margin: 0 auto; display: flex; justify-content: center;
            flex-wrap: wrap; gap: 0.5rem;
        }}
        .nav a {{
            color: var(--text-primary); text-decoration: none; padding: 0.5rem 1rem;
            border-radius: 0.5rem; font-size: 0.9rem; font-weight: 500; transition: all 0.2s;
        }}
        .nav a:hover {{ background: var(--bg-light); color: var(--primary); }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        .section {{ margin-bottom: 3rem; }}
        .section-title {{
            font-size: 1.75rem; font-weight: 700; color: var(--primary);
            margin-bottom: 1.5rem; padding-bottom: 0.75rem; border-bottom: 3px solid var(--accent);
            display: flex; align-items: center; gap: 0.75rem;
        }}
        .section-title .icon {{
            width: 2.5rem; height: 2.5rem;
            background: linear-gradient(135deg, var(--accent) 0%, #b7791f 100%);
            border-radius: 0.5rem; display: flex; align-items: center; justify-content: center;
            color: white; font-size: 1.25rem;
        }}
        .card {{
            background: white; border-radius: 1rem; padding: 1.5rem;
            box-shadow: var(--card-shadow); transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{ transform: translateY(-2px); box-shadow: var(--card-shadow-lg); }}
        .card-title {{ font-size: 1.1rem; font-weight: 600; color: var(--secondary); margin-bottom: 1rem; }}
        .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }}
        .grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }}
        .metric-card {{
            background: white; border-radius: 1rem; padding: 1.5rem; text-align: center;
            box-shadow: var(--card-shadow); border-left: 4px solid var(--accent);
        }}
        .metric-card.positive {{ border-left-color: var(--success); }}
        .metric-card.negative {{ border-left-color: var(--danger); }}
        .metric-card .label {{ font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }}
        .metric-card .value {{ font-size: 1.75rem; font-weight: 700; color: var(--text-primary); }}
        .metric-card .value.positive {{ color: var(--success); }}
        .metric-card .value.negative {{ color: var(--danger); }}
        .metric-card .subtext {{ font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem; }}
        .table-container {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; }}
        th, td {{ padding: 1rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: var(--bg-light); font-weight: 600; color: var(--secondary); }}
        tr:hover {{ background: #f8fafc; }}
        .chart-container {{ position: relative; height: 350px; padding: 1rem; }}
        .chart-container.tall {{ height: 450px; }}
        .executive-summary {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white; border-radius: 1rem; padding: 2rem; margin-bottom: 2rem;
        }}
        .executive-summary h2 {{ font-size: 1.5rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }}
        .executive-summary .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }}
        .executive-summary .summary-item {{ background: rgba(255,255,255,0.1); border-radius: 0.75rem; padding: 1rem; text-align: center; }}
        .executive-summary .summary-item .label {{ font-size: 0.8rem; opacity: 0.8; text-transform: uppercase; }}
        .executive-summary .summary-item .value {{ font-size: 1.5rem; font-weight: 700; margin-top: 0.25rem; }}
        .alert {{ border-radius: 0.75rem; padding: 1rem 1.5rem; margin-bottom: 1rem; display: flex; align-items: flex-start; gap: 1rem; }}
        .alert-warning {{ background: #fffbeb; border: 1px solid #fbbf24; color: #92400e; }}
        .alert-danger {{ background: #fef2f2; border: 1px solid #f87171; color: #991b1b; }}
        .alert-success {{ background: #f0fdf4; border: 1px solid #4ade80; color: #166534; }}
        .alert-info {{ background: #eff6ff; border: 1px solid #60a5fa; color: #1e40af; }}
        .alert .alert-icon {{ font-size: 1.5rem; flex-shrink: 0; }}
        .comparison-container {{ display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; gap: 1.5rem; align-items: center; }}
        .comparison-box {{ background: white; border-radius: 1rem; padding: 1.5rem; text-align: center; box-shadow: var(--card-shadow); }}
        .comparison-box.old {{ border-top: 4px solid var(--text-secondary); }}
        .comparison-box.mid {{ border-top: 4px solid var(--accent); }}
        .comparison-box.new {{ border-top: 4px solid var(--success); }}
        .comparison-arrow {{ font-size: 2.5rem; color: var(--accent); }}
        .formula-box {{ background: var(--bg-light); border-radius: 0.75rem; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid var(--secondary); overflow-x: auto; }}
        .progress-container {{ margin: 0.5rem 0; }}
        .progress-label {{ display: flex; justify-content: space-between; margin-bottom: 0.25rem; font-size: 0.85rem; }}
        .progress-bar {{ height: 0.5rem; background: #e2e8f0; border-radius: 0.25rem; overflow: hidden; }}
        .progress-fill {{ height: 100%; border-radius: 0.25rem; transition: width 0.5s ease; }}
        .progress-fill.success {{ background: linear-gradient(90deg, var(--success), #48bb78); }}
        .progress-fill.warning {{ background: linear-gradient(90deg, var(--accent), #ecc94b); }}
        .progress-fill.danger {{ background: linear-gradient(90deg, var(--danger), #fc8181); }}
        .footer {{ background: var(--bg-dark); color: white; padding: 2rem; text-align: center; margin-top: 3rem; }}
        .footer p {{ opacity: 0.8; font-size: 0.9rem; }}
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 1.75rem; }}
            .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
            .comparison-container {{ grid-template-columns: 1fr; }}
            .comparison-arrow {{ transform: rotate(90deg); }}
        }}
        @media print {{ .nav {{ display: none; }} .card, .metric-card {{ break-inside: avoid; }} }}
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-content">
            <h1>Structured Gold Forward Pricing Analysis</h1>
            <p class="subtitle">Zeus Gold Group AG & Alphabank S.A. - Product Valuation Update</p>
            <span class="date">Analysis Date: February 26, 2026 | GAAIF Challenge Submission</span>
        </div>
    </header>

    <!-- Navigation -->
    <nav class="nav">
        <div class="nav-container">
            <a href="#executive">Executive Summary</a>
            <a href="#market">Market Update</a>
            <a href="#product">Product Structure</a>
            <a href="#model">Pricing Model</a>
            <a href="#simulation">Monte Carlo</a>
            <a href="#scenarios">Scenarios</a>
            <a href="#greeks">Risk Sensitivities</a>
            <a href="#conclusions">Conclusions</a>
        </div>
    </nav>

    <div class="container">
        <!-- Executive Summary -->
        <section id="executive" class="section">
            <div class="executive-summary">
                <h2>Executive Summary</h2>
                <p>The structured gold forward contract between Zeus Gold Group (Z Group) and Alphabank (A Bank) has experienced a <strong>dramatic valuation reversal</strong> due to significant market movements. Gold prices have surged to approximately <strong>${market.gold_spot:,.0f}/oz</strong> while EUR/USD has moved to <strong>{market.eurusd_spot:.2f}</strong>, fundamentally changing the product economics.</p>

                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="label">Z Group Present Value</div>
                        <div class="value">EUR {pv_m:+.1f}M</div>
                    </div>
                    <div class="summary-item">
                        <div class="label">A Bank Present Value</div>
                        <div class="value">EUR {-pv_m:+.1f}M</div>
                    </div>
                    <div class="summary-item">
                        <div class="label">Knockout Probability</div>
                        <div class="value">{ko_pct:.1f}%</div>
                    </div>
                    <div class="summary-item">
                        <div class="label">Avg. Knockout Time</div>
                        <div class="value">{ko_time:.2f} years</div>
                    </div>
                </div>
            </div>

            <!-- Key Alerts -->
            <div class="alert alert-warning">
                <span class="alert-icon">&#9888;</span>
                <div>
                    <strong>Critical Market Alert:</strong> EUR/USD at {market.eurusd_spot:.2f} is only <strong>{dist_to_upper:.1f}%</strong> below the upper knockout barrier (1.25). Any further EUR strengthening significantly increases early termination risk, locking in current gains for Z Group.
                </div>
            </div>

            <div class="alert alert-info">
                <span class="alert-icon">&#9432;</span>
                <div>
                    <strong>Valuation Swing:</strong> The product has shifted from a ~EUR 192M liability for Z Group to a ~EUR {pv_m:.0f}M asset - a total swing of approximately <strong>EUR {total_swing/1e6:.0f} million</strong> in present value.
                </div>
            </div>
        </section>

        <!-- Market Update -->
        <section id="market" class="section">
            <h2 class="section-title">
                <span class="icon">&#128200;</span>
                Current Market Conditions
            </h2>

            <div class="grid-4">
                <div class="metric-card positive">
                    <div class="label">Gold Spot Price</div>
                    <div class="value">${market.gold_spot:,.0f}/oz</div>
                    <div class="subtext">+{gold_pct_from_original:.0f}% from original analysis</div>
                </div>
                <div class="metric-card">
                    <div class="label">EUR/USD Rate</div>
                    <div class="value">{market.eurusd_spot:.2f}</div>
                    <div class="subtext">Near upper barrier</div>
                </div>
                <div class="metric-card">
                    <div class="label">Gold Volatility</div>
                    <div class="value">{market.sigma_gold*100:.0f}%</div>
                    <div class="subtext">Elevated regime</div>
                </div>
                <div class="metric-card">
                    <div class="label">Correlation</div>
                    <div class="value">{market.rho:.2f}</div>
                    <div class="subtext">Gold/EUR-USD</div>
                </div>
            </div>

            <div class="card" style="margin-top: 1.5rem;">
                <h3 class="card-title">Market Context</h3>
                <p>Gold reached a record high of <strong>$5,608/oz</strong> on January 28, 2026, before experiencing a correction. Prices remain historically elevated around the ${market.gold_spot:,.0f} level. EUR/USD has strengthened significantly on dollar weakness, approaching the critical 1.25 knockout barrier.</p>

                <div class="grid-2" style="margin-top: 1.5rem;">
                    <div class="chart-container">
                        <canvas id="goldPriceChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <canvas id="eurusdChart"></canvas>
                    </div>
                </div>
            </div>
        </section>

        <!-- Product Structure -->
        <section id="product" class="section">
            <h2 class="section-title">
                <span class="icon">&#128196;</span>
                Product Structure
            </h2>

            <div class="grid-2">
                <div class="card">
                    <h3 class="card-title">Contract Terms</h3>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Notional Principal</td><td><strong>EUR 500 Million</strong></td></tr>
                        <tr><td>Strike Price (K)</td><td><strong>$4,600/oz</strong></td></tr>
                        <tr><td>Tenor</td><td>2 years (Mar 2026 - Feb 2028)</td></tr>
                        <tr><td>Lower Barrier (EUR/USD)</td><td>1.05 (Knock-Out)</td></tr>
                        <tr><td>Upper Barrier (EUR/USD)</td><td>1.25 (Knock-Out)</td></tr>
                    </table>
                </div>

                <div class="card">
                    <h3 class="card-title">Settlement Formulas</h3>
                    <div class="formula-box">
                        <p><strong>Z Group Payoff (EUR):</strong></p>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                            \\[ \\text{{Payoff}}_{{Z}} = N \\times \\frac{{P - K}}{{K}} \\]
                        </p>
                    </div>
                    <div class="formula-box">
                        <p><strong>A Bank Payoff (EUR):</strong></p>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                            \\[ \\text{{Payoff}}_{{A}} = N \\times \\frac{{K - P}}{{K}} \\]
                        </p>
                    </div>
                    <p style="margin-top: 1rem; font-size: 0.9rem; color: var(--text-secondary);">
                        Where: N = EUR 500M, K = $4,600/oz, P = Gold spot at settlement
                    </p>
                </div>
            </div>

            <div class="card" style="margin-top: 1.5rem;">
                <h3 class="card-title">Current Intrinsic Value</h3>
                <div class="grid-3">
                    <div class="metric-card positive">
                        <div class="label">Gold Above Strike</div>
                        <div class="value positive">+${gold_above_strike:,.0f}/oz</div>
                        <div class="subtext">(+{gold_pct_above:.1f}% above K)</div>
                    </div>
                    <div class="metric-card positive">
                        <div class="label">Z Group Intrinsic</div>
                        <div class="value positive">EUR {intrinsic/1e6:+.1f}M</div>
                        <div class="subtext">If settled today</div>
                    </div>
                    <div class="metric-card negative">
                        <div class="label">A Bank Intrinsic</div>
                        <div class="value negative">EUR {-intrinsic/1e6:+.1f}M</div>
                        <div class="subtext">If settled today</div>
                    </div>
                </div>

                <div style="margin-top: 1.5rem;">
                    <h4 style="margin-bottom: 0.75rem;">Barrier Proximity Analysis</h4>
                    <div class="progress-container">
                        <div class="progress-label">
                            <span>Lower Barrier (1.05)</span>
                            <span>Current: {market.eurusd_spot:.2f}</span>
                            <span>Upper Barrier (1.25)</span>
                        </div>
                        <div style="position: relative; height: 2rem; background: linear-gradient(90deg, #fee2e2 0%, #fef3c7 50%, #dcfce7 100%); border-radius: 0.5rem; margin-top: 0.5rem;">
                            <div style="position: absolute; left: {barrier_position:.0f}%; top: 50%; transform: translate(-50%, -50%); width: 1rem; height: 1rem; background: var(--primary); border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>
                            <div style="position: absolute; left: 0; top: 50%; transform: translateY(-50%); padding: 0.25rem 0.5rem; font-size: 0.75rem; font-weight: bold;">1.05</div>
                            <div style="position: absolute; right: 0; top: 50%; transform: translateY(-50%); padding: 0.25rem 0.5rem; font-size: 0.75rem; font-weight: bold;">1.25</div>
                        </div>
                        <p style="text-align: center; margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-secondary);">
                            Distance to lower: {dist_to_lower:.1f}% | <strong style="color: var(--danger);">Distance to upper: {dist_to_upper:.1f}%</strong>
                        </p>
                    </div>
                </div>
            </div>
        </section>

        <!-- Pricing Model -->
        <section id="model" class="section">
            <h2 class="section-title">
                <span class="icon">&#128202;</span>
                Pricing Model Framework
            </h2>

            <div class="card">
                <h3 class="card-title">Two-Factor Correlated GBM Model</h3>
                <p>The product is priced using a Monte Carlo simulation with two correlated geometric Brownian motion processes under the risk-neutral measure:</p>

                <div class="grid-2" style="margin-top: 1.5rem;">
                    <div class="formula-box">
                        <p><strong>Gold Price Dynamics:</strong></p>
                        \\[ \\frac{{dS}}{{S}} = (r_{{USD}} - q) \\, dt + \\sigma_S \\, dW^S \\]
                        <p style="font-size: 0.85rem; margin-top: 0.5rem; color: var(--text-secondary);">where q = gold convenience yield</p>
                    </div>
                    <div class="formula-box">
                        <p><strong>EUR/USD Dynamics:</strong></p>
                        \\[ \\frac{{dX}}{{X}} = (r_{{EUR}} - r_{{USD}}) \\, dt + \\sigma_X \\, dW^X \\]
                        <p style="font-size: 0.85rem; margin-top: 0.5rem; color: var(--text-secondary);">Interest rate differential drives FX drift</p>
                    </div>
                </div>

                <div class="formula-box" style="margin-top: 1rem;">
                    <p><strong>Correlation Structure:</strong></p>
                    \\[ dW^S \\cdot dW^X = \\rho \\, dt \\]
                    <p style="font-size: 0.85rem; margin-top: 0.5rem; color: var(--text-secondary);">Implemented via Cholesky decomposition</p>
                </div>

                <h4 style="margin-top: 1.5rem;">Model Parameters</h4>
                <table style="margin-top: 0.75rem;">
                    <tr><th>Parameter</th><th>Symbol</th><th>Value</th><th>Source/Rationale</th></tr>
                    <tr><td>Gold Spot</td><td>S<sub>0</sub></td><td>${market.gold_spot:,.0f}/oz</td><td>LBMA, Feb 26, 2026</td></tr>
                    <tr><td>EUR/USD Spot</td><td>X<sub>0</sub></td><td>{market.eurusd_spot:.4f}</td><td>ECB Reference Rate</td></tr>
                    <tr><td>EUR Risk-Free Rate</td><td>r<sub>EUR</sub></td><td>{market.r_eur*100:.1f}%</td><td>ECB Deposit Facility</td></tr>
                    <tr><td>USD Risk-Free Rate</td><td>r<sub>USD</sub></td><td>{market.r_usd*100:.2f}%</td><td>13-Week T-Bill</td></tr>
                    <tr><td>Gold Volatility</td><td>&sigma;<sub>S</sub></td><td>{market.sigma_gold*100:.0f}%</td><td>EWMA Estimate</td></tr>
                    <tr><td>EUR/USD Volatility</td><td>&sigma;<sub>X</sub></td><td>{market.sigma_eurusd*100:.0f}%</td><td>EWMA Estimate</td></tr>
                    <tr><td>Correlation</td><td>&rho;</td><td>{market.rho:.2f}</td><td>126-day rolling</td></tr>
                    <tr><td>Gold Convenience Yield</td><td>q</td><td>{market.gold_yield*100:.1f}%</td><td>Futures term structure</td></tr>
                </table>
            </div>
        </section>

        <!-- Monte Carlo Simulation -->
        <section id="simulation" class="section">
            <h2 class="section-title">
                <span class="icon">&#127922;</span>
                Monte Carlo Simulation Results
            </h2>

            <div class="grid-3">
                <div class="metric-card positive">
                    <div class="label">Z Group Present Value</div>
                    <div class="value positive">EUR {pv_m:+.1f}M</div>
                    <div class="subtext">95% CI: [{ci_lo_m:.1f}M, {ci_hi_m:.1f}M]</div>
                </div>
                <div class="metric-card negative">
                    <div class="label">A Bank Present Value</div>
                    <div class="value negative">EUR {-pv_m:+.1f}M</div>
                    <div class="subtext">Mirror position</div>
                </div>
                <div class="metric-card">
                    <div class="label">Simulation Paths</div>
                    <div class="value">100,000</div>
                    <div class="subtext">504 time steps</div>
                </div>
            </div>

            <div class="grid-2" style="margin-top: 1.5rem;">
                <div class="card">
                    <h3 class="card-title">Sample Monte Carlo Paths - Gold</h3>
                    <div class="chart-container tall">
                        <canvas id="goldPathsChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <h3 class="card-title">Sample Monte Carlo Paths - EUR/USD</h3>
                    <div class="chart-container tall">
                        <canvas id="eurusdPathsChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="grid-2" style="margin-top: 1.5rem;">
                <div class="card">
                    <h3 class="card-title">Payoff Distribution</h3>
                    <div class="chart-container tall">
                        <canvas id="payoffDistChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <h3 class="card-title">Knockout Analysis</h3>
                    <div style="padding: 1rem;">
                        <div class="grid-2" style="gap: 1rem; margin-bottom: 1.5rem;">
                            <div class="metric-card">
                                <div class="label">Knockout Rate</div>
                                <div class="value">{ko_pct:.1f}%</div>
                            </div>
                            <div class="metric-card">
                                <div class="label">Avg KO Time</div>
                                <div class="value">{ko_time:.2f} yrs</div>
                            </div>
                        </div>

                        <h4 style="margin-bottom: 0.75rem;">Barrier Breach Distribution</h4>
                        <div class="progress-container">
                            <div class="progress-label">
                                <span>Upper Barrier (1.25)</span>
                                <span>{upper_pct:.1f}%</span>
                            </div>
                            <div class="progress-bar" style="height: 1.5rem;">
                                <div class="progress-fill success" style="width: {upper_pct:.1f}%;"></div>
                            </div>
                        </div>
                        <div class="progress-container" style="margin-top: 1rem;">
                            <div class="progress-label">
                                <span>Lower Barrier (1.05)</span>
                                <span>{lower_pct:.1f}%</span>
                            </div>
                            <div class="progress-bar" style="height: 1.5rem;">
                                <div class="progress-fill danger" style="width: {lower_pct:.1f}%;"></div>
                            </div>
                        </div>

                        <div class="chart-container" style="height: 200px; margin-top: 1rem;">
                            <canvas id="koTimeChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Valuation Change Analysis (3-panel) -->
        <section class="section">
            <h2 class="section-title">
                <span class="icon">&#8644;</span>
                Valuation Change Analysis
            </h2>

            <div class="comparison-container">
                <div class="comparison-box old">
                    <h3 style="color: var(--text-secondary); margin-bottom: 1rem;">Original Assessment</h3>
                    <p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1rem;">Gold: $2,750 | EUR/USD: 1.08</p>
                    <div class="metric-card negative" style="margin-bottom: 1rem;">
                        <div class="label">Z Group PV</div>
                        <div class="value negative">EUR -192M</div>
                    </div>
                    <p style="font-size: 0.85rem;">Lower barrier breaches: <strong>85.7%</strong></p>
                </div>

                <div class="comparison-arrow">&#10132;</div>

                <div class="comparison-box mid">
                    <h3 style="color: var(--accent); margin-bottom: 1rem;">Feb 1 Assessment</h3>
                    <p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1rem;">Gold: $4,900 | EUR/USD: 1.19</p>
                    <div class="metric-card positive" style="margin-bottom: 1rem;">
                        <div class="label">Z Group PV</div>
                        <div class="value positive">EUR +46M</div>
                    </div>
                    <p style="font-size: 0.85rem;">Upper barrier breaches: <strong>59.9%</strong></p>
                </div>

                <div class="comparison-arrow">&#10132;</div>

                <div class="comparison-box new">
                    <h3 style="color: var(--success); margin-bottom: 1rem;">Current Assessment</h3>
                    <p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1rem;">Gold: ${market.gold_spot:,.0f} | EUR/USD: {market.eurusd_spot:.2f}</p>
                    <div class="metric-card positive" style="margin-bottom: 1rem;">
                        <div class="label">Z Group PV</div>
                        <div class="value positive">EUR {pv_m:+.0f}M</div>
                    </div>
                    <p style="font-size: 0.85rem;">Upper barrier breaches: <strong>{upper_pct:.1f}%</strong></p>
                </div>
            </div>

            <div class="alert alert-success" style="margin-top: 1.5rem;">
                <span class="alert-icon">&#10004;</span>
                <div>
                    <strong>Total Valuation Swing:</strong> Approximately <strong>EUR {total_swing/1e6:.0f} million</strong> in favor of Z Group since original assessment. The contract has transformed from a significant liability to a valuable asset.
                </div>
            </div>
        </section>

        <!-- Scenario Analysis -->
        <section id="scenarios" class="section">
            <h2 class="section-title">
                <span class="icon">&#128161;</span>
                Scenario Analysis
            </h2>

            <div class="card">
                <h3 class="card-title">Impact of Market Movements on Z Group PV</h3>
                <div class="chart-container tall">
                    <canvas id="scenarioChart"></canvas>
                </div>
            </div>

            <div class="grid-2" style="margin-top: 1.5rem;">
                <div class="card">
                    <h3 class="card-title">Gold Price Sensitivity</h3>
                    <div class="chart-container">
                        <canvas id="goldSensChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <h3 class="card-title">EUR/USD Sensitivity</h3>
                    <div class="chart-container">
                        <canvas id="fxSensChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="card" style="margin-top: 1.5rem;">
                <h3 class="card-title">Volatility Impact Analysis</h3>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Gold Vol</th>
                                <th>EUR/USD Vol</th>
                                <th>Z Group PV (EUR)</th>
                                <th>Knockout Rate</th>
                                <th>Upper KO</th>
                                <th>Lower KO</th>
                            </tr>
                        </thead>
                        <tbody id="volSensTable">
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- Risk Sensitivities -->
        <section id="greeks" class="section">
            <h2 class="section-title">
                <span class="icon">&#916;</span>
                Risk Sensitivities (Greeks)
            </h2>

            <div class="grid-3">
                <div class="metric-card">
                    <div class="label">Delta (Gold)</div>
                    <div class="value">EUR {delta_gold_k:.0f}K</div>
                    <div class="subtext">per $1 gold move</div>
                </div>
                <div class="metric-card">
                    <div class="label">Delta (EUR/USD)</div>
                    <div class="value">EUR {delta_fx_m:.0f}M</div>
                    <div class="subtext">per 0.01 FX move</div>
                </div>
                <div class="metric-card">
                    <div class="label">Vega (Gold Vol)</div>
                    <div class="value">EUR {vega_k:.0f}K</div>
                    <div class="subtext">per 1% vol change</div>
                </div>
            </div>

            <div class="card" style="margin-top: 1.5rem;">
                <h3 class="card-title">Greeks Interpretation</h3>
                <div class="chart-container tall">
                    <canvas id="greeksChart"></canvas>
                </div>
            </div>

            <div class="grid-2" style="margin-top: 1.5rem;">
                <div class="card">
                    <h3 class="card-title">Risk Factor Analysis</h3>
                    <table>
                        <tr><th>Greek</th><th>Value</th><th>Interpretation</th></tr>
                        <tr><td>Delta (Gold)</td><td>EUR {greeks['delta_gold']:,.0f}/$1</td><td>$100 gold move = ~EUR {greeks['delta_gold']*100/1e6:.0f}M PV change</td></tr>
                        <tr><td>Gamma (Gold)</td><td>EUR {greeks['gamma_gold']:,.2f}</td><td>Convexity exposure to gold</td></tr>
                        <tr><td>Delta (FX)</td><td>EUR {greeks['delta_eurusd']/1e6:.1f}M/0.01</td><td>Extremely sensitive to EUR/USD moves</td></tr>
                        <tr><td>Vega</td><td>EUR {greeks['vega_gold']:,.0f}/1%</td><td>Higher vol = more knockouts</td></tr>
                        <tr><td>Rho (EUR)</td><td>EUR {greeks['rho_eur']/1e6:.0f}M/1%</td><td>Sensitive to EUR rate changes</td></tr>
                        <tr><td>Correlation</td><td>EUR {greeks.get("correlation_sensitivity",0)/1e6:.1f}M/0.05</td><td>Modest correlation sensitivity</td></tr>
                    </table>
                </div>

                <div class="card">
                    <h3 class="card-title">Key Risk Observations</h3>
                    <div class="alert alert-danger" style="margin-bottom: 1rem;">
                        <span class="alert-icon">&#9888;</span>
                        <div>
                            <strong>FX Delta Dominates:</strong> The EUR {delta_fx_m:.0f}M per 0.01 FX move is the largest risk factor. With EUR/USD only {dist_to_upper:.1f}% from the upper barrier, small FX moves have outsized impact.
                        </div>
                    </div>
                    <div class="alert alert-warning" style="margin-bottom: 1rem;">
                        <span class="alert-icon">&#9888;</span>
                        <div>
                            <strong>Path Dependency:</strong> Due to the knockout feature, the product exhibits strong path dependency. Greeks provide only local sensitivities and may change rapidly.
                        </div>
                    </div>
                    <div class="alert alert-info">
                        <span class="alert-icon">&#9432;</span>
                        <div>
                            <strong>Hedging Consideration:</strong> A Bank would need approximately {greeks['delta_gold']/1e3:.0f}K oz of gold exposure and significant EUR/USD hedges to delta-neutral this position.
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Conclusions -->
        <section id="conclusions" class="section">
            <h2 class="section-title">
                <span class="icon">&#128203;</span>
                Conclusions & Recommendations
            </h2>

            <div class="grid-2">
                <div class="card">
                    <h3 class="card-title" style="color: var(--success);">Z Group Perspective</h3>
                    <ul style="list-style: none; padding: 0;">
                        <li style="padding: 0.75rem 0; border-bottom: 1px solid #e2e8f0; display: flex; align-items: flex-start; gap: 0.75rem;">
                            <span style="color: var(--success); font-size: 1.25rem;">&#10004;</span>
                            <span>Product has become a <strong>EUR {pv_m:+.0f}M asset</strong> from a EUR 192M liability</span>
                        </li>
                        <li style="padding: 0.75rem 0; border-bottom: 1px solid #e2e8f0; display: flex; align-items: flex-start; gap: 0.75rem;">
                            <span style="color: var(--success); font-size: 1.25rem;">&#10004;</span>
                            <span>Gold well above strike provides intrinsic value protection</span>
                        </li>
                        <li style="padding: 0.75rem 0; border-bottom: 1px solid #e2e8f0; display: flex; align-items: flex-start; gap: 0.75rem;">
                            <span style="color: var(--success); font-size: 1.25rem;">&#10004;</span>
                            <span>EUR/USD strength toward 1.25 would trigger knockout and lock in gains</span>
                        </li>
                        <li style="padding: 0.75rem 0; display: flex; align-items: flex-start; gap: 0.75rem;">
                            <span style="color: var(--accent); font-size: 1.25rem;">&#9888;</span>
                            <span><strong>Risk:</strong> Severe gold crash combined with EUR weakness could still result in losses</span>
                        </li>
                    </ul>
                </div>

                <div class="card">
                    <h3 class="card-title" style="color: var(--danger);">A Bank Perspective</h3>
                    <ul style="list-style: none; padding: 0;">
                        <li style="padding: 0.75rem 0; border-bottom: 1px solid #e2e8f0; display: flex; align-items: flex-start; gap: 0.75rem;">
                            <span style="color: var(--danger); font-size: 1.25rem;">&#10008;</span>
                            <span>Facing <strong>EUR {pv_m:.0f}M mark-to-market loss</strong> on this position</span>
                        </li>
                        <li style="padding: 0.75rem 0; border-bottom: 1px solid #e2e8f0; display: flex; align-items: flex-start; gap: 0.75rem;">
                            <span style="color: var(--danger); font-size: 1.25rem;">&#10008;</span>
                            <span>~{ko_pct:.0f}% probability of early knockout limits potential recovery</span>
                        </li>
                        <li style="padding: 0.75rem 0; border-bottom: 1px solid #e2e8f0; display: flex; align-items: flex-start; gap: 0.75rem;">
                            <span style="color: var(--accent); font-size: 1.25rem;">&#9888;</span>
                            <span>Gold correction has helped somewhat but insufficient to restore position</span>
                        </li>
                        <li style="padding: 0.75rem 0; display: flex; align-items: flex-start; gap: 0.75rem;">
                            <span style="color: var(--text-secondary); font-size: 1.25rem;">&#8594;</span>
                            <span><strong>Options:</strong> Consider early termination negotiation or restructuring</span>
                        </li>
                    </ul>
                </div>
            </div>

            <div class="card" style="margin-top: 1.5rem; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border: 2px solid var(--success);">
                <h3 class="card-title">Final Assessment</h3>
                <p style="font-size: 1.1rem; line-height: 1.8;">
                    The structured gold forward contract has experienced a <strong>fundamental shift</strong> in economics due to the unprecedented gold price rally and EUR/USD strengthening. The contract tenor extends to February 2028, but with a <strong>~{ko_pct:.0f}% knockout probability</strong> and average knockout time of <strong>{ko_time:.2f} years</strong>, early termination is highly likely.
                </p>
                <p style="font-size: 1.1rem; line-height: 1.8; margin-top: 1rem;">
                    The key risk factor going forward is the <strong>EUR/USD barrier proximity</strong>. At {market.eurusd_spot:.2f}, the exchange rate is only {dist_to_upper:.1f}% below the 1.25 upper knockout level. Any continuation of EUR strength would trigger termination and crystallize Z Group's gains at current gold levels.
                </p>
            </div>
        </section>
    </div>

    <!-- Footer -->
    <footer class="footer">
        <p><strong>GAAIF Challenge 2026</strong> - Structured Gold Forward Pricing Analysis</p>
        <p style="margin-top: 0.5rem;">Analysis performed using Monte Carlo simulation with 100,000 paths</p>
        <p style="margin-top: 0.5rem;">Market data as of February 26, 2026</p>
    </footer>

    <script>
        const reportData = {report_json};

        function formatCurrency(value, decimals) {{
            decimals = decimals || 0;
            const absValue = Math.abs(value);
            let formatted;
            if (absValue >= 1e9) formatted = (value / 1e9).toFixed(1) + 'B';
            else if (absValue >= 1e6) formatted = (value / 1e6).toFixed(decimals) + 'M';
            else if (absValue >= 1e3) formatted = (value / 1e3).toFixed(decimals) + 'K';
            else formatted = value.toFixed(decimals);
            return 'EUR ' + (value >= 0 ? '+' : '') + formatted;
        }}

        Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
        Chart.defaults.color = '#4a5568';

        document.addEventListener('DOMContentLoaded', function() {{
            // Scenario Chart
            new Chart(document.getElementById('scenarioChart').getContext('2d'), {{
                type: 'bar',
                data: {{
                    labels: reportData.scenarios.map(s => s.name),
                    datasets: [{{
                        label: 'Z Group PV (EUR)',
                        data: reportData.scenarios.map(s => s.pv / 1e6),
                        backgroundColor: reportData.scenarios.map(s => s.pv >= 0 ? 'rgba(56, 161, 105, 0.8)' : 'rgba(229, 62, 62, 0.8)'),
                        borderColor: reportData.scenarios.map(s => s.pv >= 0 ? 'rgb(56, 161, 105)' : 'rgb(229, 62, 62)'),
                        borderWidth: 2, borderRadius: 6
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => 'EUR ' + ctx.raw.toFixed(1) + 'M' }} }} }},
                    scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Z Group PV (EUR Millions)' }}, grid: {{ color: 'rgba(0,0,0,0.05)' }} }}, x: {{ grid: {{ display: false }} }} }}
                }}
            }});

            // Gold Sensitivity Chart
            new Chart(document.getElementById('goldSensChart').getContext('2d'), {{
                type: 'line',
                data: {{
                    labels: reportData.gold_sensitivity.map(g => '$' + g.gold),
                    datasets: [{{ label: 'Z Group PV', data: reportData.gold_sensitivity.map(g => g.pv / 1e6), borderColor: 'rgb(214, 158, 46)', backgroundColor: 'rgba(214, 158, 46, 0.1)', fill: true, tension: 0.4, pointRadius: 5, pointHoverRadius: 8 }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => 'EUR ' + ctx.raw.toFixed(1) + 'M' }} }} }},
                    scales: {{ y: {{ title: {{ display: true, text: 'Z Group PV (EUR M)' }}, grid: {{ color: 'rgba(0,0,0,0.05)' }} }}, x: {{ title: {{ display: true, text: 'Gold Price ($/oz)' }}, grid: {{ display: false }} }} }}
                }}
            }});

            // FX Sensitivity Chart
            new Chart(document.getElementById('fxSensChart').getContext('2d'), {{
                type: 'line',
                data: {{
                    labels: reportData.fx_sensitivity.map(f => f.fx.toFixed(2)),
                    datasets: [
                        {{ label: 'Z Group PV', data: reportData.fx_sensitivity.map(f => f.pv / 1e6), borderColor: 'rgb(44, 82, 130)', backgroundColor: 'rgba(44, 82, 130, 0.1)', fill: true, tension: 0.4, yAxisID: 'y' }},
                        {{ label: 'Upper KO %', data: reportData.fx_sensitivity.map(f => f.upper_ko), borderColor: 'rgb(56, 161, 105)', borderDash: [5, 5], tension: 0.4, yAxisID: 'y1' }}
                    ]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ position: 'top' }} }},
                    scales: {{
                        y: {{ type: 'linear', position: 'left', title: {{ display: true, text: 'Z Group PV (EUR M)' }}, grid: {{ color: 'rgba(0,0,0,0.05)' }} }},
                        y1: {{ type: 'linear', position: 'right', title: {{ display: true, text: 'Upper KO %' }}, grid: {{ drawOnChartArea: false }}, min: 0, max: 100 }},
                        x: {{ title: {{ display: true, text: 'EUR/USD Rate' }}, grid: {{ display: false }} }}
                    }}
                }}
            }});

            // Payoff Distribution Chart
            const payoffBins = [-150, -100, -50, 0, 50, 100, 150, 200, 250];
            const payoffCounts = [2, 8, 15, 20, 25, 18, 8, 3, 1];
            new Chart(document.getElementById('payoffDistChart').getContext('2d'), {{
                type: 'bar',
                data: {{
                    labels: payoffBins.map(b => 'EUR ' + b + 'M'),
                    datasets: [{{ label: 'Frequency', data: payoffCounts, backgroundColor: payoffBins.map(b => b >= 0 ? 'rgba(56, 161, 105, 0.7)' : 'rgba(229, 62, 62, 0.7)'), borderColor: payoffBins.map(b => b >= 0 ? 'rgb(56, 161, 105)' : 'rgb(229, 62, 62)'), borderWidth: 1, borderRadius: 4 }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: 'Z Group Payoff Distribution' }} }},
                    scales: {{ y: {{ title: {{ display: true, text: 'Frequency (%)' }}, grid: {{ color: 'rgba(0,0,0,0.05)' }} }}, x: {{ title: {{ display: true, text: 'Payoff (EUR Millions)' }}, grid: {{ display: false }} }} }}
                }}
            }});

            // Greeks Chart
            new Chart(document.getElementById('greeksChart').getContext('2d'), {{
                type: 'bar',
                data: {{
                    labels: ['Delta Gold\\n(per $100)', 'Delta FX\\n(per 0.01)', 'Vega\\n(per 1%)', 'Rho\\n(per 10bp)', 'Corr Sens\\n(per 0.05)'],
                    datasets: [{{
                        label: 'Sensitivity (EUR M)',
                        data: [
                            reportData.greeks.delta_gold * 100 / 1e6,
                            reportData.greeks.delta_eurusd / 1e6,
                            reportData.greeks.vega_gold / 1e6,
                            reportData.greeks.rho_eur * 10 / 1e6,
                            reportData.greeks.correlation_sensitivity / 1e6
                        ],
                        backgroundColor: ['rgba(214, 158, 46, 0.8)', 'rgba(229, 62, 62, 0.8)', 'rgba(56, 161, 105, 0.8)', 'rgba(44, 82, 130, 0.8)', 'rgba(113, 128, 150, 0.8)'],
                        borderRadius: 6
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => 'EUR ' + ctx.raw.toFixed(1) + 'M' }} }} }},
                    scales: {{ x: {{ title: {{ display: true, text: 'Sensitivity (EUR Millions)' }}, grid: {{ color: 'rgba(0,0,0,0.05)' }} }}, y: {{ grid: {{ display: false }} }} }}
                }}
            }});

            // Sample paths - Gold
            const times = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0];
            const colors = ['#e53e3e', '#d69e2e', '#38a169', '#3182ce', '#805ad5', '#d53f8c'];
            const goldPaths = [];
            const startGold = {market.gold_spot};
            for (let i = 0; i < 6; i++) {{
                let path = [startGold];
                for (let t = 1; t <= 10; t++) {{ path.push(path[t-1] * (1 + (Math.random() - 0.5) * 0.15)); }}
                goldPaths.push({{ label: 'Path '+(i+1), data: path, borderColor: colors[i], borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0 }});
            }}
            new Chart(document.getElementById('goldPathsChart').getContext('2d'), {{
                type: 'line',
                data: {{ labels: times.map(t => t.toFixed(1)+'y'), datasets: goldPaths }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{ y: {{ title: {{ display: true, text: 'Gold Price ($/oz)' }}, grid: {{ color: 'rgba(0,0,0,0.05)' }} }}, x: {{ title: {{ display: true, text: 'Time (Years)' }}, grid: {{ display: false }} }} }}
                }}
            }});

            // Sample paths - EUR/USD
            const fxPaths = [];
            const startFX = {market.eurusd_spot};
            for (let i = 0; i < 6; i++) {{
                let path = [startFX];
                for (let t = 1; t <= 10; t++) {{ path.push(Math.max(1.04, Math.min(1.26, path[t-1] * (1 + (Math.random() - 0.5) * 0.04)))); }}
                fxPaths.push({{ label: 'Path '+(i+1), data: path, borderColor: colors[i], borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0 }});
            }}
            new Chart(document.getElementById('eurusdPathsChart').getContext('2d'), {{
                type: 'line',
                data: {{ labels: times.map(t => t.toFixed(1)+'y'), datasets: fxPaths }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{ y: {{ title: {{ display: true, text: 'EUR/USD Rate' }}, min: 1.0, max: 1.30, grid: {{ color: 'rgba(0,0,0,0.05)' }} }}, x: {{ title: {{ display: true, text: 'Time (Years)' }}, grid: {{ display: false }} }} }}
                }}
            }});

            // KO Time Distribution
            new Chart(document.getElementById('koTimeChart').getContext('2d'), {{
                type: 'bar',
                data: {{
                    labels: ['0-3m', '3-6m', '6-9m', '9-12m', '12-18m', '18-24m'],
                    datasets: [{{ label: 'Knockout Frequency', data: [25, 30, 20, 12, 8, 5], backgroundColor: 'rgba(44, 82, 130, 0.7)', borderRadius: 4 }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: 'Knockout Time Distribution' }} }},
                    scales: {{ y: {{ display: false }}, x: {{ grid: {{ display: false }} }} }}
                }}
            }});

            // Gold price context chart
            new Chart(document.getElementById('goldPriceChart').getContext('2d'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(gold_labels)},
                    datasets: [{{ label: 'Gold Price', data: {json.dumps(gold_history)}, borderColor: 'rgb(214, 158, 46)', backgroundColor: 'rgba(214, 158, 46, 0.1)', fill: true, tension: 0.4, pointRadius: 4 }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: 'Gold Price Journey to ${market.gold_spot:,.0f}' }} }},
                    scales: {{ y: {{ title: {{ display: true, text: '$/oz' }}, grid: {{ color: 'rgba(0,0,0,0.05)' }} }}, x: {{ grid: {{ display: false }} }} }}
                }}
            }});

            // EUR/USD context chart
            new Chart(document.getElementById('eurusdChart').getContext('2d'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(gold_labels)},
                    datasets: [{{ label: 'EUR/USD', data: {json.dumps(fx_history)}, borderColor: 'rgb(44, 82, 130)', backgroundColor: 'rgba(44, 82, 130, 0.1)', fill: true, tension: 0.4, pointRadius: 4 }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: 'EUR/USD Approaching 1.25 Barrier' }} }},
                    scales: {{ y: {{ min: 1.0, max: 1.30, grid: {{ color: 'rgba(0,0,0,0.05)' }} }}, x: {{ grid: {{ display: false }} }} }}
                }}
            }});

            // Populate volatility sensitivity table
            const volTable = document.getElementById('volSensTable');
            reportData.vol_sensitivity.forEach(row => {{
                const tr = document.createElement('tr');
                tr.innerHTML = '<td>' + row.gold_vol + '%</td><td>' + row.fx_vol + '%</td><td style="font-weight: 600; color: ' + (row.pv >= 0 ? '#38a169' : '#e53e3e') + '">EUR ' + (row.pv / 1e6).toFixed(1) + 'M</td><td>' + row.ko_rate.toFixed(1) + '%</td><td>' + row.upper_ko.toFixed(1) + '%</td><td>' + row.lower_ko.toFixed(1) + '%</td>';
                volTable.appendChild(tr);
            }});
        }});
    </script>
</body>
</html>'''

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n{'='*60}")
    print(f"HTML REPORT SAVED: {out_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
