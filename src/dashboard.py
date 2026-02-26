"""
GAAIF Challenge — Interactive Pricing Dashboard
================================================

Run with:  cd src && streamlit run dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard_data import (
    fetch_live_market_data,
    run_pricing,
    run_cached_sensitivity,
    run_cached_scenarios,
    run_cached_model_comparison,
    run_cached_greeks,
)

# ──────────────────────────── Page config ────────────────────────────

st.set_page_config(
    page_title="GAAIF Structured Forward Pricer",
    page_icon="📊",
    layout="wide",
)

# Color scheme
PRIMARY = "#1a365d"
ACCENT = "#d69e2e"
DANGER = "#e53e3e"
SUCCESS = "#38a169"
LIGHT_BG = "#f7fafc"

# ──────────────────────────── Sidebar ────────────────────────────────

st.sidebar.title("Parameters")

use_live = st.sidebar.toggle("Live market data", value=True)
fetched, provenance_report = fetch_live_market_data(use_live)

st.sidebar.subheader("Market Data")
gold_spot = st.sidebar.slider(
    "Gold Spot (USD/oz)", 2000.0, 8000.0, float(fetched['gold_spot']), 50.0,
    format="$%.0f")
eurusd_spot = st.sidebar.slider(
    "EUR/USD", 1.00, 1.30, float(fetched['eurusd_spot']), 0.005,
    format="%.3f")
gold_vol = st.sidebar.slider(
    "Gold Vol (%)", 5.0, 80.0, float(fetched['sigma_gold'] * 100), 1.0,
    format="%.0f%%")
eurusd_vol = st.sidebar.slider(
    "EUR/USD Vol (%)", 2.0, 25.0, float(fetched['sigma_eurusd'] * 100), 0.5,
    format="%.1f%%")
correlation = st.sidebar.slider(
    "Correlation", -0.9, 0.9, float(fetched['rho']), 0.05)
usd_rate = st.sidebar.slider(
    "USD Rate (%)", 0.0, 8.0, float(fetched['r_usd'] * 100), 0.1,
    format="%.1f%%")
eur_rate = st.sidebar.slider(
    "EUR Rate (%)", 0.0, 6.0, float(fetched['r_eur'] * 100), 0.1,
    format="%.1f%%")
conv_yield = st.sidebar.slider(
    "Convenience Yield (%)", 0.0, 5.0, float(fetched['gold_yield'] * 100), 0.1,
    format="%.1f%%")

st.sidebar.subheader("Contract Terms")
strike = st.sidebar.number_input("Strike (USD/oz)", value=4600.0, step=100.0)
barrier_lower = st.sidebar.number_input("Lower Barrier", value=1.05, step=0.01, format="%.2f")
barrier_upper = st.sidebar.number_input("Upper Barrier", value=1.25, step=0.01, format="%.2f")

st.sidebar.subheader("Simulation")
n_paths = st.sidebar.select_slider(
    "MC Paths", options=[10000, 25000, 50000, 100000, 200000], value=50000)

# Assembled parameter dicts
market_dict = {
    'gold_spot': gold_spot,
    'eurusd_spot': eurusd_spot,
    'r_eur': eur_rate / 100,
    'r_usd': usd_rate / 100,
    'sigma_gold': gold_vol / 100,
    'sigma_eurusd': eurusd_vol / 100,
    'rho': correlation,
    'gold_yield': conv_yield / 100,
}
contract_dict = {
    'notional': 500_000_000,
    'strike': strike,
    'tenor': 2.0,
    'barrier_lower': barrier_lower,
    'barrier_upper': barrier_upper,
}

# ──────────────────────────── Header ─────────────────────────────────

st.title("GAAIF Structured Gold Forward Pricer")
st.caption("EUR 500M Notional | Gold Forward with EUR/USD Double Knock-Out Barriers")

# ──────────────────────────── Tabs ───────────────────────────────────

tab_market, tab_pricing, tab_sens, tab_scenarios, tab_models, tab_greeks = st.tabs([
    "Market Data", "Pricing", "Sensitivity", "Scenarios",
    "Model Comparison", "Greeks",
])

# ═══════════════════════════ TAB 1: MARKET DATA ══════════════════════

with tab_market:
    st.subheader("Current Market Parameters")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gold Spot", f"${gold_spot:,.0f}")
    c2.metric("EUR/USD", f"{eurusd_spot:.4f}")
    c3.metric("Gold Vol", f"{gold_vol:.0f}%")
    c4.metric("Correlation", f"{correlation:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("USD Rate", f"{usd_rate:.1f}%")
    c6.metric("EUR Rate", f"{eur_rate:.1f}%")
    c7.metric("FX Vol", f"{eurusd_vol:.1f}%")
    c8.metric("Conv. Yield", f"{conv_yield:.1f}%")

    # Forward & moneyness
    F = gold_spot * np.exp((usd_rate / 100 - conv_yield / 100) * 2.0)
    moneyness = F / strike
    lower_dist = (eurusd_spot - barrier_lower) / eurusd_spot
    upper_dist = (barrier_upper - eurusd_spot) / eurusd_spot

    st.divider()
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("2Y Forward", f"${F:,.0f}")
    d2.metric("Moneyness (F/K)", f"{moneyness:.1%}")
    d3.metric("Dist. to Lower", f"{lower_dist:.1%}")
    d4.metric("Dist. to Upper", f"{upper_dist:.1%}")

    st.divider()
    st.subheader("Data Provenance")
    st.code(provenance_report, language="text")

# ═══════════════════════════ TAB 2: PRICING ══════════════════════════

with tab_pricing:
    result = run_pricing(market_dict, contract_dict, n_paths)

    st.subheader("Base Case Valuation")
    p1, p2, p3, p4 = st.columns(4)
    pv = result['price_zgroup']
    p1.metric("Z Group PV", f"EUR {pv:+,.0f}")
    p2.metric("KO Rate", f"{result['knockout_rate']:.1%}")
    p3.metric("Avg KO Time", f"{result['avg_knockout_time']:.1f} yr")
    p4.metric("Std Error", f"EUR {result['std_error']:,.0f}")

    p5, p6, p7, p8 = st.columns(4)
    p5.metric("95% CI Lower", f"EUR {result['ci_95_lower']:+,.0f}")
    p6.metric("95% CI Upper", f"EUR {result['ci_95_upper']:+,.0f}")
    p7.metric("Lower Breaches", f"{result['lower_breach_rate']:.1%}")
    p8.metric("Upper Breaches", f"{result['upper_breach_rate']:.1%}")

    # Payoff diagram
    st.divider()
    st.subheader("Payoff Diagram")
    prices = np.linspace(max(strike * 0.5, 2000), strike * 1.8, 200)
    zgroup_payoff = 500_000_000 * (prices - strike) / strike
    abank_payoff = -zgroup_payoff

    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(
        x=prices, y=zgroup_payoff / 1e6, name="Z Group",
        line=dict(color=PRIMARY, width=2.5)))
    fig_payoff.add_trace(go.Scatter(
        x=prices, y=abank_payoff / 1e6, name="Alphabank",
        line=dict(color=DANGER, width=2.5)))
    fig_payoff.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_payoff.add_vline(x=strike, line_dash="dot",
                         annotation_text=f"Strike ${strike:,.0f}",
                         line_color=ACCENT)
    fig_payoff.add_vline(x=gold_spot, line_dash="dot",
                         annotation_text=f"Spot ${gold_spot:,.0f}",
                         line_color=SUCCESS)
    fig_payoff.update_layout(
        xaxis_title="Gold Price (USD/oz)",
        yaxis_title="Payoff (EUR millions)",
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig_payoff, use_container_width=True)

# ═══════════════════════════ TAB 3: SENSITIVITY ══════════════════════

with tab_sens:
    st.subheader("Sensitivity Analysis")
    st.caption(f"Each point is a full MC reprice ({n_paths:,} paths). Adjust path count in the sidebar.")

    if st.button("Run Sensitivity Analysis", key="btn_sens"):
        records = run_cached_sensitivity(market_dict, contract_dict, n_paths)
        st.session_state['sens_data'] = records

    if 'sens_data' in st.session_state:
        df = pd.DataFrame(st.session_state['sens_data'])

        params_map = {
            'gold_spot': ('Gold Spot (USD/oz)', '$'),
            'eurusd_spot': ('EUR/USD Spot', ''),
            'sigma_gold': ('Gold Volatility', '%'),
            'sigma_eurusd': ('EUR/USD Volatility', '%'),
            'correlation': ('Correlation', ''),
        }

        for param, (label, fmt) in params_map.items():
            sub = df[df['parameter'] == param].copy()
            if sub.empty:
                continue
            if fmt == '%':
                sub['value'] = sub['value'] * 100

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=sub['value'], y=sub['price_zgroup'] / 1e6,
                name="Z Group PV (EUR M)", mode="lines+markers",
                line=dict(color=PRIMARY, width=2),
                marker=dict(size=6),
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=sub['value'], y=sub['knockout_rate'] * 100,
                name="KO Rate (%)", mode="lines+markers",
                line=dict(color=DANGER, width=2, dash="dash"),
                marker=dict(size=6),
            ), secondary_y=True)
            fig.update_layout(
                title=f"Sensitivity to {label}",
                height=370,
                template="plotly_white",
                legend=dict(x=0.01, y=0.99),
            )
            fig.update_xaxes(title_text=label + (" (%)" if fmt == '%' else ""))
            fig.update_yaxes(title_text="Z Group PV (EUR M)", secondary_y=False)
            fig.update_yaxes(title_text="KO Rate (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click **Run Sensitivity Analysis** to generate charts.")

# ═══════════════════════════ TAB 4: SCENARIOS ════════════════════════

with tab_scenarios:
    st.subheader("Scenario Analysis")

    if st.button("Run Scenario Analysis", key="btn_scen"):
        records = run_cached_scenarios(market_dict, contract_dict, n_paths)
        st.session_state['scen_data'] = records

    if 'scen_data' in st.session_state:
        df = pd.DataFrame(st.session_state['scen_data'])

        for stype in df['scenario_type'].unique():
            sub = df[df['scenario_type'] == stype].copy()
            st.markdown(f"### {stype.replace('_', ' ').title()} Scenarios")

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=sub['parameter'].astype(str),
                y=sub['price_zgroup'] / 1e6,
                name="Z Group PV (EUR M)",
                marker_color=PRIMARY,
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=sub['parameter'].astype(str),
                y=sub['knockout_rate'] * 100,
                name="KO Rate (%)", mode="lines+markers",
                line=dict(color=DANGER, width=2),
                marker=dict(size=8),
            ), secondary_y=True)
            fig.update_layout(
                height=370,
                template="plotly_white",
                legend=dict(x=0.01, y=0.99),
            )
            fig.update_yaxes(title_text="PV (EUR M)", secondary_y=False)
            fig.update_yaxes(title_text="KO Rate (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full scenario data"):
            st.dataframe(df, use_container_width=True)
    else:
        st.info("Click **Run Scenario Analysis** to generate charts.")

# ═══════════════════════════ TAB 5: MODEL COMPARISON ═════════════════

with tab_models:
    st.subheader("Model Comparison")
    st.caption("GBM vs Heston Stochastic Vol vs Merton Jump-Diffusion")

    if st.button("Run Model Comparison", key="btn_models"):
        mc_results = run_cached_model_comparison(market_dict, contract_dict, n_paths)
        st.session_state['model_data'] = mc_results

    if 'model_data' in st.session_state:
        mc = st.session_state['model_data']
        models = list(mc.keys())
        pvs = [mc[m]['price_zgroup'] / 1e6 for m in models]
        ses = [mc[m]['std_error'] / 1e6 for m in models]
        kos = [mc[m]['knockout_rate'] * 100 for m in models]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=models, y=pvs, name="Z Group PV (EUR M)",
            marker_color=[PRIMARY, ACCENT, SUCCESS],
            error_y=dict(type='data', array=[1.96 * s for s in ses], visible=True),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=models, y=kos, name="KO Rate (%)",
            mode="lines+markers",
            line=dict(color=DANGER, width=2),
            marker=dict(size=10),
        ), secondary_y=True)
        fig.update_layout(
            height=400, template="plotly_white",
            legend=dict(x=0.01, y=0.99),
        )
        fig.update_yaxes(title_text="PV (EUR M)", secondary_y=False)
        fig.update_yaxes(title_text="KO Rate (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Detailed Results")
        rows = []
        for m in models:
            rows.append({
                'Model': m,
                'Z Group PV (EUR)': f"{mc[m]['price_zgroup']:+,.0f}",
                'Std Error': f"{mc[m]['std_error']:,.0f}",
                'KO Rate': f"{mc[m]['knockout_rate']:.1%}",
            })
        st.table(pd.DataFrame(rows))
    else:
        st.info("Click **Run Model Comparison** to generate charts.")

# ═══════════════════════════ TAB 6: GREEKS ═══════════════════════════

with tab_greeks:
    st.subheader("Risk Sensitivities (Greeks)")

    if st.button("Compute Greeks", key="btn_greeks"):
        greeks = run_cached_greeks(market_dict, contract_dict, min(n_paths, 50000))
        st.session_state['greeks_data'] = greeks

    if 'greeks_data' in st.session_state:
        g = st.session_state['greeks_data']

        greek_info = [
            ("Delta (Gold)", g['delta_gold'], "EUR per $1 gold move"),
            ("Gamma (Gold)", g['gamma_gold'], "EUR per ($1)^2"),
            ("Delta (FX)", g['delta_eurusd'], "EUR per 0.01 FX move"),
            ("Vega (Gold)", g['vega_gold'], "EUR per 1% vol"),
            ("Rho (EUR)", g['rho_eur'], "EUR per 1bp rate"),
            ("Corr Sens.", g['correlation_sensitivity'], "EUR per 0.05 corr"),
        ]

        names = [gi[0] for gi in greek_info]
        values = [gi[1] for gi in greek_info]
        colors = [SUCCESS if v >= 0 else DANGER for v in values]

        fig = go.Figure(go.Bar(
            x=names, y=values,
            marker_color=colors,
            text=[f"{v:+,.0f}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            title="Greeks Overview",
            yaxis_title="EUR",
            height=420,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Detailed Greeks")
        rows = [{"Greek": gi[0], "Value": f"{gi[1]:+,.2f}", "Unit": gi[2]}
                for gi in greek_info]
        st.table(pd.DataFrame(rows))
    else:
        st.info("Click **Compute Greeks** to generate charts.")
