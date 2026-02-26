"""
Cached data layer for the Streamlit dashboard.

Wraps existing pricing functions with @st.cache_data for reactivity.
All inputs are primitives/dicts (not dataclasses) to ensure hashability.
"""

import streamlit as st
import numpy as np
from typing import Dict, List, Tuple


def _build_market(params: dict):
    """Construct a MarketData instance from a plain dict."""
    from pricing_model import MarketData
    return MarketData(
        gold_spot=params['gold_spot'],
        eurusd_spot=params['eurusd_spot'],
        r_eur=params['r_eur'],
        r_usd=params['r_usd'],
        sigma_gold=params['sigma_gold'],
        sigma_eurusd=params['sigma_eurusd'],
        rho=params['rho'],
        gold_yield=params['gold_yield'],
    )


def _build_contract(params: dict):
    """Construct a ContractTerms instance from a plain dict."""
    from pricing_model import ContractTerms
    return ContractTerms(
        notional=params.get('notional', 500_000_000),
        strike=params.get('strike', 4600.0),
        tenor=params.get('tenor', 2.0),
        barrier_lower=params.get('barrier_lower', 1.05),
        barrier_upper=params.get('barrier_upper', 1.25),
    )


@st.cache_data(ttl=300)
def fetch_live_market_data(use_live: bool) -> Tuple[dict, str]:
    """Fetch market data via MarketDataProvider, cached 5 min.

    Returns (params_dict, provenance_report_str).
    """
    from market_data import MarketDataProvider
    provider = MarketDataProvider(use_live=use_live)
    market = provider.fetch_market_data()
    report = provider.get_provenance_report()
    params = {
        'gold_spot': market.gold_spot,
        'eurusd_spot': market.eurusd_spot,
        'r_eur': market.r_eur,
        'r_usd': market.r_usd,
        'sigma_gold': market.sigma_gold,
        'sigma_eurusd': market.sigma_eurusd,
        'rho': market.rho,
        'gold_yield': market.gold_yield,
    }
    return params, report


@st.cache_data(show_spinner="Running Monte Carlo pricing...")
def run_pricing(market_dict: dict, contract_dict: dict,
                n_paths: int, n_steps: int = 504) -> dict:
    """Run MC pricing; returns result dict (without large arrays)."""
    from pricing_model import StructuredForwardPricer
    market = _build_market(market_dict)
    contract = _build_contract(contract_dict)
    pricer = StructuredForwardPricer(market, contract)
    result = pricer.price_monte_carlo(n_paths=n_paths, n_steps=n_steps, seed=42)
    # Strip large numpy arrays for cacheability
    return {k: v for k, v in result.items()
            if not isinstance(v, np.ndarray)}


@st.cache_data(show_spinner="Running sensitivity analysis...")
def run_cached_sensitivity(market_dict: dict, contract_dict: dict,
                           n_paths: int) -> list:
    """Run sensitivity sweep; returns list-of-dicts."""
    from pricing_model import run_sensitivity_analysis
    from market_data import SensitivityRangeGenerator
    market = _build_market(market_dict)
    contract = _build_contract(contract_dict)
    ranges = SensitivityRangeGenerator(market, contract)
    df = run_sensitivity_analysis(market, contract, n_paths=n_paths, ranges=ranges)
    return df.to_dict('records')


@st.cache_data(show_spinner="Running scenario analysis...")
def run_cached_scenarios(market_dict: dict, contract_dict: dict,
                         n_paths: int) -> list:
    """Run scenario analysis; returns list-of-dicts."""
    from advanced_models import run_scenario_analysis
    from market_data import SensitivityRangeGenerator
    market = _build_market(market_dict)
    contract = _build_contract(contract_dict)
    ranges = SensitivityRangeGenerator(market, contract)
    df = run_scenario_analysis(market, contract, n_paths=n_paths, ranges=ranges)
    return df.to_dict('records')


@st.cache_data(show_spinner="Running model comparison...")
def run_cached_model_comparison(market_dict: dict, contract_dict: dict,
                                n_paths: int) -> dict:
    """Run GBM / Heston / Merton comparison."""
    from advanced_models import run_model_comparison
    market = _build_market(market_dict)
    contract = _build_contract(contract_dict)
    return run_model_comparison(market, contract, n_paths=n_paths)


@st.cache_data(show_spinner="Computing Greeks...")
def run_cached_greeks(market_dict: dict, contract_dict: dict,
                      n_paths: int) -> dict:
    """Compute finite-difference Greeks."""
    from pricing_model import StructuredForwardPricer
    market = _build_market(market_dict)
    contract = _build_contract(contract_dict)
    pricer = StructuredForwardPricer(market, contract)
    base = pricer.price_monte_carlo(n_paths=n_paths, n_steps=504, seed=42)
    return pricer.compute_greeks(base, n_paths=n_paths, seed=42)
