import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import datetime
from scipy.optimize import minimize
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import json
import os
import bcrypt

# ============================================================
# PERSISTENT STORAGE ROOT (FLY.IO)
# ============================================================

DB_DIR = "/data"
os.makedirs(DB_DIR, exist_ok=True)

SQLITE_PATH = os.path.join(DB_DIR, "app.db")

USER_DATA_DIR = os.path.join(DB_DIR, "user_data")
os.makedirs(USER_DATA_DIR, exist_ok=True)
   
# ============================================================
# USER DATA PERSISTENCE (DISK-BACKED)
# ============================================================

DEFAULT_PREFS = {
    "start_date": "1900-01-01",
    "risk_on_tickers": "TQQQ",
    "risk_on_weights": "1.0",
    "risk_off_tickers": "AGG",
    "risk_off_weights": "1.0",
    "annual_drag_pct": 0.0,
    "qs_cap_1": 10000,
    "real_cap_1": 10000,
    "end_date": "",
    "official_inception_date": "2025-12-22",
    "benchmark_ticker": "QQQ",
    "min_holding_days": 1,
}

def _user_file(username):
    return os.path.join(USER_DATA_DIR, f"{username}.json")

def load_user_prefs(username):
    path = _user_file(username)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return DEFAULT_PREFS.copy()

def save_user_prefs(username, prefs):
    with open(_user_file(username), "w") as f:
        json.dump(prefs, f, indent=2)

if "prefs" not in st.session_state:
    st.session_state.prefs = load_user_prefs(st.session_state.username)
   
# ============================================================
# TEMPORARY FIXED USER (AUTH DISABLED)
# ============================================================

st.session_state.username = "test_user"
st.session_state.name = "Test User"

# ============================================================
# YOUR STRATEGY FUNCTIONS
# ============================================================

def show_strategy_overview():
    st.markdown("""
## **Sigma System Overview**

An automated program that integrates the SIG System with the 200 Day Simple Moving Average Strategy. 

---

### **The SIG System:**

**Quarterly Target Growth Rate (QTGR):**  
Quarterly Growth Target (QGT): Quarterly growth rate derived from the historical returns of the user selected Risk On allocation (e.g., 9sig = 9% QGT for TQQQ).

- Is your Risk On Allocation above or below its quarterly growth target? If above, it's a sell signal. If below, it's a buy signal.
- If the signal is a Risk On Allocation sell, you will move proceeds of the sale to your Risk Off Allocation in the following order of events: sell an amount of the Risk On Allocation, use that amount to buy more of the Risk Off Allocation. If the signal is a Risk On Allocation buy, you will generate buying power by selling a portion of your Risk Off Allocation, then use the proceeds to buy more of your Risk On Allocation, in the following order of events: sell a portion of the Risk Off Allocation, use the proceeds to buy the Risk On Allocation.
---

### **The 200 Day Simple Moving Average Strategy (SMA or MA):**

A 200 Day SMA is constructed using a simulated index of the user selected Risk On allocation.

- If the 200 Day SMA < Risk On Allocation Index, then buy & hold the Risk On Allocation.
- If the 200 Day SMA > Risk On Allocation Index, then sell & hold the Risk Off Allocation.

A "Risk On Regime" = 200 Day SMA < Risk On Allocation Index.  
A "Risk Off Regime" = 200 Day SMA > Risk On Allocation Index.

---

### **Sigma System**

- If the 200 Day SMA < Risk On Allocation Index, then the model runs the SIG System as instructed above.
- If the 200 Day SMA > Risk On Allocation Index, then the model allocates all portfolio capital to the Risk Off Allocation.
- When model flips from "Risk Off" to "Risk On", the model refers to the Allocation Tables and resumes the current SIG System weights.
""")

# CONFIG
DEFAULT_START_DATE = "1900-01-01"
RISK_FREE_RATE = 0.0
RISK_ON_WEIGHTS = {"TQQQ": 1.0}
RISK_OFF_WEIGHTS = {"AGG": 1.0}
FLIP_COST = 0.0000
START_RISKY = 0.6
START_SAFE  = 0.4

@st.cache_data(show_spinner=True)
def load_price_data(tickers, start_date, end_date=None):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if "Adj Close" in data.columns:
        px = data["Adj Close"].copy()
        if "Close" in data.columns:
            px = px.combine_first(data["Close"])
    else:
        px = data["Close"].copy()

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    return px.dropna(how="all")

def build_portfolio_index(prices, weights_dict, annual_drag_pct=0.0):
    simple_rets = prices.pct_change().fillna(0)
    idx_rets = pd.Series(0.0, index=simple_rets.index)

    for ticker, weight in weights_dict.items():
        if ticker in simple_rets.columns:
            idx_rets += simple_rets[ticker] * weight
    
    if annual_drag_pct > 0:
        daily_drag_factor = (1 - annual_drag_pct) ** (1/252)
        idx_rets = (1 + idx_rets) * daily_drag_factor - 1
    
    cumprod = (1 + idx_rets).cumprod()
    
    valid_mask = cumprod.notna() & (cumprod > 0)
    if not valid_mask.any():
        return pd.Series(1.0, index=cumprod.index)
    
    first_valid_idx = cumprod[valid_mask].index[0]
    
    cumprod_filled = cumprod.copy()
    cumprod_filled.loc[:first_valid_idx] = 1.0
    cumprod_filled = cumprod_filled.ffill()
    
    return cumprod_filled

def compute_ma(price_series, length, ma_type):
    if ma_type.lower() == "ema":
        ma = price_series.ewm(span=length, adjust=False).mean()
    else:
        ma = price_series.rolling(window=length, min_periods=1).mean()
    
    return ma.shift(1)

def generate_testfol_signal_vectorized(price, ma, tol_series, min_holding_days=5):
    px = price.values
    ma_vals = ma.values
    n = len(px)
    
    if np.all(np.isnan(ma_vals)):
        return pd.Series(False, index=ma.index)
    
    tol_vals = tol_series.values
    upper = ma_vals * (1 + tol_vals)
    lower = ma_vals * (1 - tol_vals)
    
    sig = np.zeros(n, dtype=bool)
    
    non_nan_mask = ~np.isnan(ma_vals)
    if not np.any(non_nan_mask):
        return pd.Series(False, index=ma.index)
    
    first_valid = np.where(non_nan_mask)[0][0]
    if first_valid == 0:
        first_valid = 1
    start_index = first_valid + 1
    
    if start_index >= n:
        return pd.Series(False, index=ma.index)
    
    days_since_last_change = 0
    last_change_idx = start_index
    
    for t in range(start_index, n):
        if np.isnan(px[t]) or np.isnan(upper[t]) or np.isnan(lower[t]):
            sig[t] = sig[t-1] if t > 0 else False
        elif t - last_change_idx < min_holding_days:
            sig[t] = sig[t-1]
        elif not sig[t - 1]:
            if px[t] > upper[t]:
                sig[t] = True
                last_change_idx = t
                days_since_last_change = 0
            else:
                sig[t] = False
        else:
            if px[t] < lower[t]:
                sig[t] = False
                last_change_idx = t
                days_since_last_change = 0
            else:
                sig[t] = True
        
        if t > last_change_idx:
            days_since_last_change += 1
    
    return pd.Series(sig, index=ma.index).fillna(False)

def run_sig_engine(
    risk_on_returns,
    risk_off_returns,
    target_quarter,
    ma_signal,
    pure_sig_rw=None,
    pure_sig_sw=None,
    flip_cost=FLIP_COST,
    quarter_end_dates=None,
    quarterly_multiplier=4.0,
    ma_flip_multiplier=4.0
):

    dates = risk_on_returns.index
    n = len(dates)

    if quarter_end_dates is None:
        raise ValueError("quarter_end_dates must be supplied")

    quarter_end_set = set(quarter_end_dates)
    sig_arr = ma_signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    eq = 10000.0
    risky_val = eq * START_RISKY
    safe_val  = eq * START_SAFE

    frozen_risky = None
    frozen_safe  = None

    equity_curve = []
    risky_w_series = []
    safe_w_series = []
    risky_val_series = []
    safe_val_series = []
    rebalance_events = 0
    rebalance_dates = []

    for i in range(n):
        date = dates[i]
        r_on = risk_on_returns.iloc[i]
        r_off = risk_off_returns.iloc[i]
        ma_on = bool(ma_signal.iloc[i])
        
        if i > 0 and flip_mask.iloc[i]:
            eq *= (1 - flip_cost * ma_flip_multiplier)

        if ma_on:
            if frozen_risky is not None:
                w_r = pure_sig_rw.iloc[i]
                w_s = pure_sig_sw.iloc[i]
                risky_val = eq * w_r
                safe_val  = eq * w_s
                frozen_risky = None
                frozen_safe  = None

            risky_val *= (1 + r_on)
            safe_val  *= (1 + r_off)

            if date in quarter_end_set:
                prev_qs = [qd for qd in quarter_end_dates if qd < date]

                if prev_qs:
                    prev_q = prev_qs[-1]
                    idx_prev = dates.get_loc(prev_q)
                    risky_at_qstart = risky_val_series[idx_prev]
                    goal_risky = risky_at_qstart * (1 + target_quarter)

                    if risky_val > goal_risky:
                        excess = risky_val - goal_risky
                        risky_val -= excess
                        safe_val  += excess
                        rebalance_dates.append(date)

                    elif risky_val < goal_risky:
                        needed = goal_risky - risky_val
                        move = min(needed, safe_val)
                        safe_val -= move
                        risky_val += move
                        rebalance_dates.append(date)

                    eq *= (1 - flip_cost * quarterly_multiplier)

            eq = risky_val + safe_val
            risky_w = risky_val / eq
            safe_w  = safe_val  / eq

        else:
            if frozen_risky is None:
                frozen_risky = risky_val
                frozen_safe  = safe_val

            eq *= (1 + r_off)
            risky_w = 0.0
            safe_w  = 1.0

        equity_curve.append(eq)
        risky_w_series.append(risky_w)
        safe_w_series.append(safe_w)
        risky_val_series.append(risky_val)
        safe_val_series.append(safe_val)

    return (
        pd.Series(equity_curve, index=dates),
        pd.Series(risky_w_series, index=dates),
        pd.Series(safe_w_series, index=dates),
        rebalance_dates
    )

def build_weight_df(prices, signal, risk_on_weights, risk_off_weights):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for a, w in risk_on_weights.items():
        if a in prices.columns:
            weights.loc[signal, a] = w

    for a, w in risk_off_weights.items():
        if a in prices.columns:
            weights.loc[~signal, a] = w

    return weights

def compute_enhanced_performance(simple_returns, eq_curve, rf=0.0):
    if len(eq_curve) == 0 or eq_curve.iloc[0] == 0:
        return {
            "CAGR": 0, "Volatility": 0, "Sharpe": 0, "MaxDrawdown": 0,
            "TotalReturn": 0, "DD_Series": pd.Series([], dtype=float),
            "Calmar": 0, "Sortino": 0, "Omega": 0, "Skewness": 0,
            "Kurtosis": 0, "VaR_95": 0, "CVaR_95": 0, "WinRate": 0,
            "ProfitFactor": 0, "RecoveryFactor": 0, "UlcerIndex": 0, "TailRatio": 0
        }
    
    n_days = len(eq_curve)
    n_years = n_days / 252
    cagr = (eq_curve.iloc[-1] / eq_curve.iloc[0]) ** (1 / n_years) - 1
    vol = simple_returns.std() * np.sqrt(252) if len(simple_returns) > 0 else 0
    sharpe = (simple_returns.mean() * 252 - rf) / vol if vol > 0 else 0
    dd = eq_curve / eq_curve.cummax() - 1
    max_dd = dd.min() if len(dd) > 0 else 0
    
    downside_returns = simple_returns[simple_returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (simple_returns.mean() * 252 - rf) / downside_dev if downside_dev > 0 else 0
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    threshold = 0.0
    gains = simple_returns[simple_returns > threshold].sum()
    losses = abs(simple_returns[simple_returns < threshold].sum())
    omega = gains / losses if losses > 0 else 0
    
    positive_rets = simple_returns[simple_returns > 0]
    negative_rets = simple_returns[simple_returns < 0]
    win_rate = len(positive_rets) / len(simple_returns) if len(simple_returns) > 0 else 0
    gross_profit = positive_rets.sum()
    gross_loss = abs(negative_rets.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    var_95 = np.percentile(simple_returns, 5) * np.sqrt(252)
    cvar_95 = simple_returns[simple_returns <= np.percentile(simple_returns, 5)].mean() * np.sqrt(252) if len(simple_returns) > 0 else 0
    
    skewness = simple_returns.skew() if len(simple_returns) > 0 else 0
    kurtosis = simple_returns.kurt() if len(simple_returns) > 0 else 0
    
    ulcer_index = np.sqrt((dd ** 2).mean()) if len(dd) > 0 else 0
    
    recovery_factor = -cagr / max_dd if max_dd != 0 else 0
    
    top_5 = np.percentile(simple_returns, 95)
    bottom_5 = np.percentile(simple_returns, 5)
    tail_ratio = abs(top_5 / bottom_5) if bottom_5 != 0 else 0
    
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Omega": omega,
        "MaxDrawdown": max_dd,
        "TotalReturn": eq_curve.iloc[-1] / eq_curve.iloc[0] - 1 if eq_curve.iloc[0] != 0 else 0,
        "WinRate": win_rate,
        "ProfitFactor": profit_factor,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "UlcerIndex": ulcer_index,
        "RecoveryFactor": recovery_factor,
        "TailRatio": tail_ratio,
        "DD_Series": dd
    }

def backtest(prices, signal, risk_on_weights, risk_off_weights, flip_cost, ma_flip_multiplier=3.0, annual_drag_pct=0.0):
    simple = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)

    strategy_simple = (weights.shift(1).fillna(0) * simple).sum(axis=1)
    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    flip_costs = np.where(flip_mask, -flip_cost * ma_flip_multiplier, 0.0)
    
    if annual_drag_pct > 0:
        daily_drag_factor = (1 - annual_drag_pct) ** (1/252)
        strategy_simple = (1 + strategy_simple) * daily_drag_factor - 1
    
    strat_adj = strategy_simple + flip_costs

    eq = (1 + strat_adj).cumprod()

    return {
        "returns": strat_adj,
        "equity_curve": eq,
        "signal": signal,
        "weights": weights,
        "performance": compute_enhanced_performance(strat_adj, eq),
        "flip_mask": flip_mask,
    }

def compute_quarter_progress(risky_start, risky_today, quarterly_target):
    target_risky = risky_start * (1 + quarterly_target)
    gap = target_risky - risky_today
    pct_gap = gap / risky_start if risky_start > 0 else 0

    return {
        "Risk On Capital at Last Rebalance ($)": risky_start,
        "Risk On Capital Today ($)": risky_today,
        "Risk On Capital Target Next Rebalance ($)": target_risky,
        "Gap ($)": gap,
        "Gap (%)": pct_gap,
    }

def normalize(eq):
    if len(eq) == 0 or eq.iloc[0] == 0:
        return eq
    return eq / eq.iloc[0] * 10000

def plot_diagnostics(hybrid_eq, bh_eq, hybrid_signal):
    hybrid_eq = hybrid_eq / hybrid_eq.iloc[0]
    bh_eq     = bh_eq / bh_eq.iloc[0]

    hybrid_ret = hybrid_eq.pct_change().fillna(0)
    bh_ret = bh_eq.pct_change().fillna(0)

    hybrid_dd = hybrid_eq / hybrid_eq.cummax() - 1
    bh_dd = bh_eq / bh_eq.cummax() - 1

    window = 252
    roll_sharpe_h = hybrid_ret.rolling(window).mean() / hybrid_ret.rolling(window).std() * np.sqrt(252)
    roll_sharpe_b = bh_ret.rolling(window).mean() / bh_ret.rolling(window).std() * np.sqrt(252)

    hybrid_m = hybrid_ret.resample("M").apply(lambda x: (1 + x).prod() - 1)
    bh_m = bh_ret.resample("M").apply(lambda x: (1 + x).prod() - 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    ax1.plot(hybrid_eq, label="Sigma", linewidth=2, color="green")
    ax1.plot(bh_eq, label="Buy & Hold", linewidth=2, alpha=0.7)

    in_off = False
    start = None
    for date, on in hybrid_signal.items():
        if not on and not in_off:
            start = date
            in_off = True
        elif on and in_off:
            ax1.axvspan(start, date, color="red", alpha=0.15)
            in_off = False
    if in_off:
        ax1.axvspan(start, hybrid_signal.index[-1], color="red", alpha=0.15)

    ax1.set_title("Cumulative Returns with Regime Shading")
    ax1.set_ylabel("Growth of $1")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(hybrid_dd * 100, label="Sigma", linewidth=1.5, color="green")
    ax2.plot(bh_dd * 100, label="Buy & Hold", linewidth=1.5, alpha=0.7)
    ax2.set_title("Drawdown Comparison (%)")
    ax2.set_ylabel("Drawdown %")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3.plot(roll_sharpe_h, label="Sigma", linewidth=1.5, color="green")
    ax3.plot(roll_sharpe_b, label="Buy & Hold", linewidth=1.5, alpha=0.7)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_title("Rolling 252-Day Sharpe Ratio")
    ax3.legend()
    ax3.grid(alpha=0.3)

    bins = np.linspace(
        min(hybrid_m.min(), bh_m.min()),
        max(hybrid_m.max(), bh_m.max()),
        20
    )

    ax4.hist(hybrid_m, bins=bins, alpha=0.7, density=True, label="Sigma")
    ax4.hist(bh_m, bins=bins, alpha=0.5, density=True, label="Buy & Hold")
    ax4.axvline(0, color="black", linestyle="--", linewidth=1)
    ax4.set_title("Monthly Returns Distribution")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    return fig

def monte_carlo_strategy_analysis(strategy_returns, strategy_equity, n_sim=10000, periods=252, initial_capital=None):
    if len(strategy_returns) < 100:
        return None
    
    mu_daily = strategy_returns.mean()
    sigma_daily = strategy_returns.std()
    
    if initial_capital is not None:
        initial_price = initial_capital
    else:
        initial_price = strategy_equity.iloc[-1] if len(strategy_equity) > 0 else 10000
    
    np.random.seed(42)
    sim_returns = np.random.normal(mu_daily, sigma_daily, (n_sim, periods))
    
    sim_values = initial_price * np.cumprod(1 + sim_returns, axis=1)
    
    terminal_values = sim_values[:, -1]
    terminal_returns = (terminal_values / initial_price) - 1
    
    percentiles = np.percentile(terminal_returns, list(range(5, 100, 5)))
    
    def calculate_cvar(returns, confidence):
        threshold = np.percentile(returns, 100 - confidence)
        bad_returns = returns[returns <= threshold]
        return -np.mean(bad_returns) if len(bad_returns) > 0 else 0
    
    cvar_90 = calculate_cvar(terminal_returns, 90)
    cvar_95 = calculate_cvar(terminal_returns, 95)
    cvar_99 = calculate_cvar(terminal_returns, 99)
    
    expected_return = np.mean(terminal_returns)
    expected_vol = np.std(terminal_returns)
    prob_positive = np.mean(terminal_returns > 0)
    
    terminal_value_percentiles = np.percentile(terminal_values, [5, 25, 50, 75, 95])
    
    return {
        'sim_prices': sim_values,
        'terminal_values': terminal_values,
        'terminal_returns': terminal_returns,
        'percentiles': percentiles,
        'terminal_value_percentiles': terminal_value_percentiles,
        'cvar_90': cvar_90,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'expected_return': expected_return,
        'expected_vol': expected_vol,
        'prob_positive': prob_positive,
        'var_95': np.percentile(terminal_returns, 5),
        'var_99': np.percentile(terminal_returns, 1),
        'initial_price': initial_price
    }

def plot_monte_carlo_results(results_dict, strategy_names):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    
    ax = axes[0, 0]
    for i, (name, results) in enumerate(results_dict.items()):
        if results is not None:
            returns_pct = results['terminal_returns'] * 100
            ax.hist(returns_pct, bins=50, alpha=0.5, 
                   label=name, density=True, color=colors[i])
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('12-Month Return Distributions')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    percentile_levels = list(range(5, 100, 5))
    for i, (name, results) in enumerate(results_dict.items()):
        if results is not None:
            percentiles_pct = results['percentiles'] * 100
            ax.plot(percentile_levels, percentiles_pct, 
                   marker='o', label=name, color=colors[i])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Percentile Return Ranges (5%-95%)')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('12-Month Return (%)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 2]
    cvar_data = []
    labels = []
    for name, results in results_dict.items():
        if results is not None:
            cvar_data.append([results['cvar_95'] * 100, results['cvar_99'] * 100])
            labels.append(name)
    
    if cvar_data:
        cvar_data = np.array(cvar_data)
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, cvar_data[:, 0], width, label='CVaR 95%', alpha=0.7)
        ax.bar(x + width/2, cvar_data[:, 1], width, label='CVaR 99%', alpha=0.7)
        ax.set_title('Conditional Value at Risk (CVaR)')
        ax.set_ylabel('CVaR (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    for i, (name, results) in enumerate(results_dict.items()):
        if results is not None:
            for j in range(min(20, results['sim_prices'].shape[0])):
                ax.plot(results['sim_prices'][j, :], alpha=0.1, color=colors[i])
    ax.set_title('Sample Portfolio Paths ($)')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    for i, (name, results) in enumerate(results_dict.items()):
        if results is not None:
            ax.scatter(results['expected_vol'], results['expected_return'] * 100, 
                      s=100, label=name, color=colors[i], alpha=0.7)
            ax.text(results['expected_vol']*1.01, results['expected_return']*100*1.01, 
                   name, fontsize=9, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_title('Expected Risk-Return Profile (Annualized)')
    ax.set_xlabel('Expected Volatility (Annualized)')
    ax.set_ylabel('Expected Return (% Annualized)')
    ax.grid(alpha=0.3)
    
    ax = axes[1, 2]
    prob_data = []
    prob_labels = []
    for name, results in results_dict.items():
        if results is not None:
            prob_data.append(results['prob_positive'] * 100)
            prob_labels.append(name)
    
    if prob_data:
        bars = ax.bar(range(len(prob_data)), prob_data, color=colors[:len(prob_data)])
        ax.set_title('Probability of Positive 12-Month Return')
        ax.set_ylabel('Probability (%)')
        ax.set_xticks(range(len(prob_data)))
        ax.set_xticklabels(prob_labels, rotation=45, ha='right')
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        for bar, val in zip(bars, prob_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(page_title="Portfolio MA Regime Strategy", layout="wide")
    
    # Get user info from session state
    name = st.session_state.get('name', 'User')
    username = st.session_state.get('username', 'user')
    
    # User is authenticated at this point
    st.sidebar.title(f"Welcome {name}!")
    
    
    show_strategy_overview()
    st.markdown("---")
    
    # --- SIDEBAR INPUTS ---
    st.sidebar.header("Strategy Settings")
    
    # Load saved prefs
    prefs = st.session_state.prefs
    
    # Input fields with saved values
    start = st.sidebar.text_input("Start Date", prefs["start_date"])
    risk_on_tickers_str = st.sidebar.text_input("Risk On Tickers", prefs["risk_on_tickers"])
    risk_on_weights_str = st.sidebar.text_input("Risk On Weights", prefs["risk_on_weights"])
    risk_off_tickers_str = st.sidebar.text_input("Risk Off Tickers", prefs["risk_off_tickers"])
    risk_off_weights_str = st.sidebar.text_input("Risk Off Weights", prefs["risk_off_weights"])
    annual_drag = st.sidebar.number_input("Annual Drag %", value=float(prefs["annual_drag_pct"]))
    qs_cap_1 = st.sidebar.number_input("Portfolio Value at Last Rebalance", value=float(prefs["qs_cap_1"]))
    real_cap_1 = st.sidebar.number_input("Portfolio Value Today", value=float(prefs["real_cap_1"]))
    inception_date = st.sidebar.text_input("Inception Date", prefs["official_inception_date"])
    benchmark = st.sidebar.text_input("Benchmark", prefs["benchmark_ticker"])
    min_days = st.sidebar.number_input("Confirmation Days", value=int(prefs["min_holding_days"]))
    
    # Save button
    if st.sidebar.button("💾 Save Settings", type="primary"):
        st.session_state.prefs = {
            "start_date": start,
            "risk_on_tickers": risk_on_tickers_str,
            "risk_on_weights": risk_on_weights_str,
            "risk_off_tickers": risk_off_tickers_str,
            "risk_off_weights": risk_off_weights_str,
            "annual_drag_pct": annual_drag,
            "qs_cap_1": qs_cap_1,
            "real_cap_1": real_cap_1,
            "end_date": "",
            "official_inception_date": inception_date,
            "benchmark_ticker": benchmark,
            "min_holding_days": min_days,
        }
        save_user_prefs(username, st.session_state.prefs)
        st.sidebar.success("Settings saved permanently")
    
    run_clicked = st.sidebar.button("🚀 Run Analysis", type="secondary")
    
    if not run_clicked:
        st.info("Adjust settings in sidebar and click 'Run Analysis'")
        st.stop()
    
    # Process inputs
    risk_on_tickers = [t.strip().upper() for t in risk_on_tickers_str.split(",")]
    risk_on_weights_list = [float(x) for x in risk_on_weights_str.split(",")]
    risk_on_weights = dict(zip(risk_on_tickers, risk_on_weights_list))
    
    risk_off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",")]
    risk_off_weights_list = [float(x) for x in risk_off_weights_str.split(",")]
    risk_off_weights = dict(zip(risk_off_tickers, risk_off_weights_list))
    
    annual_drag_decimal = annual_drag / 100.0
    
    all_tickers = sorted(set(risk_on_tickers + risk_off_tickers))
    end_val = ""
    prices = load_price_data(all_tickers, start, end_val if end_val else None).dropna(how="any")
    
    if len(prices) == 0:
        st.error("No data loaded. Please check your ticker symbols and date range.")
        st.stop()
    
    st.info(f"Loaded {len(prices)} trading days of data from {prices.index[0].date()} to {prices.index[-1].date()} for backtesting")
    
    # MA is always 200-day SMA
    best_len = 200
    best_type = "sma"
    
    portfolio_index = build_portfolio_index(prices, risk_on_weights, annual_drag_pct=annual_drag_decimal)
    opt_ma = compute_ma(portfolio_index, best_len, best_type)
    
    tolerance_decimal = 0.0
    tol_series = pd.Series(tolerance_decimal, index=portfolio_index.index)
    
    if annual_drag > 0:
        daily_drag_factor = (1 - annual_drag_decimal) ** (1/252)
        daily_drag_pct = (1 - daily_drag_factor) * 100
        st.write(f"**Portfolio Drag:** {annual_drag:.1f}% annual (≈{daily_drag_pct:.4f}% daily)")
    
    sig = generate_testfol_signal_vectorized(
        portfolio_index,
        opt_ma,
        tol_series,
        min_holding_days=min_days
    )
    
    best_result = backtest(prices, sig, risk_on_weights, risk_off_weights, FLIP_COST, 
                          ma_flip_multiplier=3.0, annual_drag_pct=annual_drag_decimal)
    
    latest_signal = sig.iloc[-1]
    current_regime = "Risk On" if latest_signal else "Risk Off"
    st.subheader(f"Current 200 Day SMA Regime: {current_regime}")
    
    perf = best_result["performance"]
    switches = sig.astype(int).diff().abs().sum()
    trades_per_year = switches / (len(sig) / 252) if len(sig) > 0 else 0
    
    simple_rets = prices.pct_change().fillna(0)
    risk_on_simple = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_on_weights.items():
        if a in simple_rets.columns:
            risk_on_simple += simple_rets[a] * w
    
    if annual_drag_decimal > 0:
        daily_drag_factor = (1 - annual_drag_decimal) ** (1/252)
        risk_on_simple = (1 + risk_on_simple) * daily_drag_factor - 1
    
    risk_on_eq = (1 + risk_on_simple).cumprod()
    risk_on_perf = compute_enhanced_performance(risk_on_simple, risk_on_eq)
    
    # Calendar quarter logic
    dates = prices.index
    true_q_ends = pd.date_range(start=dates.min(), end=dates.max(), freq='Q')
    mapped_q_ends = []
    for qd in true_q_ends:
        valid_dates = dates[dates <= qd]
        if len(valid_dates) > 0:
            mapped_q_ends.append(valid_dates.max())
    
    mapped_q_ends = pd.to_datetime(mapped_q_ends)
    
    today_date = pd.Timestamp.today().normalize()
    true_next_q = pd.date_range(start=today_date, periods=2, freq="Q")[0]
    next_q_end = true_next_q
    days_to_next_q = (next_q_end - today_date).days
    
    if len(risk_on_eq) > 0 and risk_on_eq.iloc[0] != 0:
        bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
        quarterly_target = (1 + bh_cagr) ** (1/4) - 1
    else:
        bh_cagr = 0
        quarterly_target = 0
    
    risk_off_daily = pd.Series(0.0, index=simple_rets.index)
    for a, w in risk_off_weights.items():
        if a in simple_rets.columns:
            risk_off_daily += simple_rets[a] * w
    
    pure_sig_signal = pd.Series(True, index=risk_on_simple.index)
    
    pure_sig_eq, pure_sig_rw, pure_sig_sw, pure_sig_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        pure_sig_signal,
        quarter_end_dates=mapped_q_ends,
        quarterly_multiplier=2.0,
        ma_flip_multiplier=0.0
    )
    
    hybrid_eq, hybrid_rw, hybrid_sw, hybrid_rebals = run_sig_engine(
        risk_on_simple,
        risk_off_daily,
        quarterly_target,
        sig,
        pure_sig_rw=pure_sig_rw,
        pure_sig_sw=pure_sig_sw,
        quarter_end_dates=mapped_q_ends,
        quarterly_multiplier=2.0,
        ma_flip_multiplier=3.0
    )
    
    hybrid_simple = hybrid_eq.pct_change().fillna(0)
    hybrid_perf = compute_enhanced_performance(hybrid_simple, hybrid_eq)
    
    # Since-inception analysis
    inception = pd.to_datetime(inception_date)
    sigma_eq_si = hybrid_eq.loc[hybrid_eq.index >= inception]
    sigma_ret_si = sigma_eq_si.pct_change().fillna(0)
    
    bh_eq_si = risk_on_eq.loc[risk_on_eq.index >= inception]
    bh_ret_si = bh_eq_si.pct_change().fillna(0)
    
    benchmark_px = load_price_data([benchmark], inception)
    if not benchmark_px.empty and benchmark in benchmark_px.columns:
        benchmark_eq_si = (benchmark_px[benchmark] / benchmark_px[benchmark].iloc[0]).reindex(sigma_eq_si.index).ffill()
        benchmark_ret_si = benchmark_eq_si.pct_change().fillna(0)
    else:
        benchmark_eq_si = pd.Series(1.0, index=sigma_eq_si.index)
        benchmark_ret_si = pd.Series(0.0, index=sigma_eq_si.index)
        st.warning(f"Could not load benchmark data for {benchmark}")
    
    st.subheader(f"Your Performance (Sigma vs Buy & Hold vs {benchmark})")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sigma_eq_si / sigma_eq_si.iloc[0], label="Sigma", linewidth=2, color="blue")
    ax.plot(bh_eq_si / bh_eq_si.iloc[0], label="Buy & Hold", linewidth=2, alpha=0.7)
    ax.plot(benchmark_eq_si, label=benchmark, linewidth=2, linestyle="--", color="black", alpha=0.7)
    ax.set_ylabel("Growth of $1")
    ax.set_title(f"Performance Since {inception_date}")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    sigma_perf_si = compute_enhanced_performance(sigma_ret_si, sigma_eq_si)
    bh_perf_si = compute_enhanced_performance(bh_ret_si, bh_eq_si)
    benchmark_perf_si = compute_enhanced_performance(benchmark_ret_si, benchmark_eq_si)
    
    def fmt(val, kind):
        if pd.isna(val):
            return "—"
        if kind == "pct":
            return f"{val:.2%}"
        return f"{val:.3f}"
    
    rows = [
        ("CAGR", "CAGR", "pct"),
        ("Volatility", "Volatility", "pct"),
        ("Sharpe", "Sharpe", "dec"),
        ("Sortino", "Sortino", "dec"),
        ("Max Drawdown", "MaxDrawdown", "pct"),
        ("Total Return", "TotalReturn", "pct"),
    ]
    
    table_data = []
    for label, key, kind in rows:
        table_data.append([
            label,
            fmt(sigma_perf_si[key], kind),
            fmt(bh_perf_si[key], kind),
            fmt(benchmark_perf_si[key], kind),
        ])
    
    si_table = pd.DataFrame(table_data, columns=["Metric", "Sigma", "Buy & Hold", benchmark])
    st.dataframe(si_table, use_container_width=True)
    
    perf = hybrid_perf
    
    if len(hybrid_rebals) > 0:
        reb_df = pd.DataFrame({"Rebalance Date": pd.to_datetime(hybrid_rebals)})
        st.subheader("Sigma System – Historical Rebalance Dates")
        st.dataframe(reb_df)
    else:
        st.subheader("Sigma – Historical Rebalance Dates")
        st.write("No Sigma rebalances occurred during the backtest.")
    
    if len(hybrid_rebals) > 0:
        quarter_start_date = hybrid_rebals[-1]
    else:
        quarter_start_date = dates[0] if len(dates) > 0 else None
    
    st.subheader("Strategy Implementation Summary")
    if len(hybrid_rebals) > 0:
        last_reb = hybrid_rebals[-1]
        st.write(f"**Last Rebalance:** {last_reb.strftime('%Y-%m-%d')}")
    else:
        st.write("**Quarter start (last SIG rebalance):** None yet")
    st.write(f"**Next Rebalance:** {next_q_end.date()} ({days_to_next_q} days)")
    
    def get_sig_progress(qs_cap, today_cap):
        if quarter_start_date is not None and len(hybrid_rw) > 0:
            risky_start = qs_cap * float(hybrid_rw.loc[quarter_start_date])
            risky_today = today_cap * float(hybrid_rw.iloc[-1])
            return compute_quarter_progress(risky_start, risky_today, quarterly_target)
        else:
            return compute_quarter_progress(0, 0, 0)
    
    prog_1 = get_sig_progress(qs_cap_1, real_cap_1)
    st.write(f"**Quarterly Target Growth Rate:** {quarterly_target:.2%}")
    
    prog_df = pd.DataFrame.from_dict(prog_1, orient='index', columns=['Portfolio'])
    prog_df.loc["Gap (%)"] = prog_df.loc["Gap (%)"].apply(lambda x: f"{x:.2%}")
    st.dataframe(prog_df)
    
    st.markdown("### Rebalance Recommendations")
    gap = prog_1['Gap ($)']
    date_str = next_q_end.strftime("%m/%d/%Y")
    days_str = f"{days_to_next_q} days"
    dollar_amount = f"${abs(gap):,.2f}"
    
    if gap > 0:
        message = f"**Portfolio:** **Sell {dollar_amount}** of the Risk Off Allocation and **Buy {dollar_amount}** of the Risk On Allocation on **{date_str}** ({days_str})"
    elif gap < 0:
        message = f"**Portfolio:** **Sell {dollar_amount}** of the Risk On Allocation and **Buy {dollar_amount}** of Risk Off Allocation on **{date_str}** ({days_str})"
    else:
        message = f"**Portfolio:** No rebalance needed until **{date_str}** ({days_str})"
    
    st.markdown(message)
    
    # Performance metrics
    def time_in_drawdown(dd): return (dd < 0).mean() if len(dd) > 0 else 0
    def mar(c, dd): return c / abs(dd) if dd != 0 else 0
    def ulcer(dd): return np.sqrt((dd**2).mean()) if len(dd) > 0 and (dd**2).mean() != 0 else 0
    def pain_gain(c, dd): return c / ulcer(dd) if ulcer(dd) != 0 else 0
    
    def compute_stats(perf, returns, dd, flips, tpy):
        return {
            "CAGR": perf["CAGR"],
            "Volatility": perf["Volatility"],
            "Sharpe": perf["Sharpe"],
            "Sortino": perf["Sortino"],
            "Calmar": perf["Calmar"],
            "Omega": perf["Omega"],
            "MaxDD": perf["MaxDrawdown"],
            "Total": perf["TotalReturn"],
            "WinRate": perf["WinRate"],
            "ProfitFactor": perf["ProfitFactor"],
            "VaR_95": perf["VaR_95"],
            "CVaR_95": perf["CVaR_95"],
            "MAR": mar(perf["CAGR"], perf["MaxDrawdown"]),
            "TID": time_in_drawdown(dd),
            "PainGain": pain_gain(perf["CAGR"], dd),
            "UlcerIndex": perf["UlcerIndex"],
            "RecoveryFactor": perf["RecoveryFactor"],
            "TailRatio": perf["TailRatio"],
            "Skew": perf["Skewness"],
            "Kurtosis": perf["Kurtosis"],
            "Trades/year": tpy,
        }
    
    hybrid_simple = hybrid_eq.pct_change().fillna(0) if len(hybrid_eq) > 0 else pd.Series([], dtype=float)
    hybrid_perf = compute_enhanced_performance(hybrid_simple, hybrid_eq)
    
    pure_sig_simple = pure_sig_eq.pct_change().fillna(0) if len(pure_sig_eq) > 0 else pd.Series([], dtype=float)
    pure_sig_perf = compute_enhanced_performance(pure_sig_simple, pure_sig_eq)
    
    ma_perf = best_result["performance"]
    
    strat_stats = compute_stats(ma_perf, best_result["returns"], ma_perf["DD_Series"], best_result["flip_mask"], trades_per_year)
    risk_stats = compute_stats(risk_on_perf, risk_on_simple, risk_on_perf["DD_Series"], np.zeros(len(risk_on_simple), dtype=bool) if len(risk_on_simple) > 0 else np.array([], dtype=bool), 0)
    hybrid_stats = compute_stats(hybrid_perf, hybrid_simple, hybrid_perf["DD_Series"], np.zeros(len(hybrid_simple), dtype=bool) if len(hybrid_simple) > 0 else np.array([], dtype=bool), 0)
    pure_sig_stats = compute_stats(pure_sig_perf, pure_sig_simple, pure_sig_perf["DD_Series"], np.zeros(len(pure_sig_simple), dtype=bool) if len(pure_sig_simple) > 0 else np.array([], dtype=bool), 0)
    
    st.subheader("Backtest: All Strategies Performance Statistics")
    rows = [
        ("CAGR", "CAGR"),
        ("Volatility", "Volatility"),
        ("Sharpe Ratio", "Sharpe"),
        ("Sortino Ratio", "Sortino"),
        ("Calmar Ratio", "Calmar"),
        ("Omega Ratio", "Omega"),
        ("Max Drawdown", "MaxDD"),
        ("Total Return", "Total"),
        ("Win Rate", "WinRate"),
        ("Profit Factor", "ProfitFactor"),
        ("VaR (95%)", "VaR_95"),
        ("CVaR (95%)", "CVaR_95"),
        ("MAR Ratio", "MAR"),
        ("Time in Drawdown (%)", "TID"),
        ("Pain-to-Gain", "PainGain"),
        ("Ulcer Index", "UlcerIndex"),
        ("Recovery Factor", "RecoveryFactor"),
        ("Tail Ratio", "TailRatio"),
        ("Skewness", "Skew"),
        ("Kurtosis", "Kurtosis"),
        ("Trades per year", "Trades/year"),
    ]
    
    def fmt_pct(x): return f"{x:.2%}" if pd.notna(x) else "—"
    def fmt_dec(x): return f"{x:.3f}" if pd.notna(x) else "—"
    def fmt_num(x): return f"{x:,.2f}" if pd.notna(x) else "—"
    
    table_data = []
    for label, key in rows:
        sv = strat_stats.get(key, np.nan)
        rv = risk_stats.get(key, np.nan)
        hv = hybrid_stats.get(key, np.nan)
        ps = pure_sig_stats.get(key, np.nan)
        
        if key in ["CAGR", "Volatility", "MaxDD", "Total", "WinRate", "VaR_95", "CVaR_95", "TID"]:
            row = [label, fmt_pct(sv), fmt_pct(rv), fmt_pct(hv), fmt_pct(ps)]
        elif key in ["Sharpe", "Sortino", "Calmar", "Omega", "ProfitFactor", "Skew", "Kurtosis", "UlcerIndex", "RecoveryFactor", "TailRatio", "PainGain", "MAR"]:
            row = [label, fmt_dec(sv), fmt_dec(rv), fmt_dec(hv), fmt_dec(ps)]
        else:
            row = [label, fmt_num(sv), fmt_num(rv), fmt_num(hv), fmt_num(ps)]
        
        table_data.append(row)
    
    stat_table = pd.DataFrame(table_data, columns=["Metric", "MA", "Buy & Hold", "Sigma", "SIG"])
    st.dataframe(stat_table, use_container_width=True)
    
    # Allocation tables
    def compute_allocations(account_value, risky_w, safe_w, ron_w, roff_w):
        risky_dollars = account_value * risky_w
        safe_dollars = account_value * safe_w
        alloc = {"Total Risky $": risky_dollars, "Total Safe $": safe_dollars}
        for t, w in ron_w.items():
            alloc[t] = risky_dollars * w
        for t, w in roff_w.items():
            alloc[t] = safe_dollars * w
        return alloc
    
    def add_pct(df_dict):
        out = pd.DataFrame.from_dict(df_dict, orient="index", columns=["$"])
        if "Total Risky $" in out.index and "Total Safe $" in out.index:
            total_portfolio = float(out.loc["Total Risky $","$"]) + float(out.loc["Total Safe $","$"])
            out["% Portfolio"] = (out["$"] / total_portfolio * 100).apply(lambda x: f"{x:.2f}%")
            return out
        total = out["$"].sum()
        out["% Portfolio"] = (out["$"] / total * 100).apply(lambda x: f"{x:.2f}%")
        return out
    
    st.subheader("Portfolio Allocations")
    hyb_r = float(hybrid_rw.iloc[-1]) if len(hybrid_rw) > 0 else 0
    hyb_s = float(hybrid_sw.iloc[-1]) if len(hybrid_sw) > 0 else 0
    pure_r = float(pure_sig_rw.iloc[-1]) if len(pure_sig_rw) > 0 else 0
    pure_s = float(pure_sig_sw.iloc[-1]) if len(pure_sig_sw) > 0 else 0
    latest_signal = sig.iloc[-1] if len(sig) > 0 else False
    
    tab1, tab2, tab3 = st.tabs(["Sigma", "SIG", "200 Day SMA"])
    with tab1:
        st.write("### Portfolio — Sigma")
        st.dataframe(add_pct(compute_allocations(real_cap_1, hyb_r, hyb_s, risk_on_weights, risk_off_weights)))
    with tab2:
        st.write("### Portfolio — SIG")
        st.dataframe(add_pct(compute_allocations(real_cap_1, pure_r, pure_s, risk_on_weights, risk_off_weights)))
    with tab3:
        st.write("### Portfolio — 200 Day SMA")
        if latest_signal:
            ma_alloc = compute_allocations(real_cap_1, 1.0, 0.0, risk_on_weights, {"SHY": 0})
        else:
            ma_alloc = compute_allocations(real_cap_1, 0.0, 1.0, {}, risk_off_weights)
        st.dataframe(add_pct(ma_alloc))
    
    # MA distance
    st.subheader("Next 200 Day SMA Crossover Distance")
    if len(opt_ma) > 0 and len(portfolio_index) > 0:
        latest_date = opt_ma.dropna().index[-1]
        P = float(portfolio_index.loc[latest_date])
        MA = float(opt_ma.loc[latest_date])
        upper = MA * (1 + tolerance_decimal)
        lower = MA * (1 - tolerance_decimal)
        
        if latest_signal:
            delta = (P - lower) / P
            st.write(f"**Drop Required for Crossover:** {delta:.2%}")
        else:
            delta = (upper - P) / P
            st.write(f"**Gain Required for Crossover:** {delta:.2%}")
    else:
        st.write("**Insufficient data for MA distance calculation**")
    
    # Regime stats
    st.subheader("200 Day SMA Crossover Statistics")
    if len(sig) > 0:
        sig_int = sig.astype(int)
        flips = sig_int.diff().fillna(0).ne(0)
        segments = []
        current = sig_int.iloc[0]
        seg_start = sig_int.index[0]
        
        for date, sw in flips.iloc[1:].items():
            if sw:
                segments.append((current, seg_start, date))
                current = sig_int.loc[date]
                seg_start = date
        
        segments.append((current, seg_start, sig_int.index[-1]))
        regime_rows = []
        for r, s, e in segments:
            regime_rows.append(["Risk On" if r == 1 else "Risk Off", s.date(), e.date(), (e - s).days])
        
        regime_df = pd.DataFrame(regime_rows, columns=["Regime", "Start", "End", "Duration (days)"])
        st.dataframe(regime_df)
        
        on_durations = regime_df[regime_df['Regime']=='Risk On']['Duration (days)']
        off_durations = regime_df[regime_df['Regime']=='Risk Off']['Duration (days)']
        st.write(f"**Average Risk On duration:** {on_durations.mean():.1f} days" if len(on_durations) > 0 else "**Avg Risk On duration:** 0 days")
        st.write(f"**Average Risk Off duration:** {off_durations.mean():.1f} days" if len(off_durations) > 0 else "**Avg Risk Off duration:** 0 days")
    else:
        st.write("No 200 Day SMA Crossover data available")
    
    st.markdown("---")
    
    # Final performance plot
    st.subheader("All Strategy Performance Visual")
    plot_index = build_portfolio_index(prices, risk_on_weights, annual_drag_pct=annual_drag_decimal)
    plot_ma = compute_ma(plot_index, best_len, best_type)
    plot_index_norm = normalize(plot_index)
    plot_ma_norm = normalize(plot_ma.dropna()) if len(plot_ma.dropna()) > 0 else pd.Series([], dtype=float)
    strat_eq_norm = normalize(best_result["equity_curve"])
    hybrid_eq_norm = normalize(hybrid_eq) if len(hybrid_eq) > 0 else pd.Series([], dtype=float)
    pure_sig_norm = normalize(pure_sig_eq) if len(pure_sig_eq) > 0 else pd.Series([], dtype=float)
    risk_on_norm = normalize(risk_on_eq) if len(risk_on_eq) > 0 else pd.Series([], dtype=float)
    
    if len(strat_eq_norm) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(strat_eq_norm, label="MA", linewidth=2)
        if len(risk_on_norm) > 0:
            ax.plot(risk_on_norm, label="Buy & Hold", alpha=0.65)
        if len(hybrid_eq_norm) > 0:
            ax.plot(hybrid_eq_norm, label="Sigma", linewidth=2, color="blue")
        if len(pure_sig_norm) > 0:
            ax.plot(pure_sig_norm, label="SIG", linewidth=2, color="orange")
        if len(plot_ma_norm) > 0:
            ax.plot(plot_ma_norm, label="200-day SMA", linestyle="--", color="black", alpha=0.6)
        
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Insufficient data for performance plot")
    
    # Strategy diagnostics
    st.subheader("Sigma System Diagnostics")
    diag_fig = plot_diagnostics(hybrid_eq=hybrid_eq, bh_eq=risk_on_eq, hybrid_signal=sig)
    st.pyplot(diag_fig)
    
    # Monte Carlo
    st.subheader("Monte Carlo Stress Testing - Next 12-Month Simulation")
    total_current_portfolio = real_cap_1
    strategies_mc = {
        "MA Strategy": {"returns": best_result["returns"], "equity": best_result["equity_curve"], "initial_capital": total_current_portfolio},
        "Buy & Hold": {"returns": risk_on_simple, "equity": risk_on_eq, "initial_capital": total_current_portfolio},
        "Sigma": {"returns": hybrid_simple, "equity": hybrid_eq, "initial_capital": total_current_portfolio},
        "SIG": {"returns": pure_sig_simple, "equity": pure_sig_eq, "initial_capital": total_current_portfolio}
    }
    
    mc_results = {}
    with st.spinner("Running Monte Carlo simulations (100,000 paths each)..."):
        for name, data in strategies_mc.items():
            if len(data["returns"]) > 100:
                mc_results[name] = monte_carlo_strategy_analysis(
                    data["returns"], data["equity"], n_sim=100000, periods=252, initial_capital=data["initial_capital"]
                )
            else:
                mc_results[name] = None
    
    st.write(f"**Current Total Portfolio Value:** ${total_current_portfolio:,.2f}")
    st.write(f"**Monte Carlo Simulation Horizon:** 12 months (252 trading days)")
    st.write(f"**Number of Simulations:** 100,000 per strategy")
    
    if any(v is not None for v in mc_results.values()):
        mc_fig = plot_monte_carlo_results(mc_results, list(strategies_mc.keys()))
        st.pyplot(mc_fig)
        
        st.subheader("Next 12-Month Simulated-Value")
        terminal_value_data = []
        for name, results in mc_results.items():
            if results is not None:
                terminal_value_data.append({
                    "Strategy": name,
                    "Current Value": f"${results['initial_price']:,.2f}",
                    "Expected Value": f"${np.mean(results['terminal_values']):,.2f}",
                    "5th %ile (Worst 5%)": f"${results['terminal_value_percentiles'][0]:,.2f}",
                    "25th %ile": f"${results['terminal_value_percentiles'][1]:,.2f}",
                    "Median": f"${results['terminal_value_percentiles'][2]:,.2f}",
                    "75th %ile": f"${results['terminal_value_percentiles'][3]:,.2f}",
                    "95th %ile (Best 5%)": f"${results['terminal_value_percentiles'][4]:,.2f}",
                })
        
        if terminal_value_data:
            terminal_value_df = pd.DataFrame(terminal_value_data)
            st.dataframe(terminal_value_df, use_container_width=True)
        
        st.subheader("Next 12-Month Simulated-Returns(%)")
        return_data = []
        for name, results in mc_results.items():
            if results is not None:
                return_data.append({
                    "Strategy": name,
                    "Expected Return": f"{results['expected_return']:.1%}",
                    "5th %ile": f"{np.percentile(results['terminal_returns'], 5):.1%}",
                    "25th %ile": f"{np.percentile(results['terminal_returns'], 25):.1%}",
                    "Median": f"{np.percentile(results['terminal_returns'], 50):.1%}",
                    "75th %ile": f"{np.percentile(results['terminal_returns'], 75):.1%}",
                    "95th %ile": f"{np.percentile(results['terminal_returns'], 95):.1%}",
                    "Prob > 0%": f"{results['prob_positive']:.1%}",
                    "CVaR 95%": f"{results['cvar_95']:.1%}"
                })
        
        if return_data:
            return_df = pd.DataFrame(return_data)
            st.dataframe(return_df, use_container_width=True)
            
            st.subheader("Key Insights from Monte Carlo")
            col1, col2, col3 = st.columns(3)
            valid_results = [(name, r) for name, r in mc_results.items() if r is not None]
            
            if valid_results:
                with col1:
                    safest = min(valid_results, key=lambda x: x[1]['cvar_95'])
                    st.metric("Most Conservative", safest[0])
                with col2:
                    highest_return = max(valid_results, key=lambda x: x[1]['expected_return'])
                    st.metric("Highest Expected Return", highest_return[0])
                with col3:
                    highest_prob = max(valid_results, key=lambda x: x[1]['prob_positive'])
                    st.metric("Highest Probability of Positive Return", highest_prob[0])
                    
                    st.write("#### Worst-Case Scenario Analysis (12-Month Horizon)")
                    worst_cases = []
                    for name, results in valid_results:
                        worst_5 = np.percentile(results['terminal_returns'], 5)
                        worst_1 = np.percentile(results['terminal_returns'], 1)
                        worst_cases.append({
                            "Strategy": name,
                            "5th Percentile (Bad Year)": worst_5,
                            "1st Percentile (Very Bad Year)": worst_1,
                            "Average in Worst 5% (CVaR 95%)": -results['cvar_95'],
                            "Average in Worst 1% (CVaR 99%)": -results['cvar_99']
                        })
                    
                    worst_case_df = pd.DataFrame(worst_cases)
                    st.dataframe(worst_case_df.style.format({
                        "5th Percentile (Bad Year)": "{:.2%}",
                        "1st Percentile (Very Bad Year)": "{:.2%}",
                        "Average in Worst 5% (CVaR 95%)": "{:.2%}",
                        "Average in Worst 1% (CVaR 99%)": "{:.2%}"
                    }), use_container_width=True)

# ============================================================
# LAUNCH APP
# ============================================================

if __name__ == "__main__":
    main()