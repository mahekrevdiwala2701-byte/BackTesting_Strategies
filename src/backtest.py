import numpy as np
import pandas as pd

def run_portfolio_weight_backtest(asset_dfs: dict[str, pd.DataFrame], weights: dict[str, float], initial_capital: float = 100000.0) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Execute a Multi-Asset Portfolio Backtest based on user asset weights.
    
    asset_dfs: dict of asset_name -> DataFrame (must have 'Date' and 'Close')
    weights: dict of asset_name -> weight_percentage (e.g., {'Gold': 25, 'LargeCap': 25, ...})
    initial_capital: starting capital in Rupees (₹)
    """
    # 1. Normalize weights
    valid_assets = [k for k in weights if k in asset_dfs and not asset_dfs[k].empty]
    if not valid_assets:
        return pd.DataFrame(), {}, pd.DataFrame()
        
    raw_weights = {k: max(float(weights[k]), 0.0) for k in valid_assets}
    total_w = sum(raw_weights.values())
    if total_w <= 0:
        norm_weights = {k: 1.0 / len(valid_assets) for k in valid_assets}
    else:
        norm_weights = {k: v / total_w for k, v in raw_weights.items()}

    # 2. Merge prices on Date
    merged = pd.DataFrame()
    for name in valid_assets:
        df_asset = asset_dfs[name][['Date', 'Close']].copy()
        df_asset = df_asset.rename(columns={'Close': f'Close_{name}'})
        if merged.empty:
            merged = df_asset
        else:
            merged = pd.merge(merged, df_asset, on='Date', how='inner')

    merged = merged.sort_values('Date').dropna().reset_index(drop=True)
    if len(merged) < 2:
        return pd.DataFrame(), {}, pd.DataFrame()

    # 3. Calculate asset daily returns & weighted portfolio returns
    portfolio_daily_return = pd.Series(0.0, index=merged.index)
    
    asset_stats = []
    for name in valid_assets:
        close_col = f'Close_{name}'
        ret_col = f'Return_{name}'
        merged[ret_col] = merged[close_col].pct_change().fillna(0.0)
        
        # Add to weighted portfolio return
        portfolio_daily_return += merged[ret_col] * norm_weights[name]
        
        # Individual asset metrics
        asset_start = merged[close_col].iloc[0]
        asset_end = merged[close_col].iloc[-1]
        asset_tot_ret = ((asset_end - asset_start) / asset_start) * 100
        
        asset_stats.append({
            'Asset': name,
            'Weight (%)': norm_weights[name] * 100,
            'Allocated Capital (₹)': initial_capital * norm_weights[name],
            'Total Return (%)': asset_tot_ret,
            'Starting Price (₹)': asset_start,
            'Latest Price (₹)': asset_end
        })

    merged['Portfolio_Return'] = portfolio_daily_return
    merged['Portfolio_Equity'] = initial_capital * (1 + merged['Portfolio_Return']).cumprod()
    
    # Drawdown calculation
    merged['Peak'] = merged['Portfolio_Equity'].cummax()
    merged['Drawdown'] = (merged['Portfolio_Equity'] - merged['Peak']) / merged['Peak']

    # 4. Summary Statistics
    final_equity = merged['Portfolio_Equity'].iloc[-1]
    total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
    
    n_days = (merged['Date'].iloc[-1] - merged['Date'].iloc[0]).days
    n_years = max(n_days / 365.25, 0.01)
    cagr = ((final_equity / initial_capital) ** (1 / n_years) - 1) * 100
    
    daily_rets = merged['Portfolio_Return'].dropna()
    ann_volatility = (daily_rets.std() * np.sqrt(252)) * 100
    sharpe_ratio = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0.0
    max_drawdown_pct = merged['Drawdown'].min() * 100

    summary = {
        "Initial Capital": initial_capital,
        "Final Portfolio Equity": final_equity,
        "Portfolio Total Return (%)": total_return_pct,
        "CAGR (%)": cagr,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown_pct,
        "Annual Volatility (%)": ann_volatility,
        "Years": n_years,
        "Total Trading Days": len(merged)
    }

    asset_breakdown_df = pd.DataFrame(asset_stats)
    return merged, summary, asset_breakdown_df


def run_sma_crossover_backtest(df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50, initial_capital: float = 100000.0) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Execute Moving Average Crossover Backtest in Rupees (₹)."""
    data = df.copy()
    
    data['Fast_SMA'] = data['Close'].rolling(window=fast_period).mean()
    data['Slow_SMA'] = data['Close'].rolling(window=slow_period).mean()
    
    data['Signal'] = 0.0
    data['Signal'] = np.where(data['Fast_SMA'] > data['Slow_SMA'], 1.0, 0.0)
    data['Position'] = data['Signal'].shift(1).fillna(0.0)
    
    data['Market_Return'] = data['Close'].pct_change().fillna(0.0)
    data['Strategy_Return'] = data['Market_Return'] * data['Position']
    
    data['Market_Equity'] = initial_capital * (1 + data['Market_Return']).cumprod()
    data['Strategy_Equity'] = initial_capital * (1 + data['Strategy_Return']).cumprod()
    
    data['Position_Change'] = data['Position'].diff()
    entries = data[data['Position_Change'] == 1]
    exits = data[data['Position_Change'] == -1]
    
    trades = []
    for entry_idx in entries.index:
        exit_candidates = exits[exits.index > entry_idx]
        if not exit_candidates.empty:
            exit_idx = exit_candidates.index[0]
            entry_row = data.loc[entry_idx]
            exit_row = data.loc[exit_idx]
            pnl_pct = (exit_row['Close'] - entry_row['Close']) / entry_row['Close']
            trades.append({
                'Entry Date': entry_row['Date'] if 'Date' in entry_row else entry_idx,
                'Entry Price (₹)': entry_row['Close'],
                'Exit Date': exit_row['Date'] if 'Date' in exit_row else exit_idx,
                'Exit Price (₹)': exit_row['Close'],
                'PnL (%)': pnl_pct * 100
            })
            
    trades_df = pd.DataFrame(trades)
    
    strat_final = data['Strategy_Equity'].iloc[-1]
    bnh_final = data['Market_Equity'].iloc[-1]
    
    summary = {
        "Initial Capital": initial_capital,
        "Final Strategy Equity": strat_final,
        "Strategy Total Return (%)": ((strat_final - initial_capital) / initial_capital) * 100,
        "Buy & Hold Return (%)": ((bnh_final - initial_capital) / initial_capital) * 100,
        "Total Trades Executed": len(trades),
        "Win Rate (%)": (len([t for t in trades if t['PnL (%)'] > 0]) / len(trades) * 100) if trades else 0.0
    }
    
    return data, summary, trades_df
