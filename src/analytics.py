import numpy as np
import pandas as pd

def compute_technical_indicators(df: pd.DataFrame, sma_fast: int = 50, sma_slow: int = 200, rsi_period: int = 14) -> pd.DataFrame:
    """Compute moving averages, RSI, and Bollinger Bands on Close prices."""
    data = df.copy()
    
    # Simple Moving Averages
    data[f'SMA_{sma_fast}'] = data['Close'].rolling(window=sma_fast).mean()
    data[f'SMA_{sma_slow}'] = data['Close'].rolling(window=sma_slow).mean()
    
    # Exponential Moving Averages
    data[f'EMA_{sma_fast}'] = data['Close'].ewm(span=sma_fast, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = rolling_mean + (rolling_std * 2)
    data['BB_Lower'] = rolling_mean - (rolling_std * 2)
    
    # Drawdown
    data['Peak'] = data['Close'].cummax()
    data['Drawdown'] = (data['Close'] - data['Peak']) / data['Peak']
    
    return data

def calculate_performance_metrics(df: pd.DataFrame, risk_free_rate: float = 0.05) -> dict:
    """Compute key quantitative metrics: CAGR, Sharpe Ratio, Max Drawdown, Volatility, Sortino Ratio."""
    if df.empty or 'Close' not in df.columns or len(df) < 2:
        return {}
    
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    
    # Days calculation
    if 'Date' in df.columns and pd.notnull(df['Date'].iloc[0]) and pd.notnull(df['Date'].iloc[-1]):
        n_days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
        n_years = max(n_days / 365.25, 0.01)
    else:
        n_years = max(len(df) / 252, 0.01)
    
    # Total & Annualized Return (CAGR)
    total_return = (end_price - start_price) / start_price
    cagr = (end_price / start_price) ** (1 / n_years) - 1 if start_price > 0 else 0
    
    # Daily returns statistics
    daily_returns = df['Daily_Return'].dropna() if 'Daily_Return' in df.columns else df['Close'].pct_change().dropna()
    
    ann_volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
    
    # Sharpe Ratio
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = (excess_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    
    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (excess_returns.mean() * 252 / downside_std) if downside_std > 0 else 0
    
    # Maximum Drawdown
    peak = df['Close'].cummax()
    drawdowns = (df['Close'] - peak) / peak
    max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
    
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": ann_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Total Trading Days": len(df),
        "Years": n_years
    }
