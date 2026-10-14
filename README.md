# 📈 Quantitative Backtesting & Streamlit Analytics Lab

A modular, production-ready Streamlit application for quantitative financial data analysis, technical indicators calculation, moving average crossover backtesting, and multi-asset portfolio performance comparison.

---

## 🚀 Quick Start Guide

### 1. Activate the Virtual Environment
On Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Windows (Command Prompt):
```cmd
\.venv\Scripts\activate.bat
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```
*The app will automatically launch in your default web browser at `http://localhost:8501`.*

---

## 📁 Project Structure

```
backtesting/
│
├── app.py                      # Main Streamlit Dashboard Application
├── requirements.txt            # Python Package Dependencies
├── .gitignore                  # Git Ignore File
├── .streamlit/
│   └── config.toml             # Custom Dark Theme & Server Configuration
│
├── src/                        # Core Python Modules
│   ├── __init__.py
│   ├── data_loader.py          # Data Loading, Caching & Processing
│   ├── analytics.py           # Technical Indicators & Metric Calculations
│   ├── backtest.py            # Quantitative Backtesting Engines
│   └── ui.py                  # Custom Glassmorphism CSS & UI Components
│
├── gold_historical_data.csv    # Sample Historical Asset Dataset
├── largeCap_data.csv           # Sample Large Cap Index Dataset
├── midcap_data.csv             # Sample Mid Cap Index Dataset
└── smallcap_data.csv           # Sample Small Cap Index Dataset
```

---

## ✨ Features

- 📈 **Interactive Technical Charts**: Candlestick/Line overlays, Moving Averages (SMA/EMA), Bollinger Bands, RSI, and Drawdown visualizer built with Plotly.
- ⚡ **SMA Crossover Strategy Backtester**: Test custom fast/slow SMA combinations against Buy & Hold benchmarks with equity curves and trade logs.
- 📊 **Multi-Asset Performance Suite**: Compare normalized returns (rebased to 100) and analyze daily return correlations across assets.
- 📤 **Custom Data Loader & Inspector**: Supports built-in workspace datasets or uploading custom CSV files (`Date`, `Close`). Includes CSV data export.
- 🎨 **Modern Dark UI**: Powered by custom Glassmorphism CSS styling.

---

## 🛠️ Adding New Datasets or Strategies

- **Add New Datasets**: Drop any CSV file ending in `_data.csv` or `.csv` into the project root directory containing `Date` and `Close` columns. It will automatically populate in the dataset selector!
- **Add New Strategies**: Extend `src/backtest.py` with custom logic (e.g. RSI Mean Reversion, MACD Crossover, MACD Momentum) and reference it in `app.py`.
