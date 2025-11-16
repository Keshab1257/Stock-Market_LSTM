
# Stock-Market_LSTM

Clean, reproducible example for forecasting stock closing prices using a Long Short-Term Memory (LSTM) neural network. This repository includes an interactive Jupyter notebook (`UPD_STOCK.ipynb`) with a step-by-step workflow and a companion script (`upd_stock.py`) for programmatic runs.

## Table of contents

- About
- Features
- Repository layout
- Quick start
- Usage
- Configuration & hyperparameters
- Outputs & evaluation
- Notes & caveats
- Contributing
- License

## About

This project demonstrates an end-to-end pipeline for time-series forecasting of stock prices using historical data from Yahoo Finance (via `yfinance`). It covers data collection, preprocessing (moving averages, scaling), sequence generation for LSTM input, model training, evaluation, and optional future forecasting.

## Features

- Download historic OHLC data using `yfinance` (Close is used by default)
- Time-series preprocessing: moving averages, NaN handling, MinMax scaling
- Convert tabular price data to supervised learning sequences (sliding windows)
- Stacked LSTM model with Dropout and Dense output for regression
- Evaluation with MAE, MSE, RMSE and visual plots of predicted vs actual prices
- Notebook-first design for reproducibility and quick experimentation

## Repository layout

- `UPD_STOCK.ipynb` — Jupyter notebook (recommended) containing the full pipeline and visualizations
- `upd_stock.py` — Script with helper functions and a programmatic entry point for training and prediction
- `README.md` — This file

Note: training artifacts (models, plots) may be generated in the repository root when you run the notebook/script.

## Quick start

Prerequisites
- Python 3.8 — 3.11 (recommended)
- Recommended packages: pandas, numpy, matplotlib, scikit-learn, yfinance, tensorflow (or keras), joblib, jupyter

Install (PowerShell):

```powershell
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib scikit-learn yfinance tensorflow joblib jupyter
```

Tip: create and activate a virtual environment before installing dependencies.

## Usage

Notebook (recommended)
1. Launch Jupyter and open the notebook:

```powershell
jupyter notebook UPD_STOCK.ipynb
```

2. Run cells in order. The notebook is organized into sections: data download, preprocessing, sequence creation, model definition/training, evaluation, and forecasting.

Script
- `upd_stock.py` contains helpers that mirror the notebook steps. You can import and call functions like `predict_stock_price()` from other scripts, or modify `upd_stock.py` to accept command-line arguments for `--ticker`, `--start`, `--end`, and `--future-days`.

Example (pseudo-CLI usage — adapt if script has no argument parser):

```powershell
python .\upd_stock.py --ticker AAPL --start 2015-01-01 --end 2024-12-31 --future-days 30
```

## Configuration & hyperparameters

- Window length (sequence size): commonly 100 days
- Train/Test split: default 80/20
- Model: stacked LSTM layers with Dropout, final Dense (regression)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Epochs, batch size and layer sizes are configurable in the notebook/script — start small and iterate.

## Outputs & evaluation

- Trained model files (example: `stock_prediction.pkl` or Keras checkpoints)
- Plots: actual vs predicted closing prices and training history
- Evaluation metrics: MAE, MSE, RMSE (printed in notebook/script)

## Notes & caveats

- Educational example: this project is intended for learning and experimentation, not production trading.
- Financial time-series forecasting is noisy; evaluate results carefully and avoid overfitting.
- Data quality and availability depend on Yahoo Finance and the `yfinance` package.
- The notebook targets Close price forecasting by default — you can extend features/targets (volume, OHLC, technical indicators) as needed.

## Contributing

- Report bugs or request features by opening an issue.
- Preferred workflow: fork → branch → PR with a clear description and, when applicable, tests or notebook examples.

## License

No license file is included in this repository. If you plan to share or redistribute this code, add an appropriate license (e.g., MIT, Apache‑2.0).

## Contact

For questions or collaboration, open an issue in this repository.


