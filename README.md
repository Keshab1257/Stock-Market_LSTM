
# Stock-Market_LSTM

A reproducible project for exploring stock price forecasting using a Long Short-Term Memory (LSTM) neural network. The repository contains a Jupyter notebook and a supporting Python script that demonstrate data collection, preprocessing, model training, evaluation and future price forecasting using historical stock data from Yahoo Finance (via yfinance).

## Highlights
- Uses historical OHLC (open/high/low/close) data from Yahoo Finance
- Time-series preprocessing: moving averages, scaling, and sequence windowing
- LSTM-based model for sequence-to-value prediction
- Evaluation with MAE, MSE and RMSE and visualization of predictions
- Example notebook for step-by-step reproducibility (`UPD_STOCK.ipynb`)

## Repository structure

- `UPD_STOCK.ipynb` — Jupyter notebook with the full workflow (recommended for reproduction and analysis)
- `upd_stock.py` — Script version / helper script (if present) for running the pipeline from the command line
- `README.md` — This file

(Note: additional artifacts such as trained model files may be created in the project root after training.)

## Quick summary of the workflow

1. Download historical stock data using yfinance (ticker, start date, end date).
2. Preprocess data: compute moving averages, drop missing rows, scale features using MinMaxScaler.
3. Convert data to supervised sequences (e.g. 100-day input windows) for LSTM.
4. Define and train a stacked LSTM model with dropout and a final Dense output.
5. Make predictions on test data and rescale predicted values to original price range.
6. Evaluate results (MAE, MSE, RMSE) and visualize actual vs predicted prices.
7. Optionally save the trained model for later inference.

## Requirements

Recommended Python version: 3.8 — 3.11

Install the core dependencies (example using PowerShell / Windows):

```powershell
python -m pip install --upgrade pip
python -m pip install pandas numpy yfinance matplotlib scikit-learn tensorflow keras joblib jupyter
```

Tip: consider creating a virtual environment before installing packages.

## How to run

Option A — Notebook (recommended):

1. Start Jupyter and open the notebook:

```powershell
jupyter notebook UPD_STOCK.ipynb
```

2. Run the cells in order. The notebook includes sections for data download, preprocessing, model definition/training, evaluation and future prediction.

Option B — Script (if `upd_stock.py` supports CLI):

Run the script from PowerShell. If the script accepts command-line arguments, an example could be:

```powershell
python .\upd_stock.py --ticker AAPL --start 2015-01-01 --end 2024-12-31
```

If `upd_stock.py` does not accept arguments, open the file and update the parameters directly or run it as a module.

## Typical configuration / hyperparameters

- Window (sequence length): 100 days (common default in the notebook)
- Train/test split: 80/20 (configurable)
- LSTM layers: stacked LSTM with Dropout (see notebook for exact architecture)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam

Adjust these values in the notebook or script to perform experiments.

## Outputs

- Trained model file (if saved) e.g. `stock_prediction.pkl` or TensorFlow/Keras checkpoint files
- Prediction plots (matplotlib figures)
- Numeric evaluation metrics printed to the notebook/session

## Notes, assumptions & caveats

- This project demonstrates a basic LSTM approach for educational purposes. Financial markets are noisy and this model is not production-grade trading software.
- Predictions are for closing prices only (unless you change the target in the notebook).
- Model performance heavily depends on the chosen ticker, timeframe, and hyperparameters.
- The code uses Yahoo Finance via `yfinance`; availability and quality of data depend on that service.

## Contributing

Contributions are welcome. Good first steps:

- Open an issue describing the feature or bug
- Submit a pull request with a clear description and tests/examples when applicable

Please include reproducible steps and any data or configuration used for experiments.

## Acknowledgements

- yfinance — for convenient access to historical stock data
- scikit-learn, pandas, NumPy, matplotlib, TensorFlow / Keras for model and tooling

## Contact

For questions or collaboration, open an issue or contact the repository owner.

## License

No license file is included in this repository. The original README stated the project is intended for educational and research use. If you plan to reuse or redistribute this code, consider asking the repository owner to add a formal license (for example, MIT or Apache-2.0).
