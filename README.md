📈 Stock Price Prediction using LSTM

This project predicts the future stock prices using an LSTM (Long Short-Term Memory) deep learning model.
It fetches historical stock data using yfinance, performs preprocessing such as moving averages and scaling, trains an LSTM model, evaluates it, and generates future predictions.

🚀 Features

Fetches historical stock data (Open, Close, etc.) from Yahoo Finance

Computes Moving Averages (100, 200, 400 days)

Splits data into Training and Testing sets (80–20 split)

Scales values using MinMaxScaler

Builds and trains a 4-layer LSTM model

Predicts prices for test data

Rescales predictions back to the original price

Calculates error metrics:

MAE – Mean Absolute Error

MSE – Mean Squared Error

RMSE – Root Mean Squared Error

Saves the trained model using joblib

Predicts future prices based on previous 100 days

Visualizes data and predictions using Matplotlib

📦 Requirements

Install all dependencies:

pip install pandas numpy yfinance matplotlib scikit-learn keras tensorflow joblib

📁 Project Structure
UPD_STOCK.ipynb      # Jupyter notebook containing full LSTM workflow
stock_prediction.pkl # Saved trained LSTM model (generated after training)
README.md            # Documentation

🔍 Workflow Summary
1. Data Collection

Stock data is downloaded using:

data = yf.download(stock, start, end)

2. Preprocessing

Reset index

Compute 100, 200, and 400-day moving averages

Drop null values

Split into train/test sets

Apply MinMaxScaler

Create sequences of 100 previous days for LSTM input

3. LSTM Model Architecture

4 stacked LSTM layers

Dropout (0.2) to prevent overfitting

Dense layer for prediction

Optimizer: Adam

Loss: Mean Squared Error (MSE)

4. Model Training
model.fit(x, y, epochs=50, batch_size=32, verbose=1)

5. Predictions & Rescaling

Predicted values are rescaled back to original price:

scale = 1 / scaler.scale_
y_predict = y_predict * scale
y = y * scale

6. Visualization

Plots actual vs predicted stock prices.

📉 Evaluation Metrics

The notebook calculates:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

Example:

print("MAE:", mean_absolute_error(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))
print("RMSE:", mean_squared_error(y, y_pred, squared=False))

🔮 Future Prediction

The last 100 days of data are used to generate future stock price predictions step-by-step.

💾 Saving the Model
joblib.dump(model, "stock_prediction.pkl")

▶️ How to Run

Open the notebook:

jupyter notebook UPD_STOCK.ipynb


Run all cells step-by-step.

The model will be trained and saved.

Use the last section of the notebook to generate future predictions.

📝 Notes

LSTM requires sequence data, hence the 100-day window.

Accuracy depends on the selected stock and time range.

This model predicts closing prices only.

📜 License

This project is for educational and research use.
