# Stock-Price-Prediction
# Project Overview
This project analyzes historical stock market data of Apple Inc. (AAPL) and builds a quantitative model to predict next-day stock prices. Using Python, financial time-series data was collected, cleaned, and analyzed to identify market trends. The project demonstrates basic quantitative analysis, data preprocessing, and model evaluation in financial markets.
# Objective
- Analyze historical stock data
- Build a predictive model for next-day closing price
- Visualize actual vs predicted prices
- Learn fundamental quantitative analysis and modeling

## Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- yfinance

## Methodology
1. Download historical stock data using `yfinance`.
2. Create a target column (`Tomorrow_Close`) for next day's closing price.
3. Split data into training and testing sets.
4. Train a Linear Regression model.
5. Predict stock prices and visualize actual vs predicted prices.
6. Evaluate model performance using R² score.

## Sample Code

```python
import yfinance as yf
import pandas as pd

# Download Apple stock data
data = yf.download("AAPL", start="2023-01-01", end="2025-12-31")

# Create target column
data["Tomorrow_Close"] = data["Close"].shift(-1)
data = data.dropna()

# Features and target
X = data[["Close"]]
y = data["Tomorrow_Close"]

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Linear Regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
