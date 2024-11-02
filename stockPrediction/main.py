import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


stock_data = yf.download('MSFT', start='2020-01-01', end='2024-11-02')
stock_data['Short_MA'] = stock_data['Close'].rolling(window=20).mean()  # 20-day moving average
stock_data['Long_MA'] = stock_data['Close'].rolling(window=50).mean()  # 50-day moving average


stock_data['Signal'] = 0
stock_data['Signal'][20:] = np.where(stock_data['Short_MA'][20:] > stock_data['Long_MA'][20:], 1, 0)  # Buy signal
stock_data['Position'] = stock_data['Signal'].diff()  # Capture changes in signals

plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Stock Price')
plt.plot(stock_data['Short_MA'], label='Short MA (20 days)')
plt.plot(stock_data['Long_MA'], label='Long MA (50 days)')

# Plot Buy signals
plt.plot(stock_data[stock_data['Position'] == 1].index,
         stock_data['Short_MA'][stock_data['Position'] == 1],
         '^', markersize=10, color='g', label='Buy Signal')

# Plot Sell signals
plt.plot(stock_data[stock_data['Position'] == -1].index,
         stock_data['Short_MA'][stock_data['Position'] == -1],
         'v', markersize=10, color='r', label='Sell Signal')

plt.title('Stock Buy and Sell Signals')
plt.legend()
plt.show()
