import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
import yfinance as yf
from xgboost import XGBClassifier


# Example: Download data for a specific stock
data = yf.download('AAPL', start='2018-01-01', end='2024-11-02')
data['MA7'] = data['Close'].rolling(window=7).mean()
data['MA21'] = data['Close'].rolling(window=21).mean()
data['RSI'] = (100 - (100 / (1 + data['Close'].pct_change().rolling(window=14).mean())))
data['Volatility'] = data['Close'].rolling(window=21).std()

# Set target: 1 if price increases, 0 otherwise
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
data.dropna(inplace=True)

# Bollinger Bands
data['Upper Band'] = data['MA21'] + (data['Close'].rolling(window=21).std() * 2)
data['Lower Band'] = data['MA21'] - (data['Close'].rolling(window=21).std() * 2)

# Moving Average Convergence Divergence (MACD)
data['12_EMA'] = data['Close'].ewm(span=12, adjust=False).mean()
data['26_EMA'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['12_EMA'] - data['26_EMA']

# Stochastic Oscillator
data['L14'] = data['Low'].rolling(window=14).min()
data['H14'] = data['High'].rolling(window=14).max()
data['%K'] = 100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
data['%D'] = data['%K'].rolling(window=3).mean()

# Drop intermediate columns if needed
data.drop(['L14', 'H14'], axis=1, inplace=True)

# Split data
X = data[['MA7', 'MA21', 'RSI', 'Volatility', 'Target','Upper Band', 'Lower Band','12_EMA', '26_EMA', 'MACD', '%K', '%D']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Define parameters for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Use GridSearch with Random Forest as an example
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Re-evaluate the best model
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print("Cross-validated accuracy:", scores.mean())