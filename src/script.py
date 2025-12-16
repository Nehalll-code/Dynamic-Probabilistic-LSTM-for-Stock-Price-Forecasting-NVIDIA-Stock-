# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# seeds
np.random.seed(42)
tf.random.set_seed(42)

# forecast parameters
LAGS = 30
HORIZON = 30

print("GPU:", tf.config.list_physical_devices('GPU'))


# %%
import yfinance as yf
df = yf.download("NVDA", start="2010-01-01", end="2024-12-31")
df.head()


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = df[['Close']].rename(columns={'Close':'my_close'}).dropna()
df.index = pd.to_datetime(df.index)
#print(df.head())



# %%
print(df.columns)
print(df.head())


# %%
df['my_close'].plot(title='NVDA Close Price (sample)', figsize=(10,4))
plt.show()

# %%
df['logprice'] = np.log(df['my_close'])
df['logreturn'] = df['logprice'].diff() #log returns
df = df.dropna()
print(df[['my_close','logprice','logreturn']].head())
df[['logprice','logreturn']].plot(subplots=True, figsize=(10,6), title='Log Price and Log Returns')
plt.title('NVDA Log Price and Log Returns (sample)')
plt.show()

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
series = df['logreturn'].values.reshape(-1,1)
series_scaled = scaler.fit_transform(series)
print(series_scaled[:5])

# %%
def create_supervised(x, lags, horizon):
    X, Y = [], []
    n = len(x)
    for i in range(lags, n - horizon + 1):
        X.append(x[i-lags:i])
        Y.append(x[i:i+horizon].flatten())
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, lags, 1)
    return X, Y

LAGS, HORIZON = 30, 30
X, Y = create_supervised(series_scaled.flatten(), LAGS, HORIZON)
print("X.shape, Y.shape:", X.shape, Y.shape)



# %%
split = int(0.8 * len(X))
X_bt, Y_bt = X[:split], Y[:split]


# %%
#Backtest iteration
#test residual extraction logic for one window before scaling up.
# pick cut (index in supervised samples)
t_cut = split  # e.g., use previous split
X_bt = X[:t_cut]
Y_bt = Y[:t_cut]

# train small model quickly
bt_model = build_model(LAGS, HORIZON, units=8)
bt_model.fit(X_bt, Y_bt, epochs=5, batch_size=64, verbose=0)

# forecast from sample index forecast_idx
forecast_idx = t_cut - 1  # adjust carefully so you can access Y[forecast_idx]
X_input = X[forecast_idx].reshape(1,LAGS,1)
y_true = Y[forecast_idx]
y_pred_scaled = bt_model.predict(X_input).flatten()
# inverse scale both
y_true_orig = scaler.inverse_transform(y_true.reshape(-1,1)).flatten()
y_pred_orig = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
resid = y_true_orig - y_pred_orig
print("resid shape", resid.shape)

# %%
#Expand to rolling backtest loop (collect residuals matrix)
#this produces residuals per horizon across many windows — core for dynamic intervals.

residuals_matrix = []

for i in range(len(X_bt)):
    X_input = X_bt[i].reshape(1, LAGS, 1)
    y_true = Y_bt[i]
    y_pred_scaled = bt_model.predict(X_input, verbose=0).flatten()
    y_true_orig = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    resid = y_true_orig - y_pred_orig
    residuals_matrix.append(resid)

residuals_matrix = np.array(residuals_matrix)
print("residuals_matrix shape:", residuals_matrix.shape)


# %%
# --- Step 9: quantiles per horizon ---
# residuals_matrix: shape (n_windows, HORIZON)
abs_resid = np.abs(residuals_matrix)       # same shape (n_windows, HORIZON)
alpha = 0.10                               # 90% prediction intervals
q_per_horizon = np.quantile(abs_resid, 1 - alpha, axis=0)  # shape (HORIZON,)

print("q_per_horizon:", q_per_horizon)
print("is non-decreasing-ish (diff >= -tiny_tol)?",
      np.all(np.diff(q_per_horizon) >= -1e-12))  # small negative tolerances OK

# Quick plot: quantile vs horizon
plt.figure(figsize=(6,3.5))
plt.plot(np.arange(1, len(q_per_horizon)+1), q_per_horizon, marker='o')
plt.xlabel("Horizon (steps)")
plt.ylabel(f"{100*(1-alpha):.0f}% abs-residual quantile")
plt.title("Error quantile vs horizon (q_per_horizon)")
plt.grid(alpha=0.3)
plt.show()


# %%
final_model = build_model(LAGS, HORIZON, units=32)
final_model.fit(X, Y, epochs=30, batch_size=32, verbose=1)


# %%
x_last = X[-1].reshape(1,LAGS,1)
pred_scaled = final_model.predict(x_last).flatten()
pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()


# %%
last_log = df['logprice'].iloc[-1]

log_point = last_log + np.cumsum(pred)
log_upper = last_log + np.cumsum(pred + q_per_horizon)
log_lower = last_log + np.cumsum(pred - q_per_horizon)

price_point = np.exp(log_point)
price_upper = np.exp(log_upper)
price_lower = np.exp(log_lower)


# %%
plt.figure(figsize=(10,5))
plt.plot(price_point, label="Forecast")
plt.fill_between(range(len(price_point)), price_lower, price_upper, alpha=0.3)
plt.title("NVDA Dynamic Probabilistic Forecast")
plt.legend()
plt.show()



