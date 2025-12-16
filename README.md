📈 Dynamic Probabilistic LSTM for Stock Price Forecasting (NVIDIA)

A probabilistic time-series forecasting framework that uses an LSTM network to predict future NVIDIA stock prices along with dynamic confidence intervals.

Unlike standard stock prediction projects that output a single deterministic forecast, this project explicitly models forecast uncertainty using empirical residual distributions derived from rolling backtesting.

In real-world finance, understanding risk is as important as predicting price direction.
This project is built with that philosophy.

🔍 Why This Project Exists

Most stock prediction projects try to answer:

“What will the price be?”

However, practical financial forecasting asks a different question:

“What range of outcomes is plausible, and how uncertain is the forecast?”

This project addresses that gap by:

Predicting multi-step future returns

Estimating time-varying uncertainty

Producing a forecast cone instead of a single prediction line

🧠 Core Idea

The model is trained on log returns rather than raw prices and forecasts an entire future horizon in one pass.
Uncertainty is not assumed — it is learned empirically from historical prediction errors.

The result is a forecast that answers:

What is the most likely future path?

How wide is the uncertainty around it?

How does that uncertainty grow with time?

📊 Data Representation

Prices are transformed as follows:

𝑟
𝑡
=
log
⁡
(
𝑃
𝑡
)
−
log
⁡
(
𝑃
𝑡
−
1
)
r
t
	​

=log(P
t
	​

)−log(P
t−1
	​

)

Using log returns:

Stabilizes variance

Improves learning dynamics

Allows additive multi-step forecasting

Converts cleanly back to price space

🔁 Supervised Time-Series Framing

The time series is reframed as a supervised learning problem:

Input: a sliding window of past log returns

Output: a sequence of future log returns

This enables direct multi-step forecasting, avoiding recursive error accumulation.

🤖 Model Summary

LSTM-based sequence model

Predicts future return vectors

Optimized using Mean Squared Error (MSE)

The focus is intentionally not on complex architectures, but on:

Evaluation methodology

Uncertainty calibration

Financial interpretability

🧪 Rolling Backtesting

Instead of trusting training loss:

The model is evaluated across rolling historical windows

Forecast errors (residuals) are collected for each horizon step

This builds an empirical error distribution over time

This step is critical for realistic uncertainty estimation.

📉 Dynamic Uncertainty Estimation

For each forecast horizon:

Residual distributions are analyzed

Empirical quantiles (e.g., 90%) are computed

These quantiles define a time-varying confidence radius

This allows uncertainty to:

Increase with forecast horizon

Adapt to market volatility

Reflect real model performance

📈 Price-Space Forecasting

Predicted log returns are:

Cumulatively summed

Added to the last observed log price

Exponentiated back to price space

Final output:

Point forecast

Upper confidence band

Lower confidence band

This produces a probabilistic price cone, not a single brittle estimate.

✅ What Makes This Project Strong

Multi-step sequence forecasting

Residual-based uncertainty modeling

Rolling backtest evaluation

Financially sound transformations

Clear separation of prediction vs risk

This reflects real quantitative modeling practices, not toy examples.

⚠️ Limitations

Assumes historical dynamics persist

Does not incorporate macroeconomic or news signals

Single-asset forecasting (NVIDIA)

These are deliberate trade-offs for clarity and focus.
