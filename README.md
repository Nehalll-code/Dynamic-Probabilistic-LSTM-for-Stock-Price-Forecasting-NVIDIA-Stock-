Dynamic Probabilistic LSTM for Stock Forecasting

This project implements a probabilistic time-series forecasting framework using an LSTM network to predict future stock prices along with dynamic confidence intervals.
Rather than producing a single deterministic forecast, the model estimates forecast uncertainty using empirical residual distributions obtained through rolling backtesting.

The goal is to move closer to real-world financial forecasting, where understanding risk and uncertainty is as important as the prediction itself.

Why This Project Exists

Most stock prediction projects focus on answering:

“What will the price be?”

In practice, finance asks:

“What range of outcomes is plausible, and how uncertain is the forecast?”

This project addresses that gap by:

Predicting multi-step future returns

Estimating time-varying uncertainty

Producing a forecast cone instead of a single line

High-Level Approach

The workflow follows a realistic quantitative modeling pipeline:

Transform raw prices into log returns to stabilize variance

Train an LSTM to predict future return sequences

Perform rolling backtests to observe model errors over time

Use empirical quantiles of residuals to estimate uncertainty

Convert predictions back into price-space confidence bands

This avoids strong distributional assumptions and lets the data define uncertainty.

Data Representation

Stock prices are first converted into log prices, and then differenced to obtain log returns:

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

Log returns are preferred because they:

Are closer to stationary

Allow additive multi-step forecasting

Map cleanly back to price space via exponentiation

Supervised Sequence Framing

Time series forecasting is framed as a supervised learning problem:

Input: a fixed window of past log returns

Output: a sequence of future log returns

This enables the model to predict an entire future horizon in one forward pass, rather than step-by-step recursion.

Model Overview

A standard LSTM is used to capture temporal dependencies in financial returns:

Input: sliding windows of past returns

Output: multi-step future return predictions

Loss function: Mean Squared Error (MSE)

The emphasis is not on architectural complexity, but on how predictions are evaluated and calibrated.

Rolling Backtesting

Instead of relying on training loss, the model is evaluated using rolling windows:

The model generates forecasts across multiple historical windows

Prediction errors (residuals) are recorded at each forecast horizon

This produces an empirical distribution of errors for every future step

This step is crucial for realistic uncertainty estimation.

Dynamic Uncertainty Estimation

For each forecast horizon:

Residual distributions are analyzed

Quantiles (e.g., 90%) are computed empirically

These values define a time-varying confidence radius

Unlike fixed-variance models, uncertainty:

Expands with forecast horizon

Adapts to market volatility

Reflects model performance over time

Price-Space Forecasting

Predicted log returns are:

Cumulatively summed

Added to the last observed log price

Converted back to prices via exponentiation

This yields:

A point forecast

Upper and lower confidence bounds

A visually interpretable price cone
