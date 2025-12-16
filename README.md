# 📈 Dynamic Probabilistic LSTM for Stock Price Forecasting  
### *Uncertainty-Aware Multi-Step Forecasting for NVIDIA (NVDA)*

> **A quantitative time-series forecasting framework that models not just price direction, but forecast uncertainty — producing dynamic confidence cones instead of brittle point estimates.**

---

## 🚀 Overview

This project implements a **probabilistic deep learning framework** for forecasting NVIDIA stock prices using an LSTM trained on **log returns**, paired with **empirical uncertainty estimation** derived from rolling historical backtests.

Unlike most stock prediction projects that output a single deterministic price path, this system explicitly answers a more realistic financial question:

> **“What range of outcomes is plausible, and how uncertain is the forecast?”**

The result is a **time-varying probabilistic forecast** that reflects real model error behavior across different horizons.

---

## 🎯 Why This Project Exists

Most retail and academic stock-prediction projects focus on:

> *“What will the price be?”*

However, in real quantitative finance, that question is incomplete.

Professionals care about:

- Forecast **ranges**, not just point estimates  
- How **uncertainty grows with time**
- Whether risk estimates are **empirically calibrated**
- How models behave **out-of-sample**, not just in training

This project is designed with that philosophy.

---

## 🧠 Core Concept

The model predicts **future log returns** over a fixed horizon in a **single forward pass**.

Uncertainty is **not assumed** (e.g., Gaussian noise).  
Instead, it is **learned empirically** from historical prediction errors using **rolling backtesting**.

This enables:

- Horizon-specific uncertainty estimation  
- Volatility-adaptive confidence bands  
- Realistic probabilistic forecasting

---

## 📊 Data Representation

Instead of raw prices, the model operates on **log returns**:

\[
r_t = \log(P_t) - \log(P_{t-1})
\]

### Why log returns?

- Stabilizes variance  
- Improves learning dynamics  
- Allows **additive multi-step forecasting**
- Converts cleanly back to price space
- Aligns with standard financial modeling practices

---

## 🔁 Supervised Time-Series Framing

The time series is reframed as a supervised learning problem:

- **Input:** Sliding window of past log returns  
- **Output:** Vector of future log returns over a fixed horizon  

This allows **direct multi-step forecasting**, avoiding the error accumulation inherent in recursive one-step models.

---

## 🤖 Model Architecture

- LSTM-based sequence model
- Outputs a **future return vector**, not a single value
- Optimized using **Mean Squared Error (MSE)**

The emphasis is **not** on architectural complexity, but on:

- Evaluation methodology  
- Uncertainty calibration  
- Financial interpretability  

This mirrors real-world quantitative workflows.

---

## 🧪 Rolling Backtesting Framework

Training loss alone is meaningless in finance.

Instead, this project evaluates the model using **rolling historical backtests**:

- The model is repeatedly retrained on expanding windows
- Forecasts are generated for unseen future segments
- Errors are recorded **per forecast horizon**

This produces a **residual matrix** capturing how the model actually performs across time and horizons.

---

## 📉 Dynamic Uncertainty Estimation

For each forecast step:

1. Residual distributions are constructed from backtests  
2. Empirical quantiles (e.g., 90%) are computed  
3. These define **horizon-specific confidence radii**

This approach allows uncertainty to:

- Increase naturally with forecast horizon  
- Adapt to changing market volatility  
- Reflect **real, observed model errors**

No parametric assumptions.  
No artificial confidence.

---

## 📈 From Returns Back to Prices

To produce interpretable forecasts:

1. Predicted log returns are **cumulatively summed**
2. Added to the last observed log price
3. Exponentiated back to price space

### Final output:

- Point forecast  
- Upper confidence band  
- Lower confidence band  

This produces a **probabilistic price cone**, not a single fragile prediction line.

---

## ✅ What Makes This Project Strong

- Multi-step sequence forecasting  
- Empirical, residual-based uncertainty modeling  
- Rolling backtest evaluation methodology  
- Financially sound transformations  
- Clear separation of **prediction vs risk**

This reflects **real quantitative modeling practices**, not toy ML examples.

---

## ⚠️ Limitations

- Assumes historical dynamics persist  
- No macroeconomic or news features  
- Single-asset focus (NVIDIA)

These trade-offs are **intentional**, prioritizing clarity, interpretability, and methodological rigor.

---

## 🔮 Possible Extensions

- Bayesian LSTM or MC Dropout  
- Regime-aware volatility modeling  
- Transformer-based time-series models  
- Multi-asset or portfolio-level uncertainty estimation  

---

## 🧾 Disclaimer

This project is for **educational and research purposes only**.  
It does **not** constitute financial or investment advice.

---

## ⭐ Final Note

This repository is not about predicting the market.

It is about **understanding uncertainty** —  
and building forecasts that acknowledge it.
