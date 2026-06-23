# 🚀 Stock Price Prediction (AAPL Next-Day Closing Price)

A beginner-to-intermediate machine learning project that predicts Apple Inc. (AAPL) next-day closing prices using engineered lag and technical features derived from S&P 500 OHLCV data.

---

## 📌 Problem Statement

Given historical price and volume data for AAPL, the goal is to predict the next trading day’s closing price.  
This is a time-series regression problem with strict chronological splitting to avoid data leakage.

---

## 📊 Dataset

**Source:** S&P 500 Stock Data (Kaggle — camnugent)  
**Full dataset:** ~619,040 daily OHLCV rows (~505 stocks, 2013–2018)  
**Filtered dataset:** AAPL only → ~1,238 trading days after feature engineering  

**Train/Test Split:**
- First 80% (chronological) → Training set  
- Last 20% → Test set  

---

## 🔧 Feature Engineering

**Lag Features**
- lag_1, lag_2, lag_3, lag_5 → previous closing prices  

**Rolling Features**
- ma_5, ma_10, ma_20 → moving averages  
- std_5 → rolling volatility  

**Price Action Features**
- daily_return → % change in close price  
- volume_change → % change in volume  
- high_low_range → intraday volatility  
- open_close_range → intraday direction  

**Target**
- next_close → next-day closing price  

All features are computed using past data only (no look-ahead bias).

---

## ⚙️ Workflow

EDA → Feature Engineering → Chronological Split → Model Training → Hyperparameter Tuning (GridSearchCV) → Evaluation

---

## 🤖 Models Used

- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  

---

## 📈 Model Performance (Test Set)

| Model | MAE (USD) | RMSE (USD) | R² | MAPE |
|------|----------|------------|------|------|
| Linear Regression | 1.33 | 1.90 | 0.9751 | 0.0085 |
| Ridge | 1.33 | 1.90 | 0.9750 | 0.0085 |
| Ridge (Tuned) | 1.33 | 1.90 | 0.9751 | 0.0085 |
| Lasso | 1.33 | 1.90 | 0.9750 | 0.0085 |
| KNN (k=3) | 25.94 | 28.55 | -4.64 | 0.1614 |
| Random Forest | 26.26 | 28.92 | -4.79 | 0.1634 |
| Gradient Boosting | 27.26 | 29.82 | -5.15 | 0.1698 |
| Decision Tree | 27.93 | 30.46 | -5.42 | 0.1742 |

---

## 🏆 Best Model

**Ridge Regression (Tuned)**  
- alpha = 0.1  
- Cross-validation R² ≈ 0.977  
- Test R² ≈ 0.975  

---

## 🧠 Key Insights

- `lag_1` (yesterday’s close) is the strongest predictor of next-day price.
- Stock prices show strong autocorrelation (yesterday ≈ today).
- Moving averages improve short-term trend signal but do not dominate lag features.
- Linear models outperform tree-based models due to smooth, sequential structure.
- Tree models fail due to inability to extrapolate beyond training range.
- High R² (~0.975) reflects persistence, not true predictive trading edge.
- Predicting returns is a more realistic financial ML task than predicting price levels.

---

## 💡 Important Note

Although performance metrics are strong, this is **not a profitable trading strategy**.  
The model mainly learns that future price ≈ current price.

---

## 🛠 Tech Stack

- Python  
- pandas, numpy  
- matplotlib, seaborn  
- scipy  
- scikit-learn  

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
