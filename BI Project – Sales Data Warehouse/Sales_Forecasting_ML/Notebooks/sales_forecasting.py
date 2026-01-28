import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# 1) Load data
# =========================
data = pd.read_csv(
    "Data/monthly_sales.csv",
    sep=";",
    header=None,
    names=["date", "sales"]
)

data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date")

ts = data.set_index("date")["sales"]

print("Rows:", len(ts))


# =========================
# 2) Log transformation
# =========================
ts_log = np.log(ts)

print("\nLog series head:")
print(ts_log.head())


# =========================
# 3) Stationarity check (ADF)
# =========================
adf = adfuller(ts_log.dropna())
print("\n---- ADF TEST (LOG) ----")
print("p-value:", adf[1])


# =========================
# 4) Train/Test split
# =========================
test_size = 12
train = ts_log.iloc[:-test_size]
test  = ts_log.iloc[-test_size:]

print("\nTrain:", len(train), "Test:", len(test))


# =========================
# 5) ARIMA grid (log series)
# =========================
orders = [(1,0,1), (2,0,1), (2,0,2), (3,0,1), (3,0,2)]

best_order = None
best_rmse = float("inf")
best_fit = None
best_pred_log = None

print("\n---- GRID SEARCH (ARIMA + LOG) ----")
for order in orders:
    try:
        model = ARIMA(train, order=order)
        fit = model.fit()
        pred_log = fit.forecast(steps=len(test))

        # Back to real scale
        pred = np.exp(pred_log)
        real = np.exp(test)

        rmse = np.sqrt(mean_squared_error(real, pred))
        print(f"ARIMA{order} -> RMSE = {rmse:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_order = order
            best_fit = fit
            best_pred_log = pred_log

    except Exception as e:
        print(f"ARIMA{order} -> ERROR")

print("\n✅ BEST ARIMA (LOG):", best_order, "RMSE =", round(best_rmse, 2))


# =========================
# 6) Evaluation
# =========================
pred_test = np.exp(best_pred_log)
real_test = np.exp(test)

mae = mean_absolute_error(real_test, pred_test)
rmse = np.sqrt(mean_squared_error(real_test, pred_test))

print("\n---- EVALUATION (LOG ARIMA) ----")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))


# =========================
# 7) Plot (test only)
# =========================
plt.figure(figsize=(9,4))
plt.plot(real_test.index, real_test.values, label="Real (Test)")
plt.plot(real_test.index, pred_test.values, label=f"Forecast ARIMA{best_order} (log)", linestyle="--")
plt.legend()
plt.title("ARIMA with Log-Transformation (Test Period)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()


# =========================
# 8) Forecast future 6 months
# =========================
future_steps = 6
future_log = best_fit.forecast(steps=future_steps)
future_sales = np.exp(future_log)

last_date = ts.index.max()
future_dates = pd.date_range(last_date, periods=future_steps+1, freq="M")[1:]

future_df = pd.DataFrame({
    "date": future_dates,
    "forecast_sales": future_sales
})

print("\n---- FUTURE 6 MONTHS (LOG ARIMA) ----")
print(future_df)

future_df.to_csv("Outputs/arima_log_forecast_6months.csv", index=False)
print("\n✅ Saved: Outputs/arima_log_forecast_6months.csv")







