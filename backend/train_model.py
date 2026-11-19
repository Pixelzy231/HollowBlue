"""Train HydroSense models.

1. RandomForest: predicts irrigation need (mm/day) from climate + soil.
2. IsolationForest: flags leak-like anomalies from pressure/flow readings.

Data here is SYNTHETIC (physically plausible). To use real data, replace
make_demand_data() with a loader for NASA POWER / SMAP / FAO CSVs that
returns the same columns.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)


def make_demand_data(n=8000):
    temp = rng.uniform(15, 46, n)          # deg C
    humidity = rng.uniform(10, 70, n)      # %
    soil = rng.integers(0, 3, n)           # 0 sandy, 1 loam, 2 clay
    moisture = rng.uniform(5, 45, n)       # volumetric %
    et = 0.12 * temp - 0.03 * humidity + 2.0
    need = et * (1.4 - moisture / 50) * (1.3 - 0.15 * soil)
    need = np.clip(need + rng.normal(0, 0.3, n), 0.5, None)
    return pd.DataFrame(dict(temp=temp, humidity=humidity, soil=soil,
                             moisture=moisture, need_mm=need))


def make_normal_network(n=5000):
    return pd.DataFrame(dict(pressure=rng.normal(4.0, 0.12, n),   # bar
                             flow=rng.normal(100, 5, n)))         # L/s


if __name__ == "__main__":
    df = make_demand_data()
    X, y = df[["temp", "humidity", "soil", "moisture"]], df["need_mm"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
    reg = RandomForestRegressor(n_estimators=150, random_state=1).fit(Xtr, ytr)
    pred = reg.predict(Xte)
    print(f"Demand model  MAE={mean_absolute_error(yte, pred):.3f} mm/day  "
          f"R2={r2_score(yte, pred):.3f}")

    iso = IsolationForest(contamination=0.02, random_state=1)
    iso.fit(make_normal_network())

    joblib.dump(reg, "demand_model.joblib")
    joblib.dump(iso, "anomaly_model.joblib")
    print("Saved demand_model.joblib and anomaly_model.joblib")
