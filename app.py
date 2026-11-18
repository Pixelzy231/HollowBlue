"""HydroSense API.  Run: python app.py  ->  http://localhost:5000"""
import json
import pathlib
import random
import subprocess
from collections import deque

import joblib
import pandas as pd
from flask import Flask, jsonify

ROOT = pathlib.Path(__file__).resolve().parent
SIM = ROOT.parent / "sensor_sim" / "sensor_sim"
app = Flask(__name__, static_folder=str(ROOT.parent / "frontend"), static_url_path="")

demand_model = joblib.load(ROOT / "demand_model.joblib")
anomaly_model = joblib.load(ROOT / "anomaly_model.joblib")

# Demo area: Toshka, southern Egypt (coordinates are illustrative)
ZONES = [
    {"name": "Toshka A", "lat": 22.95, "lon": 31.55, "soil": 0, "ha": 420},
    {"name": "Toshka B", "lat": 22.88, "lon": 31.62, "soil": 1, "ha": 380},
    {"name": "Toshka C", "lat": 22.80, "lon": 31.50, "soil": 0, "ha": 510},
    {"name": "Toshka D", "lat": 22.74, "lon": 31.66, "soil": 2, "ha": 300},
    {"name": "Toshka E", "lat": 22.66, "lon": 31.57, "soil": 1, "ha": 450},
    {"name": "Toshka F", "lat": 22.60, "lon": 31.70, "soil": 0, "ha": 350},
]
ROUTINE_MM = 8.0  # fixed schedule used as the baseline
history = deque(maxlen=20)


def read_sensors():
    """Use the C++ simulator if compiled, otherwise a Python fallback."""
    if SIM.exists():
        out = subprocess.run([str(SIM), str(len(ZONES))], capture_output=True,
                             text=True, timeout=5).stdout
        return json.loads(out)
    data = []
    for i in range(len(ZONES)):
        p, f = random.gauss(4.0, 0.12), random.gauss(100, 5)
        if random.random() < 0.12:
            p -= random.uniform(1.2, 2.2)
            f += random.uniform(25, 55)
        data.append(dict(zone=i, pressure=p, flow=f,
                         temp=random.gauss(35, 3),
                         humidity=min(max(random.gauss(25, 6), 8), 70),
                         moisture=min(max(random.gauss(22, 8), 4), 45)))
    return data


@app.get("/api/status")
def status():
    readings = read_sensors()
    zones, rec_total, routine_total = [], 0.0, 0.0
    for z, r in zip(ZONES, readings):
        feats = pd.DataFrame([[r["temp"], r["humidity"], z["soil"], r["moisture"]]],
                             columns=["temp", "humidity", "soil", "moisture"])
        need = float(demand_model.predict(feats)[0])
        leak = anomaly_model.predict(
            pd.DataFrame([[r["pressure"], r["flow"]]], columns=["pressure", "flow"]))[0] == -1
        stress = "high" if r["moisture"] < 15 else "medium" if r["moisture"] < 25 else "low"
        rec_m3 = need * z["ha"] * 10      # 1 mm over 1 ha = 10 m3
        routine_m3 = ROUTINE_MM * z["ha"] * 10
        rec_total += rec_m3
        routine_total += routine_m3
        zones.append({**z, **{k: round(v, 2) for k, v in r.items() if k != "zone"},
                      "need_mm": round(need, 2), "stress": stress, "leak": bool(leak),
                      "recommended_m3": round(rec_m3)})
    history.append({"recommended": round(rec_total), "routine": round(routine_total)})
    saved = 100 * (1 - rec_total / routine_total)
    return jsonify(zones=zones, history=list(history),
                   kpi={"saved_pct": round(saved, 1),
                        "saved_m3": round(routine_total - rec_total),
                        "leaks": sum(z["leak"] for z in zones),
                        "high_stress": sum(z["stress"] == "high" for z in zones)})


@app.get("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
