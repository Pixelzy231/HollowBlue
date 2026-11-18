# HydroSense AI prototype

```
hydrosense/
  backend/       train_model.py, app.py (Flask API), requirements.txt
  sensor_sim/    sensor_sim.cpp (simulated field sensors)
  frontend/      index.html (Leaflet map + Chart.js dashboard)
```

## Run

```bash
cd backend
pip install flask scikit-learn pandas joblib
python train_model.py          # creates the two .joblib model files

# optional: compile the C++ sensor simulator (Python fallback is built in)
cd ../sensor_sim && g++ -O2 -o sensor_sim sensor_sim.cpp && cd ../backend

python app.py                  # open http://localhost:5000
```

The dashboard polls `/api/status` every 3 seconds. The page needs internet for
map tiles, Leaflet, Chart.js and the font.

## Honest notes for the pitch

- Models are trained on synthetic data. Say so, and show the loader swap point
  in `make_demand_data()` for NASA POWER / SMAP / FAO data.
- "Water saved" compares the AI recommendation with an assumed fixed 8 mm/day
  schedule. Replace it with a real baseline when you have one.
- Zone coordinates and areas are illustrative.
