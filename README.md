# Solar Power Generation Prediction & Fault/Abnormalities Analysis

Predict AC power output from a photovoltaic (PV) solar plant using weather and generation telemetry, and visualize faults/abnormalities from daily DC power patterns.

## Project Structure

```
solar-power-generation-forecast/
├── data/
│   ├── Plant_2_Generation_Data.csv
│   └── Plant_2_Weather_Sensor_Data.csv
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── visualization.py
│   ├── model_training.py
│   └── main.py
├── models/
├── notebooks/
│   └── solar-power-generation-forecast.ipynb
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<username>/solar-power-generation-forecast.git
cd solar-power-generation-forecast
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

## Results

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| LinearRegression | 99.99% | 75.18 | 138.70 |
| RandomForest | 99.99% | 85.27 | 163.36 |
| DecisionTree | 99.97% | 108.38 | 215.05 |

Inverter Efficiency (AC/DC): **96.345%**

## Dataset

- **Plant_2_Generation_Data.csv**: DATE_TIME, PLANT_ID, DC_Power, AC_POWER, DAILY_YIELD, TOTAL_YIELD
- **Plant_2_Weather_Sensor_Data.csv**: DATE_TIME, PLANT_ID, SOURCE_KEY, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION

## License

MIT