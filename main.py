import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import prepare_pipeline
from visualization import plot_daily_dc_power, calculate_inverter_efficiency
from model_training import (
    prepare_features, split_data,
    train_linear_regression, train_random_forest, train_decision_tree,
    evaluate_model, create_prediction_dataframe
)


def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data')

    generation_path = os.path.join(data_path, 'Plant_2_Generation_Data.csv')
    weather_path = os.path.join(data_path, 'Plant_2_Weather_Sensor_Data.csv')

    print("Loading and preprocessing data...")
    df, encoder = prepare_pipeline(generation_path, weather_path)

    print("Dataset shape:", df.shape)
    print("Missing values:")
    print(df.isnull().sum())

    print("\nCalculating inverter efficiency...")
    efficiency = calculate_inverter_efficiency(df)
    print(f"Power ratio AC/DC (Efficiency) of Solar Power Plant: {efficiency:.3f}%")

    print("\nPreparing features for ML...")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTraining models...")
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)

    print("\n" + "=" * 50)
    print("Model Results:")
    print("=" * 50)

    models = {
        'LinearRegression': lr_model,
        'RandomForest': rf_model,
        'DecisionTree': dt_model
    }

    for name, model in models.items():
        y_pred, r2, mae, rmse = evaluate_model(model, X_test, y_test)
        print(f"\n{name}:")
        print(f"  R2 Score: {r2}%")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")

    print("\nGenerating predictions sample...")
    predictions = rf_model.predict(X_test)
    pred_df = create_prediction_dataframe(y_test, predictions)
    print(pred_df.head(10))

    print("\nDaily DC Power Analysis...")
    plot_daily_dc_power(df)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
