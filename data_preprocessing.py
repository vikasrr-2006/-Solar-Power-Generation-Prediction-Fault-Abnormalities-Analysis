import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(generation_path, weather_path):
    generation_data = pd.read_csv(generation_path)
    weather_data = pd.read_csv(weather_path)
    return generation_data, weather_data


def preprocess_datetime(df, datetime_col='DATE_TIME', format='%Y-%m-%d %H:%M'):
    df[datetime_col] = pd.to_datetime(df[datetime_col], format=format)
    return df


def merge_datasets(generation_df, weather_df):
    df_solar = pd.merge(
        generation_df.drop(columns=['PLANT_ID']),
        weather_df.drop(columns=['PLANT_ID']),
        on='DATE_TIME'
    )
    return df_solar


def add_time_features(df):
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE_TIME"]).dt.date
    df["TIME"] = pd.to_datetime(df["DATE_TIME"]).dt.time
    df['DAY'] = pd.to_datetime(df['DATE_TIME']).dt.day
    df['MONTH'] = pd.to_datetime(df['DATE_TIME']).dt.month
    df['WEEK'] = pd.to_datetime(df['DATE_TIME']).dt.isocalendar().week
    df['HOURS'] = pd.to_datetime(df['TIME'], format='%H:%M:%S').dt.hour
    df['MINUTES'] = pd.to_datetime(df['TIME'], format='%H:%M:%S').dt.minute
    df['TOTAL_MINUTES_PASS'] = df['MINUTES'] + df['HOURS'] * 60
    df["DATE_STRING"] = df["DATE"].astype(str)
    df["HOURS"] = df["HOURS"].astype(str)
    df["TIME"] = df["TIME"].astype(str)
    return df


def encode_source_key(df):
    encoder = LabelEncoder()
    df['SOURCE_KEY_NUMBER'] = encoder.fit_transform(df['SOURCE_KEY'])
    return df, encoder


def prepare_pipeline(generation_path, weather_path):
    generation_data, weather_data = load_data(generation_path, weather_path)
    generation_data = preprocess_datetime(generation_data, format='%Y-%m-%d %H:%M:%S')
    weather_data = preprocess_datetime(weather_data, format='%Y-%m-%d %H:%M:%S')
    df = merge_datasets(generation_data, weather_data)
    df = add_time_features(df)
    df, encoder = encode_source_key(df)
    return df, encoder
