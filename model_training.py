import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def prepare_features(df):
    X = df[['DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 
            'MODULE_TEMPERATURE', 'IRRADIATION', 'DC_Power']]
    y = df['AC_POWER']
    return X, y


def split_data(X, y, test_size=0.2, random_state=21):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(random_state=21)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=21)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = round(r2_score(y_pred, y_test) * 100, 2)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return y_pred, r2, mae, rmse


def create_prediction_dataframe(y_test, y_pred):
    cross_check = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    cross_check['Error'] = cross_check['Actual'] - cross_check['Predicted']
    return cross_check
