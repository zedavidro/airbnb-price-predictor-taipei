from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, make_scorer, root_mean_squared_error
import xgboost as xg
from xgboost import XGBRegressor, plot_importance
import numpy as np
import pandas as pd

def prepare_training_data(df, cap_low, cap_high, feature_groups, use_log):
    df = df.copy()
    df['price_capped'] = df['price'].clip(lower=cap_low, upper=cap_high)

    if use_log:
        df['target'] = np.log1p(df['price_capped'])
    else:
        df['target'] = df['price_capped']

    selected_features = []
    for group in feature_groups.values():
        selected_features += group

    X = df[selected_features]
    y = df['target']

    return X, y

def evaluate_model(model, X_train, y_train, X_test, y_test, use_log=False):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if use_log:
        y_test_raw = np.expm1(y_test)
        y_pred_raw = np.expm1(y_pred)
    else:
        y_test_raw = y_test
        y_pred_raw = y_pred

    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
    mape = np.mean(np.abs((y_test_raw - y_pred_raw) / y_test_raw)) * 100

    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return y_test_raw, y_pred_raw