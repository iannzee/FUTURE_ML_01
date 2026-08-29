import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score
from xgboost import XGBRegressor
import joblib

def temporal_train_test_split(df:pd.DataFrame, split_ratio: float = 0.8):
    """Splits data strictly along chronological index."""
    feature_cols = [
        'day_of_week', 'day_of_month','month','quarter','is_weekend',
        'lag_1','lag_7','lag_14','lag_30',
        'rolling_mean_7','rolling_std_7','rolling_mean_30'
    ]
    target_col = 'sales'

    split_index = int(len(df) * split_ratio)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    return X_train, X_test, y_train, y_test, test_df[['date', 'sales']]

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Trains an XGBoost regressor and outputs regression metrics."""
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Metric Calculations
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    r2 = r2_score(y_test, predictions)
    
    print("--- MODEL PERFORMANCE EVALUATION ---")
    print(f"MAE  (Mean Absolute Error)     : {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error)  : {rmse:.2f}")
    print(f"MAPE (Mean Absolute % Error)    : {mape:.2f}%")
    print(f"R² Score                        : {r2:.4f}")
    
    # Save model artifact
    joblib.dump(model, 'outputs/xgb_forecast_model.pkl')
    
    return model, predictions