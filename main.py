import numpy as np
import pandas as pd
from src.features import load_and_aggregate_data, build_features
from src.train import temporal_train_test_split, train_and_evaluate
from src.visualize import generate_forecast_plot, export_for_powerbi

def generate_mock_data():
    """Generates synthetic multi-year sales data if raw dataset is absent."""
    date_range = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
    n = len(date_range)
    
    trend = np.linspace(50, 120, n)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    weekly = 10 * (date_range.dayofweek >= 5).astype(int)
    noise = np.random.normal(0, 5, n)
    
    sales = np.maximum(10, trend + seasonality + weekly + noise).round().astype(int)
    df = pd.DataFrame({'date': date_range, 'sales': sales})
    df.to_csv('data/sales_data.csv', index=False)

if __name__ == '__main__':
    # 1. Ensure data exists
    import os
    if not os.path.exists('data/sales_data.csv'):
        print("Generating mock sales data...")
        generate_mock_data()
        
    # 2. Pipeline Execution
    raw_df = load_and_aggregate_data('data/sales_data.csv')
    featured_df = build_features(raw_df)
    
    X_train, X_test, y_train, y_test, test_meta = temporal_train_test_split(featured_df)
    model, predictions = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 3. Compile output dataframe
    eval_df = test_meta.copy()
    eval_df.rename(columns={'sales': 'actual_sales'}, inplace=True)
    eval_df['forecasted_sales'] = np.round(predictions, 2)
    
    # 4. Generate Visuals & Power BI Artifacts
    generate_forecast_plot(eval_df)
    export_for_powerbi(eval_df)