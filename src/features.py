import pandas as pd


def load_and_aggregate_data(filepath: str) -> pd.DataFrame:
    """Loads raw sales data, parses dates, and aggregates to daily level."""
    df = pd.read_csv(filepath)
    required_columns = {'date', 'sales'}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df['date'] = pd.to_datetime(df['date'], errors='raise')

    # Aggregate transaction records to daily totals.
    daily_df = df.groupby('date', as_index=False, sort=True)['sales'].sum()
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    return daily_df
    

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates calendar,lag, and rolling window features."""
    df = df.copy()
    if not {'date', 'sales'}.issubset(df.columns):
        raise ValueError("Input data must contain 'date' and 'sales' columns")
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    df = df.sort_values('date').reset_index(drop=True)

    # Calendar features.
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    # Lag features use only historical demand.
    for lag in (1, 7, 14, 30):
        df[f'lag_{lag}'] = df['sales'].shift(lag)

    # Shift before rolling to prevent using the current target (data leakage).
    df['rolling_mean_7'] = df['sales'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df['sales'].shift(1).rolling(window=7).std()
    df['rolling_mean_30'] = df['sales'].shift(1).rolling(window=30).mean()

    df = df.dropna().reset_index(drop=True)
    return df