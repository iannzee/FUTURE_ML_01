import matplotlib.pyplot as plt
import pandas as pd

def generate_forecast_plot(eval_df: pd.DataFrame, output_path: str = 'outputs/forecast_vs_actual.png'):
    """Generates comparison plots and residual error distribution."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Actual vs Predicted Plot
    ax1.plot(eval_df['date'], eval_df['actual_sales'], label='Actual Sales', color='#1f77b4', lw=2)
    ax1.plot(eval_df['date'], eval_df['forecasted_sales'], label='XGBoost Forecast', color='#d62728', linestyle='--', lw=2)
    ax1.set_title('Demand & Sales Forecasting: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sales Units / Revenue', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Residual Error Plot
    residuals = eval_df['actual_sales'] - eval_df['forecasted_sales']
    ax2.fill_between(eval_df['date'], residuals, color='#2ca02c', alpha=0.3, label='Prediction Residual (Actual - Pred)')
    ax2.axhline(0, color='black', linestyle='-', lw=1)
    ax2.set_title('Forecast Residual Variance', fontsize=12)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Error Margin', fontsize=11)
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {output_path}")

def export_for_powerbi(eval_df: pd.DataFrame, output_csv: str = 'outputs/powerbi_export.csv'):
    """Exports structured actuals and forecasts for direct ingestion into Power BI."""
    eval_df['residual'] = eval_df['actual_sales'] - eval_df['forecasted_sales']
    eval_df['error_pct'] = (eval_df['residual'].abs() / eval_df['actual_sales']) * 100
    eval_df.to_csv(output_csv, index=False)
    print(f"Power BI dataset exported to {output_csv}")