# FUTURE_ML_01: Sales & Demand Forecasting for Businesses

This repository contains an end-to-end Machine Learning pipeline designed to forecast daily business demand and sales using historical data[cite: 1]. It was developed as part of the Future Interns Machine Learning track (Task 1)[cite: 1].

## 📌 Project Overview
The project transforms sequential time-series data into actionable business intelligence[cite: 1]. It leverages feature engineering and an XGBoost Regressor to generate robust predictions, outputting visual dashboards and structured datasets for tools like Power BI[cite: 1].

## ✨ Key Features
*   **Time-Series Transformations:** Extracts calendar attributes, multi-period lags (1, 7, 14, and 30 days), and moving averages to capture seasonal trends[cite: 1].
*   **Leak-Free Temporal Splitting:** Implements strict chronological splitting for training (80%) and testing (20%) to prevent data leakage and lookahead bias[cite: 1].
*   **Model Evaluation:** Assesses predictive performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE)[cite: 1].
*   **Business Reporting:** Automatically generates visual diagnostic curves (actuals vs. predicted and residual errors) and structured CSV exports for interactive Power BI dashboards[cite: 1].

## 📁 Repository Structure
```text
FUTURE_ML_01/
├── data/                       # Contains raw and aggregated sales data
├── notebooks/                  # Experimental Jupyter notebooks
├── src/                        # Core source code modules
│   ├── __init__.py
│   ├── features.py             # Data aggregation and feature engineering
│   ├── train.py                # Model training and evaluation logic
│   └── visualize.py            # Plotting and Power BI export functions
├── outputs/                    # Generated visuals and prediction CSVs
├── .gitignore                  # Git tracking exclusion rules
├── main.py                     # Master execution pipeline
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```[cite: 1]

## 🛠️ Tech Stack & Dependencies
*   **Data Processing:** Pandas, NumPy[cite: 1].
*   **Machine Learning:** Scikit-Learn, XGBoost[cite: 1].
*   **Visualization:** Matplotlib, Seaborn, Power BI[cite: 1].

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/FUTURE_ML_01.git](https://github.com/iannzee/FUTURE_ML_01.git)
cd FUTURE_ML_01
```[cite: 1]

**2. Set up a virtual environment**
```bash
python -m venv venv
```[cite: 1]
*   *On Windows:* `venv\Scripts\activate`[cite: 1]
*   *On macOS/Linux:* `source venv/bin/activate`[cite: 1]

**3. Install dependencies**
```bash
pip install -r requirements.txt
```[cite: 1]

**4. Execute the pipeline**
Running the main script will generate mock data (if no dataset is present), extract features, train the model, and output the performance metrics and visualizations[cite: 1].
```bash
python main.py
```[cite: 1]
