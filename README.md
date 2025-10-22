# Store Sales Time Series Forecasting

🔗 [Kaggle Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

## Overview
This project tackles the Store Sales Time Series Forecasting challenge on Kaggle. The goal is to predict store sales for 16 days using historical sales data spanning 1,684 days, along with various supplementary features such as oil prices, holidays, and transaction data.

## Methodology

### 1. Exploratory Data Analysis (EDA)
- **Target Variable Transformation**: Applied log transformation (`log1p`) to normalize the highly skewed sales distribution
- **Seasonality Analysis**: 
  - Identified strong weekly seasonality patterns
  - Detected biweekly and monthly seasonal components using periodogram analysis
  - Used Fourier features (order=4) to capture complex seasonal patterns

### 2. Feature Engineering

#### Oil Price Features
- Created 7-day rolling average of oil prices
- Generated 3 lagged features (oil_lag_1, oil_lag_2, oil_lag_3)
- Observed inverse relationship between oil prices and sales

#### Holiday Features
- Engineered weekday indicators considering:
  - Bridge days and transfers
  - Work days vs. holidays
  - Holiday transfers affecting business days
- One-hot encoded holiday types (Additional, Bridge, Event, Holiday, Transfer, Work Day)

#### Transaction Features
- Created lagged transaction features (lag 16) as leading indicators

#### Sales Lag Features
- Generated 20 lagged sales features (sales_lag_1 through sales_lag_20)
- Used for recursive forecasting in the prediction phase

#### Calendar Features
- Month, day of month, day of year
- Week of year, day of week
- Year

### 3. Feature Selection
- **Correlation Analysis**: Identified features with strong linear relationships to sales
- **Mutual Information**: Measured non-linear dependencies between features and target variable

### 4. Modeling Approach

#### Baseline Model
- Linear Regression with OneHotEncoder for categorical variables (store_nbr, family, city, state, store_type)

#### Final Model
- **LightGBM** with the following hyperparameters:
  - Boosting type: GBDT
  - Number of leaves: 8
  - Learning rate: 0.2
  - Max depth: 7
  - Early stopping: 200 rounds
  - Metric: MSE

#### Recursive Prediction Strategy
For the 16-day forecast period, implemented a recursive prediction approach:
1. Predict sales for day 16
2. Use predicted values to fill lag features for subsequent days
3. Iteratively predict remaining days (17-31)

### 5. Evaluation Metric
- **RMSLE (Root Mean Squared Logarithmic Error)**: Measures prediction accuracy on log-transformed scale
