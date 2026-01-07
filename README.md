# Demand Forecasting using Time Series Models

## Overview
This project focuses on demand forecasting at a granular level (Store ID × Product ID) using multiple time series and machine learning models. The goal is to build baseline models, systematically fine-tune them, and perform a fair comparison using standard evaluation metrics.
The pipeline is designed to be:
* Modular
* Interpretable
* Research-paper ready
* Industry-aligned


## Objectives
* Preserve store level and product level granularity
* Implement baseline forecasting models
* Perform step by step fine tuning for each model
* Compare baseline vs fine-tuned models
* Identify the best-performing model per Store–Product
* Provide a clean evaluation using MAPE and R²


## Data Description
The dataset consists of time-stamped transactional demand data collected across multiple stores and products. Each record corresponds to the demand observed for a specific product in a particular store on a given date. Along with the target variable Demand, the dataset includes several explanatory features such as pricing information, promotional indicators, competitor pricing, and contextual attributes like region, category, and weather conditions. These additional variables enable both univariate and multivariate forecasting approaches. The temporal nature of the data makes it suitable for classical time series modeling, while the presence of external regressors allows the exploration of more advanced models such as SARIMAX and Prophet with regressors.


## Data Preprocessing

The preprocessing pipeline ensures that the data is clean, consistent and suitable for time series modeling. Key steps include:
* Conversion of the date column to datetime format and proper temporal sorting
* Handling missing values using forward filling within each Store–Product series
* Log transformation of demand (log1p) to stabilize variance

Feature engineering, including:
* Lag features (1 day, 7 day, 14 day)
* Time-based features (day, week, month, weekday)
* Encoding categorical variables using label encoding
* Ensuring no data leakage through strict temporal train–test splitting

## Models Implemented
### Baseline Models
The following baseline models are implemented to establish a performance reference:
* Linear Regression
* Holt-Winters Exponential Smoothing
* ARIMA (1,1,1)
* Prophet (Univariate)
* Prophet (Multivariate)
* XG Boost


### Fine-Tuned Models
Each model is fine-tuned independently, using interpretable and statistically grounded techniques.
1️. Linear Regression (Ridge-Regularized)
  - Lag-based feature engineering to capture temporal dependencies
  - Ridge regularization to mitigate multicollinearity
  - Hyperparameter tuning via cross-validated alpha selection

2️. Holt-Winters Exponential Smoothing
  - Selection of additive vs multiplicative trend and seasonality
  - Seasonality optimization (weekly/monthly patterns)
  - Evaluation of smoothing parameters




