# Celebal-week1-assignment

ML Foundations - Week 1 Assignment

This repository contains my Week 1 assignment for Machine Learning Foundations.
The notebook covers core Python, NumPy, Pandas, Linear Algebra, Statistics, and Probability concepts with practical examples and implementations.

Topics Covered
1. Python Fundamentals
Data Types & Control Flow
Data Structures
Exception Handling
Functions & Lambda Expressions

3. NumPy Basics
Array Creation & Shapes
Indexing & Slicing
Mathematical Operations
Dot Product

4. Pandas Basics
DataFrames vs Series
iloc and loc
Filtering & GroupBy
Handling Missing Data

5. Linear Algebra for ML
Vectors & Matrices
Matrix Operations
Eigenvalues & Eigenvectors
SVD & PCA Concepts

6. Statistics for ML
Descriptive vs Inferential Statistics
Hypothesis Testing
Error Metrics
Distribution Testing
Model Monitoring Concepts
7. Probability for ML
Core Probability Concepts
Probability Distributions
Bayes’ Theorem
Central Limit Theorem

File
week1_Prateeksha Khichi_.ipynb — Main assignment notebook
Tools & Libraries Used
Python
NumPy
Pandas
Matplotlib
SciPy
Purpose

The goal of this assignment is to build a strong foundation in the mathematical and programming concepts required for Machine Learning and Data Science.


#CELEBAL WEEK 2 ASSIGNMENT 

# Week 2 – Regression, EDA, and Time Series Forecasting

## Overview

This week focused on strengthening my understanding of Machine Learning fundamentals through Exploratory Data Analysis (EDA), Regression techniques, and Time Series Forecasting using a Tesla Deliveries dataset.

The goal was not only to build models but also to understand the complete workflow from data exploration to model evaluation.

---

## Topics Covered

### 1. Exploratory Data Analysis (EDA)

- Data cleaning and preprocessing
- Handling missing values
- Understanding feature distributions
- Correlation analysis
- Outlier detection
- Data visualization using Matplotlib and Seaborn
- Extracting business insights from the dataset

### Key Learnings

- Importance of understanding data before modeling
- How feature relationships impact model performance
- Identifying trends, patterns, and anomalies through visualization

---

### 2. Regression

Implemented and evaluated multiple regression models:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

### Evaluation Metrics Used

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

### Key Learnings

- Difference between Linear, Ridge, Lasso, and Elastic Net
- Understanding regularization and overfitting
- Importance of feature scaling in regression
- Model comparison using multiple evaluation metrics

---

### 3. Time Series Forecasting

Built forecasting pipelines using Tesla Deliveries data.

### Feature Engineering

- Lag Features (lag_1, lag_2)
- Rolling Mean Features
- Seasonal Features using Sin/Cos Encoding
- Monthly Aggregations

### Models Used

#### SARIMA

- Captures trend and seasonality
- Forecasted future monthly deliveries
- Evaluated using MAE

#### XGBoost Regressor

- Used engineered time-series features
- Trained on historical delivery patterns
- Generated future delivery predictions

### Key Learnings

- Difference between classical statistical forecasting and machine learning forecasting
- Importance of lag and rolling window features
- Handling seasonality using cyclical encoding
- Comparing SARIMA and XGBoost approaches

---

## Dataset Used

Tesla Deliveries Dataset containing:

- Date
- Region
- Model
- Production Units
- Average Price
- Battery Capacity
- Range
- Estimated Deliveries
- CO₂ Saved

---

## Skills Developed

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Regression Modeling
- Model Evaluation
- Time Series Forecasting
- Forecast Visualization
- Business Insight Generation

---

## Reflection

This week helped me understand the complete machine learning workflow from raw data analysis to predictive modeling. I gained practical experience in regression techniques, evaluation metrics, feature engineering, and time series forecasting. Working on the Tesla Deliveries dataset improved my ability to analyze real-world data and apply both statistical and machine learning approaches to forecasting problems.
