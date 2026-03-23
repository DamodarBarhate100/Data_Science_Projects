# Customer Churn Prediction: End-to-End Pipeline & Custom Logistic Regression

## Project Overview
Customer churn is a critical, multi-million dollar problem in the telecommunications industry. This project builds an end-to-end Machine Learning pipeline to predict which customers are at high risk of canceling their subscriptions. 

To demonstrate a deep understanding of the underlying algorithmic mechanics, this project features a **custom Logistic Regression model built entirely from scratch using pure NumPy matrix calculus**, which is then benchmarked head-to-head against the industry-standard `scikit-learn` library.

## Dataset
The dataset used is the classic **Telco Customer Churn** dataset.
* **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Target Variable:** `Churn` (Yes/No)
* **Features:** Customer demographics, account information, and service usage.

## Project Architecture
This pipeline follows strict industry standards to prevent data leakage and ensure mathematical stability:

1. **Exploratory Data Analysis (EDA):** Identifies hidden anomalies (e.g., empty strings in numeric columns) and generates distribution visualizations (KDE and Boxplots) to uncover churn trends based on tenure and monthly charges.
2. **Data Cleaning & Preprocessing:** * Safe coercion of anomalous strings to `NaN` and row dropping.
    * Strict pre-split separation: Data is split into 80/20 Train/Test sets *before* any scaling or encoding to prevent structural data leakage.
    * One-Hot Encoding of categorical features (dropping the first category to avoid multicollinearity).
    * Z-score Normalization (Standard Scaling) of continuous variables.
3. **Modeling & Benchmarking:** Side-by-side training and evaluation of the custom NumPy model vs. the scikit-learn model.

## Mathematical Implementation (Scratch Model)
Rather than relying solely on black-box libraries, the `Scratch_Logistic_Regression` class implements the raw gradient ascent matrix operations. 

**Hypothesis & Activation (Sigmoid):**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Gradient Calculation & Weight Update:**
Optimized using vectorized NumPy dot products where the gradient of the Log-Likelihood is calculated and applied to the parameter weights ($\theta$):
$$\theta := \theta + \alpha \frac{1}{m} X^T (y - \hat{y})$$



## Expected Results
Both models converge to highly similar performance metrics, proving the mathematical accuracy of the custom implementation:
* **Accuracy:** ~80%
* **F1-Score:** ~0.59
* Outputs include a detailed Confusion Matrix highlighting False Negatives (critical for business churn metrics).
* Automatically generates `.png` visualizations in the `Visualizations/` directory.


