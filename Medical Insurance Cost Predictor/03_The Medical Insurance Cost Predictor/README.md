# Medical Insurance Cost Prediction: From Scratch vs. Scikit-Learn

This repository contains a comprehensive Machine Learning project that predicts medical insurance charges. The core of the project is a **Multiple Linear Regression** model built entirely from scratch using **Batch Gradient Descent**, benchmarked against Scikit-Learn's `SGDRegressor`.

## Project Overview
The goal was to understand the underlying mathematical optimization of Linear Regression. By deriving the gradient update rules and implementing them in NumPy, this project achieves a high level of predictive accuracy on the "Medical Cost Personal Dataset."

### Key Features:
* **Exploratory Data Analysis (EDA):** Detailed visualization of BMI vs. Charges, Smoker interactions, and outlier detection using Seaborn/Matplotlib.
* **Manual Feature Engineering:** Custom Z-score standardization and One-Hot Encoding (handling the Dummy Variable Trap).
* **Matrix Calculus from Scratch:** Implementation of the Cost Function (MSE) and Gradient Descent without using high-level ML libraries.
* **Benchmarking:** A direct comparison between the "From Scratch" model and `sklearn.linear_model.SGDRegressor`.

## Performance Comparison
| Metric | Scratch (Batch GD) | Scikit-Learn (SGD) |
| :--- | :--- | :--- |
| **R² Score** | **0.8045** | 0.8033 |
| **MSE** | **35,914,551** | 36,141,422 |

*The "From Scratch" model slightly outperformed the library version, demonstrating successful convergence to the Global Minimum.*

## Tech Stack
* **Language:** Python 3.x
* **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn

## Project Structure
* `insurance_analysis.py`: Main script containing the data pipeline and custom Gradient Descent logic.
* `Visualizations/`: Folder containing generated EDA plots (Correlation Heatmaps, Boxplots, Scatter plots).
* `Datasets/`: The raw insurance.csv data.

## What I Learned
1. The importance of **Feature Scaling** for Gradient Descent stability.
2. How the **Learning Rate ($\alpha$)** affects the convergence of the cost function.
3. Managing the trade-off between **Batch GD** (stability) and **Stochastic GD** (speed).
