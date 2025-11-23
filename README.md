California Housing Price Prediction
Regression Analysis using Ridge, Polynomial Ridge, and Lasso
This project predicts median house values for California districts using the California Housing dataset from Scikit-Learn.
The goal is to compare different regression techniques and identify which model provides the best predictive performance.

Project Overview
This project includes:

Data loading & preprocessing
Train/test splitting
Feature scaling using StandardScaler

Model training using:
Ridge Regression (initial approach)
Polynomial Ridge Regression (final improved model)
Lasso Regression

Cross-validation using RidgeCV and LassoCV

Model evaluation using R² Score and RMSE

Correlation heatmap visualization

Objective
To build and evaluate regression models capable of predicting:
MedHouseVal — the median house value of a California district using 8 numerical housing attributes.

Preprocessing
No missing values in dataset

Features separated from target (MedHouseVal)

Train/test split (80/20)

Scaling using:
StandardScaler()


Models Implemented

1. Ridge Regression (Initial Model)
Initially, simple Ridge Regression was applied. While it produced acceptable results, the performance remained limited because the model was strictly linear.

2. Polynomial Ridge Regression (Final Improved Model)
To capture non-linear relationships in the dataset, the features were expanded using:
PolynomialFeatures(degree = 2)

Then the model was optimized using RidgeCV:
RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
This significantly improved the R² and reduced the RMSE.

3. Lasso Regression
LassoCV was used to explore feature shrinkage and sparsity, but performance remained lower than Polynomial Ridge. Still included for comparison.

Results
Polynomial Ridge performed the best, proving that the dataset benefits from non-linear feature expansion.

Why Polynomial Ridge Was Chosen
Ridge alone was not capturing non-linear interactions
Polynomial features (degree 2) expanded the feature space
Ridge regularization handled the increased complexity

Cross-validation found the best alpha value automatically

The result:
Better generalization, higher accuracy, and lower error.

Correlation Heatmap
A heatmap was generated to visualize relationships between features and the target.
The image is saved as: correlation_heatmap.png


How to Run
pip install -r requirements.txt
python src/main.py

Requirements
numpy
matplotlib
seaborn
scikit-learn

Contact if you want to suggest improvements in my model :) email: jawadhaider204@gmail.com