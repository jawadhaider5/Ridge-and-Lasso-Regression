# PROBLEM
# Predict median house value for California Districts using numerical and categorical features.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error

# Load Dataset
housing = fetch_california_housing(as_frame=True)
housing_data = housing.frame

# Split features/target
X = housing_data.drop(columns=["MedHouseVal"])
y = housing_data["MedHouseVal"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial Ridge Regression
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
ridge_cv.fit(X_train_poly, y_train)
y_pred_ridge = ridge_cv.predict(X_test_poly)

print("\nPolynomial Ridge Regression")
print("Best Alpha:", ridge_cv.alpha_)
print("R²:", r2_score(y_test, y_pred_ridge))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))

# Lasso Regression
lasso_cv = LassoCV(alphas=np.logspace(-4, 0, 50), max_iter=50000)
lasso_cv.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_cv.predict(X_test_scaled)

print("\nLasso Regression")
print("Best Alpha:", lasso_cv.alpha_)
print("R²:", r2_score(y_test, y_pred_lasso))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lasso)))

corr_matrix = housing_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, linewidths=.5)
plt.title("Correlation Heatmap - California Housing Dataset")
plt.savefig("correlation_heatmap.png")
plt.show()