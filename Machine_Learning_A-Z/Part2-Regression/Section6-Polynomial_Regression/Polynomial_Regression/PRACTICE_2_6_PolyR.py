# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:49:43 2019

@author: Guru
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
from sklearn.linear_model import LinearRegression
L_regressor = LinearRegression()
L_regressor.fit(X,y)
# Predicting a new result
y_pred = L_regressor.predict(X)

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, L_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
"""
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, L_regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Linear Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""

from sklearn.metrics import mean_squared_error,r2_score
print("LR MSE -> ",mean_squared_error(y,y_pred))
print("LR r2_score -> ",r2_score(y,y_pred))





from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# Fitting the Regression Model to the dataset
L_regressor_2 = LinearRegression()
L_regressor_2.fit(X_poly,y)
# Predicting a new result
y_pred = L_regressor_2.predict(X_poly)

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, L_regressor_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Linear Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, L_regressor_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Linear Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



print("Poly LR MSE -> ",mean_squared_error(y,y_pred))
print("Poly LR r2_score -> ",r2_score(y,y_pred))



# Predicting a new result with Linear Regression
print("lin_reg.predict 6.5 ",lin_reg.predict(6.5))

# Predicting a new result with Polynomial Regression
print("lin_reg_2.predict 6.5 ",lin_reg_2.predict(poly_reg.transform(6.5)))




















