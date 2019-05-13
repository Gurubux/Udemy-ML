# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:32:26 2019

@author: Guru
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# Avoiding the Dummy Variable Trap
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score
print(mean_squared_error(y_pred,y_test))
print(r2_score(y_test,y_pred))
from scipy.stats.stats import pearsonr
xx2_train = [x[2]for x in X_train]
xx3_train = [x[3]for x in X_train]
xx4_train = [x[4]for x in X_train]
print(pearsonr(xx2_train, y_train))
print(pearsonr(xx3_train, y_train))
print(pearsonr(xx4_train, y_train))


xx2 = [x[2]for x in X_test]
xx3 = [x[3]for x in X_test]
xx4 = [x[4]for x in X_test]

plt.scatter(xx2, y_test, color = 'red')
plt.plot(xx2_train, regressor.predict(X_train), color = 'blue')
plt.plot(xx2, y_pred, color = 'yellow')
print(X_test,y_pred)
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



plt.scatter(xx3, y_test, color = 'red')
plt.plot(xx3_train, regressor.predict(X_train), color = 'blue')
plt.plot(xx3, y_pred, color = 'yellow')
print(X_test,y_pred)
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



print(len(xx4_train),len(regressor.predict(X_train)))
plt.scatter(xx4, y_test, color = 'red')
plt.plot(xx4_train, regressor.predict(X_train), color = 'blue')
plt.plot(xx4, y_pred, color = 'yellow')
print(X_test,y_pred)
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


print('----------------------------')
print(np.array(xx2_train).reshape(-1,1),y_train)
print(type(np.array(xx2_train).reshape(-1,1)),type(y_train))
print(len(np.array(xx2_train).reshape(-1,1)),len(y_train))
regressor1 = LinearRegression()
regressor1.fit(np.array(xx2_train).reshape(-1,1), y_train)
y_pred = regressor1.predict(np.array(xx2).reshape(-1,1))
plt.scatter(xx2, y_test, color = 'red')
plt.plot(xx2, y_pred, color = 'yellow')
plt.show()
"""
# Predicting the Test set results
y_pred = regressor1.predict(xx2)
plt.scatter(xx2, y_test, color = 'red')
plt.plot(xx2, y_pred, color = 'yellow')
plt.show()

"""

"""
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.plot(X_test, y_pred, color = 'yellow')
print(X_test,y_pred)
[[ 1.5]
 [10.3]
 [ 4.1]
 [ 3.9]
 [ 9.5]
 [ 8.7]
 [ 9.6]
 [ 4. ]
 [ 5.3]
 [ 7.9]] [ 40835.10590871 123079.39940819  65134.55626083  63265.36777221
 115602.64545369 108125.8914992  116537.23969801  64199.96201652
  76349.68719258 100649.1375447 ]
"""








