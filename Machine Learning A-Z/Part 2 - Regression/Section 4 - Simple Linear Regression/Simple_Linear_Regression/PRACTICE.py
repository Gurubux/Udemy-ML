# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:28:14 2019

@author: Guru
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
print(X,y)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
print(X_train,y_train)
regressor.fit(X_train, y_train)
print(regressor.coef_)
print(regressor.intercept_)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

'''
Using Pyplot

trace = go.Scatter(
    x = X_train,
    y = y_train,
    mode = 'markers',
    name = 'Training Dataset '
)

trace1 = go.Scatter(
    x = X_train,
    y = regressor.predict(X_train),
    mode = 'lines',
    name = 'Predicted Values Training Set'
)

data = [trace,trace1]

layout = dict(title = 'Salary vs Experience(Training set)',
              xaxis = dict(title = 'Years of Experience'),
              yaxis = dict(title = 'Salary'),
              )
fig = dict(data=data, layout=layout)

py.plot(data, filename='Simple Linear Regression 1')
'''


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.plot(X_test, y_pred, color = 'yellow')
print(X_test,y_pred)
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

'''
Using Pyplot

trace = go.Scatter(
    x = X_test,
    y = y_test,
    mode = 'markers',
    name = 'Test Dataset '
)

trace1 = go.Scatter(
    x = X_test,
    y = y_pred,
    mode = 'lines',
    name = 'Predicted Values Test Set'
)

trace2 = go.Scatter(
    x = X_train,
    y = regressor.predict(X_train),
    mode = 'lines',
    name = 'Predicted Values train Set'
)


data = [trace,trace1,trace2]

layout = dict(title = 'Salary vs Experience (Test set)',
              xaxis = dict(title = 'Years of Experience'),
              yaxis = dict(title = 'Salary'),
              )
fig = dict(data=data, layout=layout)

py.plot(data, filename='Simple Linear Regression 2')
'''



print("MSE",np.mean((y_test-y_pred)**2),np.mean((y_pred - y_test) ** 2))
print("MAE",np.mean(np.abs(y_test-y_pred)),np.mean(np.absolute(y_pred - y_test)))
print("R2_score",r2_score(y_pred,y_test))