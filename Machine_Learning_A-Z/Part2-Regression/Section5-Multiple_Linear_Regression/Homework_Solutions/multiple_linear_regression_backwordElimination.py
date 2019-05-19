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
from sklearn.cross_validation import train_test_split
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

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X1=X
X = np.append(arr = np.ones((40, 1)).astype(int), values = X_train, axis = 1)#sm.add_constant(X)
y1=y
y=y_train

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
print("[0, 1, 2, 3, 4, 5]",["%.3f"%x for x in regressor_OLS.pvalues])
print("rsquared_adj ",regressor_OLS.rsquared_adj)
print("Max PValue ",np.max(regressor_OLS.pvalues),'\n')

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
print("[0, 1, 3, 4, 5]",["%.3f"%x for x in regressor_OLS.pvalues])
print("rsquared_adj ",regressor_OLS.rsquared_adj)
print("Max PValue ",np.max(regressor_OLS.pvalues),'\n')

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
print("[0, 3, 4, 5]",["%.3f"%x for x in regressor_OLS.pvalues])
print("rsquared_adj ",regressor_OLS.rsquared_adj)
print("Max PValue ",np.max(regressor_OLS.pvalues),'\n')

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
print("[0, 3, 5]",["%.3f"%x for x in regressor_OLS.pvalues])
print("rsquared_adj ",regressor_OLS.rsquared_adj)
print("Max PValue ",np.max(regressor_OLS.pvalues),'\n')

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
print("[0, 3]",["%.3f"%x for x in regressor_OLS.pvalues])
print("rsquared_adj ",regressor_OLS.rsquared_adj)
print("Max PValue ",np.max(regressor_OLS.pvalues),'\n')

X_test = np.append(arr = np.ones((10, 1)).astype(int), values = X_test, axis = 1)
y_pred_ols = regressor_OLS.predict(X_test[:, [0, 3]])


from sklearn.metrics import r2_score,mean_squared_error
print("LR r2_score-> ",r2_score(y_test,y_pred))
print("OLS r2_score-> ",r2_score(y_test,y_pred_ols))


print("LR MSE-> ",mean_squared_error(y_test,y_pred))
print("OLS MSE-> ",mean_squared_error(y_test,y_pred_ols))






"""print('----------------------------')
print(np.array(xx2_train).reshape(-1,1),y_train)
print(type(np.array(xx2_train).reshape(-1,1)),type(y_train))
print(len(np.array(xx2_train).reshape(-1,1)),len(y_train))
regressor1 = LinearRegression()
regressor1.fit(np.array(xx2_train).reshape(-1,1), y_train)
y_pred = regressor1.predict(np.array(xx2).reshape(-1,1))"""
plt.scatter(X_test[:, [3]], y_test, color = 'red')
plt.plot(X_test[:, [3]], y_pred, color = 'yellow')
plt.show()


plt.scatter(X1[:, [2]], y1, color = 'black',alpha=0.5,s=100)
plt.scatter(X_test[:, [3]], y_test, color = 'red')
plt.scatter(X_test[:, [0]], y_test, color = 'blue')
plt.plot(X_test[:, [3]], y_pred_ols, color = 'yellow')
plt.plot(X_test[:, [0]], y_pred_ols, color = 'yellow')
plt.xlabel('R&D Spend')
plt.ylabel('Profits')
plt.show()