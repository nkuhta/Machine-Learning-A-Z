# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Importing the data
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#  Encoding Categorical Data 
#  Encode the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
#  remove first column
X = X[:, 1:]

#  Split into testing and training data set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0) 

#  Fit Mutiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# fit to training set
regressor.fit(X_train,y_train)

#  Predicting teh Test Set Results
y_pred = regressor.predict(X_test)

#  Building optimal model using Backward Elimination
import statsmodels.formula.api as sm
#  add column of ones 
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
# optimal matrix features 
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())
#  R&D spent is the most useful feature! 
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())

#  Backward elimination (automation)
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
