# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
rand_seed = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = rand_seed)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#  Fit Linear Model to the training set
regressor.fit(X_train,y_train)

# Predicting the Test results
y_pred = regressor.predict(X_test)

#print(regressor.coef_)
#print(regressor.intercept_)

#  Visualising the Training Dataset
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Data)',size=12)
plt.xlabel('Experience (years)',size=12)
plt.ylabel('Salary',size=12)
plt.show()

# Visualizing the Test Dataset
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,y_pred,color='blue')
plt.title('Salary vs Experience (Test Data)',size=12)
plt.xlabel('Experience (years)',size=12)
plt.ylabel('Salary',size=12)
plt.show()

