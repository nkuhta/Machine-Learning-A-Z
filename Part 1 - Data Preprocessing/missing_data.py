# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#print(X)
#print()

#  Missing data (replace with column mean, median, KNN)
#  boolean check which columns have missing data
print(dataset.isnull().any())
#print()

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
#print(X[:,dataset.isnull().any()[:-1]]-X[:,1:3])
#  Generalizing for more than 3 columns of independent features,
#  No need to find what columns have null values,
#  rightmost column is dependent set of values in data file
#imputer = imputer.fit(X[:, dataset.isnull().any()[:-1]])
#X[:,dataset.isnull().any()[:-1]] = imputer.transform(X[:,dataset.isnull().any()[:-1]])

#print(X)
#print()
