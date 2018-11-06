# Data Preprocessing Template

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#  Missing data (replace with column mean, median, KNN)
#  boolean check which columns have missing data
#print(dataset.isnull().any())
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


#  Encode categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
#  identify first column as categorical
X[:,0] = labelencoder_X.fit_transform(X[:,0])
#  use one hot encoder to makes N categorical columns
onehotencoder = OneHotEncoder(categorical_features = [0])
# replace X categorical columns with N categorical columns of (0,1)
X = onehotencoder.fit_transform(X).toarray()
#  just need labelencoder for 0,1 for the independent y column
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#  Splitting dataset into (test, training) subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

print(X_train)

#  Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print(X_train)
