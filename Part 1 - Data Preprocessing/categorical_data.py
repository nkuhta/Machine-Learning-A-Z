# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#  find the column mean values for specified X columns
imputer = imputer.fit(X[:, 1:3])
#  replace nan values with means
X[:, 1:3] = imputer.transform(X[:, 1:3])


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


print(X)
print(y)
