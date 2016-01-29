## Kaggle Display Advertising Challenge
##
## Author: Ashwin Raman
##
## Last Edited: 29th Jan 2016
########################################

import os
import numpy as ny

# Data Extraction from file into data frame
import pandas as pd

# Model - K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Preprocessing data 
# 1. Removing nulls (NaNs)
# 2. Encoding categorical variables with numerals
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

# Crossvalidation - Spliting training data into training 
# 					and test and measuring performance
from sklearn.cross_validation import train_test_split
from sklearn import metrics 

data = pd.read_csv('/home/ashwin/Academics/Machine-Learning/Display-Ads/Data/dac_sample.txt', sep='\t', header=None)
#print data.head()

enc = LabelEncoder()

categorical_data = data.ix[:, 14:]
numerical_data = data.ix[:, 1:13]

#print categorical_data.head()
#print numerical_data.head()
#print categorical_data.ix[:, 14]

#populate feature array processed
X = numerical_data
for i in range(14, 40):
	test = enc.fit_transform(categorical_data.ix[:, i])
	test = pd.DataFrame(test, columns=[i])
	X = X.join(test)
	
#print X.head()

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

X = pd.DataFrame(imp.fit_transform(X), columns=range(1, 40))
y = data.ix[:, 0]

#print X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print metrics.accuracy_score(y_test, y_pred)