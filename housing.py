
# ------------------------> Importing the required Libraries -----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

# ------------------------> Getting the data ---------------------------------------------------

dataset = pd.read_csv('train.csv')
dataset1 = pd.read_csv('test.csv')
# ------------------------> Preprocessing the data ---------------------------------------------

column = []
from sklearn.preprocessing import LabelEncoder
for i in dataset.columns:
    if type(dataset[i][1]) == str:
#        encoder = LabelEncoder()
#        dataset[i] = encoder.fit_transform(dataset[i])
        column.append(i)

X = dataset.iloc[:, :-1]
X = dataset
y = dataset.iloc[:, -1]

X = pd.get_dummies(X, drop_first = True)

from sklearn.preprocessing import Imputer
for i in X.columns:
    if X[i].isnull().any():
        imputer = Imputer(missing_values = np.nan, strategy = 'mean')
        imputer.fit(X[i].values.reshape(-1, 1))
        X[i] = imputer.transform(X[i].values.reshape(-1, 1))

# -----------------> Splitting the data ---------------------
        
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# -----------------> Applying Random Forest ----------------

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)

# -----------------> Using Cross val Score -----------------------

from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator = regressor, X = X, y = y, cv = 10)

score.mean()

#feature = []
#for i in range(len(regressor.feature_importances_)):
#    feature.append([i, regressor.feature_importances_[i]])
#    
#feature.sort(key = lambda x: x[1], reverse = True)
#
##X = X.iloc[:, [i[0] for i in feature[0: 16: 1]]]
#
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#
#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 100, random_state = 0, n_jobs=-1)
#regressor.fit(X_train, y_train)
#
#from sklearn.model_selection import cross_val_score
#score = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
#
#score.mean()

y_pred = regressor.predict(X.iloc[:, :-1])



    
    

