# ------------------------> Importing the required Libraries -----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------> Getting the data ---------------------------------------------------

dataset = pd.read_csv('train.csv')

# ------------------------> Preprocessing the data ---------------------------------------------

dataset.drop('Alley', axis = 1, inplace = True)
dataset.drop('Id', axis=1, inplace = True)

# - (Checking and evaluating null values) - 

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = np.nan, strategy = 'median')
imputer.fit(dataset['LotFrontage'].values.reshape(-1, 1))
dataset['LotFrontage'] = imputer.transform(dataset['LotFrontage'].values.reshape(-1, 1))

# ------> Dropping some columns with nan values ---
dataset.drop(['PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)
dataset.drop('FireplaceQu', axis = 1, inplace=True)
sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis')

dataset.loc[dataset['MasVnrType'].isnull(), 'MasVnrType'] = 'None'
dataset.loc[dataset['MasVnrArea'].isnull(), 'MasVnrArea'] = 0
dataset.dropna(axis = 0, inplace = True)

#dataset.drop('Utilities', axis = 1, inplace = True)
#dataset.drop('Street', axis=1, inplace = True)
#len(dataset[dataset['LandContour'] == 'Lvl'])
#dataset.drop('LandContour', axis = 1, inplace = True)
dataset_filter = dataset

from sklearn.preprocessing import LabelEncoder
for i in (dataset.columns):
    if type(dataset[i][1]) == str:
        encoder = LabelEncoder()
        dataset[i] = encoder.fit_transform(dataset[i])

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
        
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)

score.mean()





