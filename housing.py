# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:22:32 2018

@author: kekishor
"""

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
dataset[dataset['MasVnrType'].isnull()].drop()



