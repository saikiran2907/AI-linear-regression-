# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:22:08 2020

@author: NANI
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("student_scores.csv")

features = dataset['Hours'].values
labels = dataset['Scores'].values
type(dataset)

dataset.shape
dataset.ndim

from sklearn.linear_model import LinearRegression

regression = LinearRegression()


features.shape

features = features.reshape(28,1)

features.shape

regression.fit(features, labels)

x = [9]

x = np.array(x)

type(x)

x.shape

x = x.reshape(1,1)

regression.predict(x)
plt.scatter(features, labels)
plt.plot(features, regressor.predict((features)), color = 'yellow')

regressor.coef_

regressor.intercept_





x = int(input("enter the number of hours : "))

type(x)

x=[x]

type(x)

x = np.array(x)

x = x.reshape(1,1)

regressor.predict(x)