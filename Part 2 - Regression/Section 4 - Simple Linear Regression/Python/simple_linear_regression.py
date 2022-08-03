# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 18:56:10 2022

@author: Michele Nardini
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#     YearsExperience    Salary
# 0               1.1   39343.0
# 1               1.3   46205.0
# 2               1.5   37731.0
# 3               2.0   43525.0
# 4               2.2   39891.0
# 5               2.9   56642.0
# 6               3.0   60150.0
# 7               3.2   54445.0
# 8               3.2   64445.0
# 9               3.7   57189.0
# 10              3.9   63218.0
# 11              4.0   55794.0
# 12              4.0   56957.0
# 13              4.1   57081.0
# 14              4.5   61111.0
# 15              4.9   67938.0
# 16              5.1   66029.0
# 17              5.3   83088.0
# 18              5.9   81363.0
# 19              6.0   93940.0
# 20              6.8   91738.0
# 21              7.1   98273.0
# 22              7.9  101302.0
# 23              8.2  113812.0
# 24              8.7  109431.0
# 25              9.0  105582.0
# 26              9.5  116969.0
# 27              9.6  112635.0
# 28             10.3  122391.0
# 29             10.5  121872.0

dataset = pd.read_csv('Salary_Data.csv')

print(dataset)

# Qui andremo a prendere i dati in un determinato intervallo.
# Con : andremo a prendere tutte le righe/colonne.

# x sarà la matrice delle caratteristiche del dataset
x = dataset.iloc[:, :-1].values

# y sarà la variabile dipendente, ovvero quella di cui vorremo 
# prevedere il cambiamento.
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

print(regressor.predict([[0]]))