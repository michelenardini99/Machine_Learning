# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:23:37 2022

@author: Michele Nardini
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset:
    
# Country,  Age,   Salary,   Purchased
# France,   44,    72000,    No
# Spain,    27,    48000,    Yes
# Germany,  30,    54000,    No
# Spain,    38,    61000,    No
# Germany,  40,         ,    Yes
# France,   35,    58000,    Yes
# Spain,      ,    52000,    No
# France,   48,    79000,    Yes
# Germany,  50,    83000,    No
# France,   37,    67000,    Yes


dataset = pd.read_csv('Data.csv')

# Qui andremo a prendere i dati in un determinato intervallo.
# Con : andremo a prendere tutte le righe/colonne.

# x sarà la matrice delle caratteristiche del dataset
x = dataset.iloc[:, :-1].values

# y sarà la variabile dipendente, ovvero quella di cui vorremo 
# prevedere il cambiamento.
y = dataset.iloc[:, -1].values

print(x)

print(y)

#Ora andremo a sostituire i valori mancanti
# tramite la classe SimpleImputer che richiederà come primo 
# valore, i valori da sostituire e come secondo la strategia di
# sostituzione.
# Con 'mean' si sostituiscono i valori mancanti utilizzando 
# la media lungo ciascuna colonna. 
# Può essere utilizzato solo con dati numerici.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

# Questo stimatore consente di trasformare separatamente diverse
# colonne o sottoinsiemi di colonne dell'input e le caratteristiche 
# generate da ciascun trasformatore verranno concatenate per formare 
# un unico spazio delle caratteristiche.
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

#ColumnTransformer parametri:
    #lista trasformatori:
        # nome: serve a impostare il trasformatore
        # trasformatore: viene specificato il trasformatore
        # colonne: indicherà l'indice della colonna che subirà la trasformazione
    # reminder: Specificando remainder='passthrough', tutte le colonne 
    # rimanenti che non sono state specificate in transformersverranno 
    # automaticamente passate.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)

# Codifica le etichette di destinazione con un valore compreso tra
#  0 e n_classes-1.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

# Splitting the dataset into Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)

# A seguito di queste operazioni avremo le caratteristiche del
# dataset con i valori scalati























