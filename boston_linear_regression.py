#!/usr/bin/python3

# importiamo le funzionalita' che useremo
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# carica dataset sulle case di Boston
boston_dataset = load_boston()

# esplora i contenuti del dataset
print( boston_dataset['DESCR'] )

# estraiamo le features (input) e il target (output)
X = boston_dataset.data # dati
y = boston_dataset.target # prezzo

# dividiamo i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

# train li usiamo per addestrare il modello iterando su tutti i dati e correggendo gli errori
# test sono i dati su cui validiamo il modello quando l'addestramento e' finito

# addestriamo il modello sui dati di training
model = LinearRegression()
model.fit(X_train, y_train)


# testiamo il modello sui dati di test
pred_test = model.predict(X_test)
test_score = r2_score(y_test, pred_test)
print('Test score', test_score)














