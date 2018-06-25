#!/usr/bin/python3
# importiamo le funzionalita'
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scikitplot.estimators import plot_learning_curve

# carica dataset
dataset = load_digits()
X = dataset.data
y = dataset.target


print(np.shape(X))

# costruisci il modello
model = LogisticRegression()

# ripeti l'addestramento con un numero sempre maggiore di dati
# misurando la performance su train e test set
plot_learning_curve(model, X, y)

# mostra il grafico della curva di apprendimento
plt.show()


