#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd
from scikitplot.clustering import plot_elbow_curve
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# carica i dati
iris_dataset = load_iris()
print(iris_dataset['DESCR'])
data = iris_dataset.data

# costruisci oggetto per il clustering
model = KMeans()

# lancia clustering tante volte con numero di cluster crescente
plot_elbow_curve(model, data)

plt.show()