#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# carica i dati
iris_dataset = load_iris()
print(iris_dataset['DESCR'])
data = iris_dataset.data

# lancia il clustering scegliendo il numero di cluster richiesti
model = KMeans(n_clusters=3)
model.fit(data)

# ottieni punteggio di dispersione
print('Inerzia', model.inertia_)
# ottieni il cluster assegnato a ogni punto
cluster_assignations = model.predict(data)
print('cluster di appartenenza dei dati')
print(cluster_assignations)
# ottieni le coordinate dei cluster
cluster_centers  = model.cluster_centers_
print('posizione dei cluster')
print(cluster_centers)

# grafico
dataframe = pd.DataFrame(data)
dataframe.columns =['lunghezza sepalo', 'larghezza sepalo', 'lunghezza petalo', 'larghezza petalo']
dataframe.plot.scatter(x='lunghezza sepalo', y='larghezza sepalo', c=cluster_assignations)
dataframe.plot.scatter(x='lunghezza petalo', y='larghezza petalo', c=cluster_assignations)

plt.show()