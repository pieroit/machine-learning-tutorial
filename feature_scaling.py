#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.datasets import load_iris

# carichiamo dati iris
iris_dataset = load_iris()
print(iris_dataset['DESCR'])
iris_data = iris_dataset.data


# scaliamo i dati
scaler = StandardScaler()
iris_data_scaled = scaler.fit_transform(iris_data)

# grafici del prima e dopo lo scaling
pd.DataFrame(iris_data).plot.scatter(x=0, y=1, title='non scalati')
pd.DataFrame(iris_data_scaled).plot.scatter(x=0, y=1, title='scalati')

# verifica che lo StandardScaler abbia normalizzato i dati
#print('non scalati', np.mean( iris_data[:,0]), np.std(iris_data[:,0] ))
#print('scalati', np.mean( iris_data_scaled[:,0]), np.std(iris_data_scaled[:,0] ))

plt.show()
