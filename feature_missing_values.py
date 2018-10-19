#!/usr/bin/python3
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# immaginiamo un dataset con valori mancanti
data = [
    [3, 1],
    [3, 1],
    [2, 0],
    [np.nan, np.nan]
]

# sostituiamo i valori mancanti con la media della colonna
imputer = SimpleImputer(strategy='mean')
data_with_mean_strategy = imputer.fit_transform(data)
print('valori mancanti sostituiti dalla media')
print(data_with_mean_strategy)

# sostituiamo i valori mancanti con la moda della colonna
imputer = SimpleImputer(strategy='most_frequent')
data_with_mode_strategy = imputer.fit_transform(data)
print('valori mancanti sostituiti dalla moda')
print(data_with_mode_strategy)

# se i dati sono tanti, possiamo direttamente cancellare i record
data_df = pd.DataFrame(data)
data_df = data_df.dropna()
print('buttiamo via gli esempi con valori mancanti')
print(data_df.values)