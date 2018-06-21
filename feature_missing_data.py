from sklearn.preprocessing import Imputer
import pandas as pd

# immaginiamo un dataset con valori mancanti
data = [
    [2, 0],
    [2, 0],
    [3, 1],
    [None, None]
]

# sostituiamo i valori mancanti con la media della colonna
imputer = Imputer(strategy='mean')
data_with_mean_strategy = imputer.fit_transform(data)
print('valori mancanti sostituiti dalla media')
print(data_with_mean_strategy)

# sostituiamo i valori mancanti con la moda della colonna
imputer = Imputer(strategy='most_frequent')
data_with_mode_strategy = imputer.fit_transform(data)
print('valori mancanti sostituiti dalla moda')
print(data_with_mode_strategy)

# se i dati sono tanti, possiamo direttamente cancellare i record
data_df = pd.DataFrame(data)
data_df = data_df.dropna()
print('buttiamo via gli esempi con valori mancanti')
print(data_df.values)