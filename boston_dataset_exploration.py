import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# carica boston dataset
boston_dataset     = load_boston()
print(boston_dataset['DESCR'])

# converti a dataframe pandas e assegna nomi alle colonne
boston_df          = pd.DataFrame(boston_dataset.data)
boston_df.columns  = boston_dataset.feature_names
boston_df['PRICE'] = boston_dataset.target

# stampa informazioni
print(boston_df.head())
print(boston_df.describe())

# plots
boston_df['PRICE'].plot.hist()
boston_df.plot.scatter(x='CRIM', y='PRICE')
plt.show()