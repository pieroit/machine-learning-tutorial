#!/usr/bin/python3
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# carica boston dataset
boston_dataset     = load_boston()
print(boston_dataset['DESCR'])

# converti a dataframe pandas e assegna nomi alle colonne
boston_df          = pd.DataFrame(boston_dataset.data)
boston_df.columns  = boston_dataset.feature_names
boston_df['PRICE'] = boston_dataset.target

# aggiungi variabile categorica
def price_category(p):
    if p > 25:
        return 'ALTO'
    if p < 15:
        return 'BASSO'
    return 'MEDIO'
boston_df['PRICE_CATEGORY'] = boston_df['PRICE'].apply(price_category)

# stampa informazioni
print(boston_df.head())
print(boston_df.describe())

# grafici esplorativi
boston_df['PRICE_CATEGORY'].value_counts().plot.pie(title='Distribuzione fasce di prezzo')
plt.figure()
boston_df['PRICE'].plot.hist(title='Distribuzione prezzi')
boston_df.plot.scatter(x='CRIM', y='PRICE', title='Prezzo vs Tasso criminalita')
boston_df.plot.scatter(x='INDUS', y='AGE', c='PRICE', title='Prezzo vs Zona industriale e vecchie costruzioni')
scatter_matrix(boston_df[ ['RM', 'DIS', 'RAD', 'PRICE'] ])
plt.show()