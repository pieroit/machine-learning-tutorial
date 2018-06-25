#!/usr/bin/python3
from sklearn.feature_extraction import DictVectorizer
import pandas as pd

# immaginiamo dei dati con una colonna categorica
df = pd.DataFrame({
    'altezza': [180, 200, 160, 175],
    'peso'   : [100, 110, 80, 100],
    'sport'  : ['rugby', 'basket', 'rugby', 'soccer']
})
# per utilizzare facilmente DictVectorizer passiamo per pandas
# e convertiamo tutti i record in dizionari
df_dictionary = df.to_dict(orient='records')
print(df_dictionary)

# vettorizziamo tutto il dataset
vectorizer = DictVectorizer(sparse=False)
vectorized_data = vectorizer.fit_transform(df_dictionary)

print('dati vettorizzati')
print(vectorized_data)

print('riporta vettori a categorie')
print( vectorizer.inverse_transform( vectorized_data ) )


