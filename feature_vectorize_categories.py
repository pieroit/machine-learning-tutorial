#!/usr/bin/python3
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
import pandas as pd

# immaginiamo dei dati con una colonna categorica
df = pd.DataFrame({
    'altezza': [180, 200, 160, 175],
    'peso'   : [100, 110, 80, 100],
    'sport'  : ['rugby', 'basket', 'rugby', 'soccer']
})


transformer = ColumnTransformer(
    [
        [ 'sport_vector', OneHotEncoder(), ['sport'] ]
        # qui si possono specificare altri trasformatori per altre colonne
    ],
    remainder='passthrough'
)

vectorized_data = transformer.fit_transform(df)

print('one hot encoding per la colonna "sport"')
print(vectorized_data)



