#!/usr/bin/python3

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# immaginiamo dei dati con una colonna categorica (peso dell'atleta, altezza, sport)
X = [
    [110, 1.70, 'rugby'],
    [100, 1.90, 'basket'],
    [120, 1.90, 'rugby'],
    [ 70, 1.60, 'soccer'],
]

transformer = ColumnTransformer(
    [
        [ 'sport_vector', OneHotEncoder(), [2] ] # one hot encoding per la colonna di indice 2
        # qui si possono specificare altri trasformatori per altre colonne
    ],
    remainder='passthrough'
)

X = transformer.fit_transform(X)

print('one hot encoding per la colonna "sport"')
print(X)



