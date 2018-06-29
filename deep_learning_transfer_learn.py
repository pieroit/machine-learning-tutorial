#!/usr/bin/python3

import numpy as np
np.random.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten

# carichiamo le immagini vettorizzate da VGG16 e l'output desiderato
directory = './../actors_dataset/'
X = np.loadtxt(directory + 'X.txt')
y = np.loadtxt(directory + 'y.txt')

# definiamo l'architettura della rete
dropout_rate = 0.7
model = Sequential([
    #Flatten(input_shape=(7,7,512)),
    #Dense(5000, activation='relu'),
    Dense(500, input_shape=(1000,), activation='relu'),
    Dropout(dropout_rate),
    Dense(1, activation='sigmoid')
])

# definiamo misure di errore e performance
model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='binary_crossentropy')

# addestriamo e testiamo la rete
model.fit(X, y,
          validation_split=0.3,
          epochs=20,
          #class_weight={0: 0.8, 1:0.2}
)




