#!/usr/bin/python3

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import glob

# carichiamo il modello di computer vision preaddestrato, privato degli strati di output
# ci permettera' di convertire le immagini in un vettore a poche dimensioni
# che possiamo utilizzare come input per la nostra porzione di rete
model = VGG16(weights='imagenet', include_top=True)

# carichiamo le immagini
X = []
for img_file in glob.glob('dataset/actors/*.jpg'):

    # utilita' keras per scalare le immagini
    img = image.load_img(img_file, target_size=(224,224) )
    x = image.img_to_array(img)
    X.append(x)

# converti a numpy array
X = np.array(X)

# utilita' keras per preprocessare i canali dei colori
X = preprocess_input(X)

features = model.predict(X)
print(features)

