#!/usr/bin/python3

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
np.random.seed(1)
import glob

# carichiamo le immagini
directory = './../actors_dataset/'
img_files = glob.glob(directory + 'pics/*.jpg')
X = []
y = []
for i, img_file in enumerate(img_files):

    try:
        print(i, 'loading', img_file)

        # utilita' keras per scalare le immagini
        img = image.load_img(img_file, target_size=(224,224) )
        x = image.img_to_array(img)
        X.append(x)

        # estrai output desiderato dal nome del file
        if 'female_' in img_file:
            y.append( [0] )
        else:
            y.append( [1] )

    except Exception:
        pass

# converti a numpy array
X = np.array(X)

# utilita' keras per preprocessare i canali dei colori
X = preprocess_input(X)

# carichiamo il modello di computer vision preaddestrato, privato degli strati di output
# ci permettera' di convertire le immagini in un vettore a poche dimensioni
# che possiamo utilizzare come input per la nostra porzione di rete
pretrained_model = VGG16(weights='imagenet', include_top=True)
X = pretrained_model.predict(X, verbose=1)

print(np.shape(X))
print(np.shape(y))

# salva immagini vettorizzate e output desiderato
np.savetxt(directory + 'X.txt', X)
np.savetxt(directory + 'y.txt', y)
