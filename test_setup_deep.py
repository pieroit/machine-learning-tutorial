#!/usr/bin/python3

from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense


# KERAS/TENSORFLOW

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test  = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_train /= 255
X_test  /= 255

y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

model = Sequential([
    Dense(20, input_shape=(784,), activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1)   # metti epochs=10 per apprendimento completo

score = model.evaluate(X_test, y_test, verbose=0)
print('accuracy', score[1])





