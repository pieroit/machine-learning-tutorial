
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from scikitplot.metrics import plot_confusion_matrix

from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense


# PANDAS

df = pd.DataFrame({
    'altezza': [180, 200, 160, 175],
    'peso'   : [100, 110, 80, 100],
    'sport'  : ['rugby', 'basket', 'rugby', 'rugby']
})

print( df.head() )
print( df['sport'].unique() )
print( df['altezza'].describe() )

df.plot.scatter(x='altezza', y='peso')
df.plot.bar()


# SCIKIT

dataset = load_wine()

X = dataset['data']
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print( 'accuracy', accuracy_score(y_test, predictions) )
plot_confusion_matrix(y_test, predictions)


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


plt.show()




