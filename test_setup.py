#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from scikitplot.metrics import plot_confusion_matrix


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


plt.show()




