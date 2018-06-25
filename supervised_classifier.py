#!/usr/bin/python3
import pandas as pd
import numpy as np
np.random.seed(1)   # per avere semrpe lo stesso risultato
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# carichiamo i dati
digits_dataset = load_digits()
print(digits_dataset['DESCR'])
X = digits_dataset.data
y = digits_dataset.target

# dividiamo in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# costruiamo una struttura dati in grado di ospitare tutti i modelli
all_models = [
    {
        'name' : 'dummy',
        'model': DummyClassifier()
    },
    {
        'name' : 'kNN',
        'model': KNeighborsClassifier()
    },
    {
        'name' : 'tree',
        'model': DecisionTreeClassifier()
    },
    {
        'name' : 'logistic',
        'model': LogisticRegression()
    },
    {
        'name' : 'bayes',
        'model': GaussianNB()
    }
]

# addestriamo tutti i modelli e raccogliamo le prestazioni
for m in all_models:

    # addestramento
    print('Addestramento:', m['name'])
    m['model'].fit(X_train, y_train)

    # prestazione sul training set
    pred_train = m['model'].predict(X_train)
    m['train accuracy'] = accuracy_score(y_train, pred_train)

    # prestazione sul test set
    pred_test = m['model'].predict(X_test)
    m['test accuracy']  = accuracy_score(y_test, pred_test)

    # delete model object for compatibility with pandas
    del m['model']

# plots
scores_df = pd.DataFrame(all_models)
print(scores_df)
scores_df.plot.bar(x='name')

plt.show()