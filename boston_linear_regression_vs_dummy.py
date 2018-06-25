#!/usr/bin/python3
# importiamo le funzionalita' che useremo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(1)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

# carica dataset sulle case di Boston
boston_dataset = load_boston()

# esplora i contenuti del dataset
print(boston_dataset['DESCR'])

# estraiamo le features (input) e il target (output)
X = boston_dataset.data
y = boston_dataset.target

# dividiamo i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

# addestriamo il modello sui dati di training
model = LinearRegression()
model.fit(X_train, y_train)
# verifichiamo l'apprendimento sui dati di training
pred_train = model.predict(X_train)
train_score = mean_squared_error(y_train, pred_train)
print('Train error', train_score)
# testiamo il modello sui dati di test
pred_test = model.predict(X_test)
test_score = mean_squared_error(y_test, pred_test)
print('Test error', test_score)

# prestazioni del modello dummy (baseline)
dummy = DummyRegressor()
dummy.fit(X_train, y_train)
# dummy sul training set
pred_train_dummy = dummy.predict(X_train)
dummy_train_score = mean_squared_error(y_train, pred_train_dummy)
print('Dummy train error', dummy_train_score)
# dummy sul test set
pred_test_dummy  = dummy.predict(X_test)
dummy_test_score = mean_squared_error(y_test, pred_test_dummy)
print('Dummy test error', dummy_test_score)

# prepariamo dati per il grafico di confronto
report = {
    'train': [dummy_train_score, train_score],
    'test' : [dummy_test_score, test_score],
    'model': ['dummy', 'regression']
}
report_df = pd.DataFrame(report)
report_df = report_df.set_index(report_df['model'])
report_df.plot.bar(title='model error')

plt.show()
