#!/usr/bin/python3
# importiamo le funzionalita' che useremo
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scikitplot.estimators import plot_feature_importances

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

# testiamo il modello sui dati di test
pred_test = model.predict(X_test)
test_score = r2_score(y_test, pred_test)
print('Test score', test_score)

# scopriamo a quali feature e' stato dato maggior peso
model.feature_importances_ = model.coef_
plot_feature_importances(model, feature_names=boston_dataset.feature_names)
plt.show()













