#!/usr/bin/python3
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# random seed
np.random.seed(1)

# caricamento dati
iris_dataset = datasets.load_iris()
print iris_dataset['DESCR']
X = iris_dataset.data
y = iris_dataset.target

# separazione dei dati in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# creazione del modello
model = GaussianNB()

# addestramento del modello
model.fit(X_train, y_train)

# predizione su dati non utilizzati nell'addestramento
y_predicted = model.predict(X_test)

# misura di accuratezza delle predizioni
print '======= Validation ========='
print 'Confusion Matrix:\n', confusion_matrix(y_test, y_predicted)
print 'Accuracy:\n', accuracy_score(y_test, y_predicted)
print 'Other metrics:\n', classification_report(y_test, y_predicted)

