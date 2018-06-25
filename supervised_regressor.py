#!/usr/bin/python3
import pandas as pd
import numpy as np
np.random.seed(1)   # per avere semrpe lo stesso risultato
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# carichiamo i dati
digits_dataset = load_boston()
print(digits_dataset['DESCR'])
X = digits_dataset.data
y = digits_dataset.target

# scaliamo i dati (sorpresa!)
#scaler = RobustScaler()
#X = scaler.fit_transform(X)

# dividiamo in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# costruiamo una struttura dati in grado di ospitare tutti i modelli
all_models = [
    {
        'name' : 'dummy',
        'model': DummyRegressor()
    },
    {
        'name' : 'linear',
        'model': LinearRegression()
    },
    {
        'name' : 'network',
        'model': MLPRegressor(hidden_layer_sizes=[20, 10], max_iter=10000, learning_rate='adaptive') # neural network!
    }
]

# addestriamo tutti i modelli e raccogliamo le prestazioni
for m in all_models:

    # addestramento
    print('Addestramento:', m['name'])
    m['model'].fit(X_train, y_train)

    # prestazione sul training set
    pred_train = m['model'].predict(X_train)
    m['train r2'] = r2_score(y_train, pred_train)

    # prestazione sul test set
    pred_test = m['model'].predict(X_test)
    m['test r2']  = r2_score(y_test, pred_test)

    # delete model object for compatibility with pandas
    del m['model']

# plots
scores_df = pd.DataFrame(all_models)
print(scores_df)
scores_df.plot.bar(x='name')

plt.show()