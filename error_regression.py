#!/usr/bin/python3
# importiamo le funzionalita' per calcolare gli errori di un regressore
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# assumiano di avere un array con le risposte corrette
y_test = np.array( [2, 1, 0, 3, 2, -1, 5] )

# assumiano di avere un array con le risposte predette dal modello
y_pred = np.array( [1, 1, -1, 3, 0, 0, 4] )

# calcoliamo la correttezza delle risposte
mse = mean_squared_error(y_test, y_pred)
print('Mean square error', mse)

r2 = r2_score(y_test, y_pred)
print('R2 score', r2)

residuals = y_pred - y_test
plt.scatter(y_test, residuals)
plt.xlabel('True')
plt.ylabel('Error')

plt.show()
