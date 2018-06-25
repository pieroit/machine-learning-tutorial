#!/usr/bin/python3
# importiamo le funzionalita' per calcolare gli errori di un classificatore
from sklearn.metrics import accuracy_score, confusion_matrix

# importiamo le librerie per i grafici
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix

# assumiano di avere un array con le risposte corrette
y_test = [
    "basket",
    "rugby",
    "rugby",
    "calcio",
    "basket",
    "basket",
    "calcio"
]

# assumiano di avere un array con le risposte predette dal modello
y_pred = [
    "basket",
    "rugby",
    "calcio",
    "basket",
    "basket",
    "rugby",
    "calcio"
]

# calcoliamo l'accuratezza delle risposte
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy', accuracy)

# entriamo nel dettaglio con la matrice di confusione
confusion = confusion_matrix(y_test, y_pred)
print(confusion)

# facciamo un grafico della matrice di confusione
plot_confusion_matrix(y_test, y_pred)

plt.show()
