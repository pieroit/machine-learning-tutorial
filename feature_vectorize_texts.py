#!/usr/bin/python3
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# immaginiamo una serie di testi (preprocessati)
data = [
    'ciao amico gatto',
    'ciao amico cane',
    'ciao ciao'
]

# vettorizziamoli
text_vectorizer = CountVectorizer()
vectorized_data = text_vectorizer.fit_transform(data)

print('rappresentazione sparsa')
print(vectorized_data)

print('rappresentazione densa')
print(vectorized_data.todense())

print('dai vettori ritorniamo al testo')
print( text_vectorizer.inverse_transform( vectorized_data ) )