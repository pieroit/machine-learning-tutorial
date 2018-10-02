import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
import tensorflow_hub as hub
import tensorflow as tf


df = pd.read_csv('data/winemag-data-130k-v2.csv')[:10000]
print(df.columns)

df.dropna(inplace=True)

X = df['description'].values
y = df['price'].values

count_vectorizer = TfidfVectorizer(stop_words='english')
X = count_vectorizer.fit_transform(X)

#X = PCA(n_components=100).fit_transform(X.todense())

#USE = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/2')
#with tf.Session() as session:
#  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#  X = session.run(USE(X))

X_train, X_test, y_train, y_yest = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
print('Train r2', r2_score(y_train, p_train))
p_test  = model.predict(X_test)
print('Test r2', r2_score(y_yest, p_test))

