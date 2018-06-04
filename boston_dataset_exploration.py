import pandas as pd
from sklearn.datasets import load_boston





boston_dataset     = load_boston()
boston_df          = pd.DataFrame(boston_dataset.data)
boston_df.columns  = boston_dataset.feature_names
boston_df['PRICE'] = boston_dataset.target
print(boston_df.head())