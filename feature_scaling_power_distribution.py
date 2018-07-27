#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.datasets import load_iris

# variabile distribuita secondo power law
data = np.array([
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    5,5,5,5,5,5,5,5,
    10,10,10,10,
    100,100,
    1000
])
data = data.reshape(-1, 1)

plt.figure()

plt.subplot(231)
plt.title('non scalati')
plt.hist(data)

# scaliamo i dati in vari modi

min_max_scaler = MinMaxScaler()
data_min_max_scaled = min_max_scaler.fit_transform(data)
plt.subplot(232)
plt.title('min_max')
plt.hist(data)

standard_scaler = StandardScaler()
data_standard_scaled = standard_scaler.fit_transform(data)
plt.subplot(233)
plt.title('standard')
plt.hist(data_standard_scaled)

robust_scaler = RobustScaler()
data_robust_scaled = robust_scaler.fit_transform(data)
plt.subplot(234)
plt.title('robust')
plt.hist(data_robust_scaled)

quantile_uniform_scaler = QuantileTransformer(output_distribution='uniform')
data_quantile_uniform_scaled = quantile_uniform_scaler.fit_transform(data)
plt.subplot(235)
plt.title('quantile_uniform')
plt.hist(data_quantile_uniform_scaled)

quantile_normal_scaler = QuantileTransformer(output_distribution='normal')
data_quantile_normal_scaled = quantile_normal_scaler.fit_transform(data)
plt.subplot(236)
plt.title('quantile_normal')
plt.hist(data_quantile_normal_scaled)


plt.show()
