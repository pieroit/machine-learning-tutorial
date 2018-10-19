#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer


# dati distribuiti secondo power law
x = np.array([
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    5,5,5,5,5,5,5,5,
    10,10,10,10,
    100,100,
    1000
])
x = np.random.power(5, len(x)) * 1000
y = np.random.power(5, len(x))
#x = x + np.random.rand(len(x))
#y = x + np.random.rand(len(x))

data = np.random.power(5, (100, 2))
#data = np.column_stack((x, y))

# scaliamo i dati in vari modi
all_scalings = {
    'A_non_scaled'      : data,
    'B_min_max'         : MinMaxScaler().fit_transform(data),
    'C_standard'        : StandardScaler().fit_transform(data),
    'D_robust'          : RobustScaler().fit_transform(data),
    'E_quantile_uniform': QuantileTransformer(output_distribution='uniform').fit_transform(data),
    'F_quantile_normal' : QuantileTransformer(output_distribution='normal').fit_transform(data)
}

# plot
i = 0
for scaling in sorted(all_scalings.keys()):
    i+=1
    plt.subplot('32' + str(i))
    plt.title(scaling)
    x = all_scalings[scaling][:,0]
    y = all_scalings[scaling][:,1]
    plt.scatter(x=x, y=y)
plt.tight_layout()
plt.show()
