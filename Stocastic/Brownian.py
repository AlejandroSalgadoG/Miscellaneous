import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

n = 100
k = 1000

def get_norm_numbers(mu, sigma, shape):
    return np.random.normal(mu, sigma, shape)

et = get_norm_numbers(mu=0, sigma=1, shape=[k, n])
bt = np.zeros(shape=[k,n])
time = np.arange(n)

sqrt_dt = sqrt(1/n)

for i in range(k):
    for t in time[1:]:
        bt[i][t] = bt[i][t-1] + sqrt_dt * et[i][t-1]
    plt.plot(time, bt[i], linewidth=1)

plt.show()
