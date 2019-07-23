import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def get_norm_numbers(mu, sigma, shape):
    return np.random.normal(mu, sigma, shape)

def brownian_movement(n, k, dt):
    et = get_norm_numbers(mu=0, sigma=1, shape=[k, n-1])
    bt = np.zeros(shape=[k,n])
    t = np.arange(0,n*dt,dt)
    
    sqrt_dt = sqrt(dt)
    
    for i in range(k):
        for j in range(1,n):
            bt[i][j] = bt[i][j-1] + sqrt_dt * et[i][j-1]

    return bt, t

simulations, time = brownian_movement(n=100, k=1000, dt=0.01)

for simulation in simulations:
    plt.plot(time, simulation, linewidth=1)
plt.show()
