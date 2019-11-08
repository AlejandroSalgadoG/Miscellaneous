import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

from Functions import vector_times_scalars

def standard_brownian(n, T, k):
    dt = T/n
    b1 = np.random.normal(0, 1, [k, n-1])
    bt = np.zeros(shape=[k,n])
    t = np.arange(0,T,dt)
    
    sqrt_dt = sqrt(dt)
    
    for i in range(k):
        for j in range(1,n):
            bt[i][j] = bt[i][j-1] + sqrt_dt * b1[i][j-1]

    return bt, b1, t

def bridge_brownian(n, k):
    bt, b1, t = standard_brownian(n, 1, k)
    b_end = bt[:,-1]

    wt = bt - vector_times_scalars(t, b_end)

    return wt, t

def drift_brownian(n, T, k, u, s):
    dt = T/n
    bt, b1, t = standard_brownian(n, T, k)
    wt = t*u + s*bt

    return wt, t

def geometric_brownian(n, T, k, a, l):
    dt = T/n
    bt, b1, t = standard_brownian(n, T, k)
    wt = np.exp(a*t + l*bt)

    return wt, t

if __name__ == '__main__':
    simulations, rand_num, time = standard_brownian(n=500, T=1, k=1000)
    #simulations, time = bridge_brownian(n=100, k=1000)
    #simulations, time = drift_brownian(n=100, T=1, k=1000, u=20, s=5)
    #simulations, time = geometric_brownian(n=100, T=1, k=1000, a=0.9, l=0.2)
    
    for simulation in simulations:
        plt.plot(time, simulation, linewidth=1)
    plt.show()
