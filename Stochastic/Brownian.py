import numpy as np
from math import sqrt

from Functions import vector_times_scalars

def get_norm_numbers(mu, sigma, shape):
    return np.random.normal(mu, sigma, shape)

def standard_brownian(n, k, dt):
    b1 = get_norm_numbers(mu=0, sigma=1, shape=[k, n-1])
    bt = np.zeros(shape=[k,n])
    t = np.arange(0,n*dt,dt)
    
    sqrt_dt = sqrt(dt)
    
    for i in range(k):
        for j in range(1,n):
            bt[i][j] = bt[i][j-1] + sqrt_dt * b1[i][j-1]

    return bt, t

def bridge_brownian(n, k, dt):
    bt, t = standard_brownian(n, k, dt)
    b1 = bt[:,-1]

    wt = bt - vector_times_scalars(t, b1)

    return wt, t

def drift_brownian(n, k, dt, u, s):
    bt, t = standard_brownian(n, k, dt)
    wt = t*u + s*bt

    return wt, t

def geometric_brownian(n, k, dt, a, l):
    bt, t = standard_brownian(n, k, dt)
    wt = np.exp(a*t + l*bt)

    return wt, t
