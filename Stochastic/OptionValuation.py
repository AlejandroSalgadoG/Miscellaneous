import time
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=1000, precision=3, suppress=True)

from Brownian import standard_brownian
from Numerical import homogeneous_equation

def montecarlo(x_0, r, s, n, T, m, w, k):
    fcall = []

    for i in range(w):
        bt, b1, time = standard_brownian(n, T, m)
        x_t = homogeneous_equation(x_0, r, s, n, m, bt, time)
        fcall_T = np.mean( np.maximum(x_t[:,-1] - k, np.zeros(m)) )
        fcall.append( np.exp(-r) * fcall_T )
        print("Montecarlo: %.2f%%" % (i/w*100), end='\r')
    print()

    return np.mean(fcall), fcall

def black_scholes(x_0, r, s, k, T):
    d1 = ( np.log(x_0/k) + (r+s**2/2)*T ) / (s*np.sqrt(T))
    d2 = d1 - s*np.sqrt(T)
    return x_0 * norm.cdf(d1) - k*np.exp(-r*T)*norm.cdf(d2)

def binomial_tree(x_0, r, s, k, T, n):
    dt = T/n
    u = np.exp(s*np.sqrt(dt))
    d = np.exp(-s*np.sqrt(dt))
    p = (np.exp(r*dt) - d)/(u - d)
    f = np.zeros([n,n])
    n -= 1

    for i in range(n,-1,-1):
        for j in range(i,-1,-1):
            if i == n:
                f[n,j] = max(x_0*(u**j)*(d**(n-j)) - k, 0)
            else:
                f[i,j] = np.exp(-r*dt) * (p*f[i+1,j+1] + (1-p)*f[i+1,j])
        print("Binomial tree: %.2f%%" % ((n-i)/n * 100), end='\r')
    print()
    return f[0,0]

def finite_differences(s, r, k, T, m):
    ds = 2*k / m
    dt = 0.9 / (s**2 * m**2)
    n = int(T/dt) + 1
    dt = T / n

    S = np.zeros(m+1)
    f = np.zeros([m+1, n+1])

    for i in range(m+1):
        S[i] = i*ds
        f[i,0] = np.maximum(S[i]-k, 0)

    for k in range(1,n+1):
        for i in range(1,m):
            delta = (f[i+1, k-1] - f[i-1,k-1]) / (2*ds)
            gamma = (f[i+1, k-1] - 2*f[i,k-1] + f[i-1,k-1]) / ds**2
            theta = -0.5 * s**2 * S[i]**2 * gamma - r * S[i] * delta + r * f[i,k-1]

            f[i,k] = f[i,k-1] - dt * theta

        f[0,k] = f[0,k-1] * (1 - r * dt)
        f[m,k] = 2 * f[m-1,k] - f[m-2,k]
        print("Finite diff: %.2f%%" % (k/n*100), end='\r')
    print()

    [ans_pos] = np.argwhere(f[:,0]>0)[0]
    return f[ans_pos-1,-1]

if __name__ == '__main__':
    x_0, u, s, n, T, k = 100, 0.05, 0.3, 500, 1, 100

    fcall = black_scholes(x_0, u, s, k, T)
    print(fcall)

    m, w = 1000, 100
    fcall_hat, fcalls = montecarlo(x_0, u, s, n, T, m, w, k)
    print(fcall_hat)

    fcall_hat = binomial_tree(x_0, u, s, k, T, 500)
    print(fcall_hat)

    fcall_hat = finite_differences(s, u, k, T, 500)
    print(fcall_hat)
