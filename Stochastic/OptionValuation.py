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
        print("%.2f%%\r" % (i/w*100), end='')
    print()

    return np.mean(fcall), fcall

def black_scholes(x_0, r, s, k, T):
    d1 = ( np.log(x_0/k) + (r+s**2/2)*T ) / (s*np.sqrt(T))
    d2 = d1 - s*np.sqrt(T)
    return x_0 * norm.cdf(d1) - k*np.exp(-r*T)*norm.cdf(d2)

def binomial_tree(x_0, r, s, k, n):
    dt = 1/n
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
    return f[0,0]

def finite_differences(vol, int_rate, strike, expiration, nas):
    ds = 2*strike / nas
    dt = 0.9 / (vol**2 * nas**2) # for stability
    nts = int(expiration/dt) + 1
    dt = expiration / nts

    S = np.zeros(nas+1)
    f = np.zeros([nas+1, nts+1])

    for i in range(nas+1):
        S[i] = i*ds
        f[i,0] = np.maximum(S[i]-strike, 0)

    for k in range(1,nts+1):
        for i in range(1,nas):
            delta = (f[i+1, k-1] - f[i-1,k-1]) / (2*ds)
            gamma = (f[i+1, k-1] - 2*f[i,k-1] + f[i-1,k-1]) / ds**2
            theta = -0.5 * vol**2 * S[i]**2 * gamma - int_rate * S[i] * delta + int_rate * f[i,k-1]

            f[i,k] = f[i,k-1] - dt * theta

        f[0,k] = f[0,k-1] * (1 - int_rate * dt)
        f[nas,k] = 2 * f[nas-1,k] - f[nas-2,k]
        print("%.2f%%" % (k/nts*100), end='\r')

    ans_pos = np.argwhere(f[:,0]>0)[0]
    return f[ans_pos-1,-1]

if __name__ == '__main__':
    x_0, u, s, n, T, k = 100, 0.05, 0.3, 500, 1, 100

    fcall = black_scholes(x_0, u, s, k, T)
    print(fcall)

#    m, w = 1000, 100
#    fcall_hat, fcalls = montecarlo(x_0, u, s, n, T, m, w, k)
#    print(fcall_hat)

#    fcall_hat = binomial_tree(x_0, u, s, k, 5000)
#    print(fcall_hat)

#    fcall_hat = finite_differences(s, u, k, T, 500)
#    print(fcall_hat)
