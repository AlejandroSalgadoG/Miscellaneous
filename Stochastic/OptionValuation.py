import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=1000, precision=2)

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

def black_scholes(x_0, r, s, k):
    d1 = ( np.log(x_0/k) + (r+s**2/2) ) / s
    d2 = d1 - s
    return x_0 * norm.cdf(d1) - k*np.exp(-r)*norm.cdf(d2)

def binomial_tree(x_0, r, s, n):
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

def finite_differences(x_0, r, s, m, T, k):
    x_max = x_0 * 3
    dx = x_max / m
    n = int( T*s**2*m**2 ) + 1
    dt = T/n

    assert(dt <= (dx**2/ (s*x_max)**2 ))

    a = lambda j: dt/(r*dt+1) * (0.5*s**2*j**2 - 0.5*r*j)
    b = lambda j: dt/(r*dt+1) * (1/dt - s**2*j**2)
    c = lambda j: dt/(r*dt+1) * (0.5*s**2*j**2 + 0.5*r*j)

    f = np.zeros([n+1, m+1])
    f[:, m] = np.maximum(x_max-k, 0)
    f[n, :] = [np.maximum(j*dx-k, 0) for j in range(m+1)]

    for i in range(n-1,-1,-1):
        for j in range(m-1,0,-1):
            f[i,j] = a(j)*f[i+1,j-1] + b(j)*f[i+1,j] + c(j)*f[i+1,j+1]
        print("%d\r" % i, end='')


    print(f)

if __name__ == '__main__':
    x_0, u, s, n, T, m, w, k = 100, 0.05, 0.3, 500, 1, 100, 100, 100

#    fcall = black_scholes(x_0, u, s, k)
#    print(fcall)
#
#    fcall_hat, fcalls = montecarlo(x_0, u, s, n, T, m, w, k)
#    print(fcall_hat)
#
#    fcall_hat = binomial_tree(x_0, u, s, n)
#    print(fcall_hat)

    fcall_hat = finite_differences(x_0, 0.05, 0.2, 20, 1, k)
    print(fcall_hat)