import numpy as np
import matplotlib.pyplot as plt

from Brownian import standard_brownian

def homogeneous_equation(x_0, u, s, n):
    [bt], [b1], time = standard_brownian(n, 1, 1/n)
    x_t = [ x_0*np.exp( (u-0.5*s**2)*time[i] + s*bt[i] ) for i in range(n)]

    return x_t, b1, time

def euler(x_0, u, s, n, b1):
    dt = 1/n
    sqrt_dt = np.sqrt(dt)
    x = np.zeros(n)

    x[0] = x_0
    for i in range(1,n):
        x[i] = x[i-1] + u*x[i-1]*dt + s*x[i-1]* (sqrt_dt*b1[i-1])

    return x

if __name__ == '__main__':
    x_0, u, s, n = 1, 1.5, 2.5, 500

    x_t, b1, time = homogeneous_equation(x_0, u, s, n)
    x_t_e = euler(x_0, u, s, n, b1)
    
    plt.plot(time, x_t, linewidth=1)
    plt.plot(time, x_t_e, linewidth=1)
    plt.legend(["x_t", "euler"])
    plt.show()
