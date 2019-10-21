import numpy as np
import matplotlib.pyplot as plt

from Brownian import standard_brownian
from Numerical import homogeneous_equation

def instant_returns(n, x_t):
    r = np.zeros(n)
    for i in range(1,n):
        r[i] = (x_t[i] - x_t[i-1])/x_t[i-1]
    return r

if __name__ == '__main__':
    x_0, u, s, n = 1, 1.5, 2.5, 500

    print("real: u = %f, s = %f" % (u, s))

    for i in range(10):
        [bt], [b1], time = standard_brownian(n, 1, 1/n)
        x_t = homogeneous_equation(x_0, u, s, n, bt, time)

        r_t = instant_returns(n, x_t)
        dt = 1/n
        u = np.mean(r_t) / dt
        s = np.sqrt( np.var(r_t, ddof=0) / dt )
        print("u = %f, s = %f" % (u, s))
