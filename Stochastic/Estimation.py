import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

from Brownian import standard_brownian
from Numerical import homogeneous_equation

def instant_returns(n, x_t):
    r = np.zeros(n)
    for i in range(1,n):
        r[i] = (x_t[i] - x_t[i-1])/x_t[i-1]
    return r

if __name__ == '__main__':
    x_0, u, s, n, k, alpha = 1, 1.5, 2.5, 500, 500, 0.05

    print("real: u = %f, s = %f" % (u, s))

    u_bars, s_bars = [], []

    for i in range(k):
        [bt], [b1], time = standard_brownian(n, 1, 1/n)
        x_t = homogeneous_equation(x_0, u, s, n, bt, time)

        r_t = instant_returns(n, x_t)
        dt = 1/n
        u_bars.append( np.mean(r_t) / dt )
        s_bars.append( np.sqrt( np.var(r_t, ddof=0) / dt ) )

    u_bar, s_bar = np.mean(u_bars), np.mean(s_bars)

    print("Estimation: u = %f, s = %f" % (u_bar, s_bar) )

    du = norm.ppf(1 - alpha/2) * np.std(u_bars)/np.sqrt(n) 
    ds = norm.ppf(1 - alpha/2) * np.std(s_bars)/np.sqrt(n) 

    print("Intervals %.2f%%:" % ((1-alpha)*100), 
          "u = (%f, %f)" % (u_bar-du, u_bar+du),
          "s = (%f, %f)" % (s_bar-ds, s_bar+ds) )
