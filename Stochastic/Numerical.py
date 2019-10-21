import numpy as np
import matplotlib.pyplot as plt

from Brownian import standard_brownian

def homogeneous_equation(x_0, u, s, n, bt, time):
    x = np.zeros(n)
    x[0] = x_0
    for i in range(1,n):
        x[i] = x_0*np.exp( (u-0.5*s**2)*time[i] + s*bt[i])

    return x

def euler(x_0, u, s, n, b1):
    dt = 1/n
    sqrt_dt = np.sqrt(dt)
    x = np.zeros(n)

    x[0] = x_0
    for i in range(1,n):
        x[i] = x[i-1] + u*x[i-1]*dt + s*x[i-1]* (sqrt_dt*b1[i-1])

    return x

def milstein(x_0, u, s, n, b1):
    dt = 1/n
    sqrt_dt = np.sqrt(dt)
    x = np.zeros(n)

    x[0] = x_0
    for i in range(1,n):
        x[i] = x[i-1] + u*x[i-1]*dt + s*x[i-1]* (sqrt_dt*b1[i-1]) + 0.5*(s**2)*x[i-1]*((sqrt_dt*b1[i-1])**2 - dt)

    return x


if __name__ == '__main__':
    x_0, u, s, n = 1, 1.5, 2.5, 500

    [bt], [b1], time = standard_brownian(n, 1, 1/n)

    x_t = homogeneous_equation(x_0, u, s, n, bt, time)
    x_t_e = euler(x_0, u, s, n, b1)
    x_t_m = milstein(x_0, u, s, n, b1)
    
    plt.plot(time, x_t_m, linewidth=1)
    #plt.legend(["x_t", "milstein"])
    plt.show()

    print("euler \t\t milstein \t dt")
    for i in range(1,11):
        n = 50*i
        x_t = homogeneous_equation(x_0, u, s, n, bt, time)
        x_t_e = euler(x_0, u, s, n, b1)
        x_t_m = milstein(x_0, u, s, n, b1)

        error_e = np.mean(np.abs(x_t - x_t_e))
        error_m = np.mean(np.abs(x_t - x_t_m))

        print("%f \t %f \t %f" % (error_e, error_m, 1/n) )
