import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Brownian import standard_brownian
from Numerical import homogeneous_equation
from OptionValuation import black_scholes, montecarlo, binomial_tree, finite_differences

def punto1_1_1(x_0, u, s, n=365, T=1, m=1, plot=False):
    bt, b1, time = standard_brownian(n, T, m)
    x_t = homogeneous_equation(x_0, u, s, n, m, bt, time)

    if plot:
        for simulation in x_t:
            plt.plot(simulation, linewidth=1)
        plt.show()

    return x_t

def punto1_1_2(x_t, ub, um, ua, s, n=365, T=1, plot=False):
    m = 1

    bt_b, _, time_b = standard_brownian(n, T, m)
    bt_m, _, time_m = standard_brownian(n, T, m)
    bt_a, _, time_a = standard_brownian(n, T, m)

    x_t_b = homogeneous_equation(x_t[0,-1], ub, s, n, m, bt_b, time_b)
    x_t_m = homogeneous_equation(x_t[0,-1], um, s, n, m, bt_m, time_m)
    x_t_a = homogeneous_equation(x_t[0,-1], ua, s, n, m, bt_a, time_a)

    if plot:
        plt.plot(x_t[0])
        plt.plot(np.arange(n,n*2), x_t_b[0])
        plt.plot(np.arange(n,n*2), x_t_m[0])
        plt.plot(np.arange(n,n*2), x_t_a[0])
        plt.legend(["conocido", "bajista", "constante", "alcista"])
        plt.show()

    return x_t_b, x_t_m, x_t_a

def fun(x):
    print(x.shape)

def punto1_1_3(x_t, x_t_b, x_t_m, x_t_a, u, ub, um, ua, s, n=365, T=1):
    m = 100
    bt, _, time = standard_brownian(n, T, m)
    x_t_b = homogeneous_equation(x_t[0,-1], ub, s, n, m, bt, time)

    #for simulation in x_t_b:
    #    plt.plot(simulation, linewidth=1, color='k')
    #plt.show()

    #np.apply_along_axis(fun, 0, x_t_b)

def punto2_1_2(info):
    r = 0.0269
    r_t = info["ret.adjusted.prices"][1:]
    n = len(r_t)
    dt = 1/n
    s_hat = np.sqrt( np.var(r_t, ddof=0) / dt )
    return r, s_hat

def punto2_1_3(info, r, s, T=1):
    price = info["price.adjusted"]
    n = len(price)
    k = x_0 = price.iloc[-1]

    f_bs = black_scholes(x_0, r, s, k, T)

    m, w = 1000, 100
    f_mc, fcalls = montecarlo(x_0, r, s, n, T, m, w, k)
    f_bt = binomial_tree(x_0, r, s, k, 5000)
    f_fd = finite_differences(s, r, k, T, 500)

    return f_bs, f_mc, f_bt, f_fd

if __name__ == '__main__':
    #s = 0.75
    #u, ub, um, ua = 1, -0.5, 1, -0.75
    #x_t = punto1_1_1(5, u, s)
    #x_t_b, x_t_m, x_t_a = punto1_1_2(x_t, ub, um, ua, s, plot=True)
    #punto1_1_3(x_t, x_t_b, x_t_m, x_t_a, u, ub, um, ua, s)

    info_activo = pd.read_csv("AABA_2019.csv")
    r, s_hat = punto2_1_2(info_activo)
    f_bs, f_mc, f_bt, f_fd = punto2_1_3(info_activo, r, s_hat)
    print(f_bs, f_mc, f_bt, f_fd)

