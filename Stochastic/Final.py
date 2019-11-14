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

def punto2_1_2(returns):
    r = 0.0269
    r_t = returns[1:]
    n = len(r_t)
    dt = 1/n
    s_hat = np.sqrt( np.var(r_t, ddof=0) / dt )
    return r, s_hat

def punto2_1_3(x_0, r, s, n, k, T, m, w):
    f_bs = black_scholes(x_0, r, s, k, T)
    f_mc, fcalls = 1,1#montecarlo(x_0, r, s, n, T, m, w, k)
    f_bt = 1#binomial_tree(x_0, r, s, k, T, 500)
    f_fd = finite_differences(s, r, k, T, 500)
    return f_bs, f_mc, f_bt, f_fd

def punto2_2_1(x_0, r, s, n, k, T, m, w):
    for T in np.linspace(0.25,T,4):
        f_bs, f_mc, f_bt, f_fd = punto2_1_3(x_0, r, s, n, k, T, m, w)
        print("2.2.1 T=%f bs=%f mc=%f bt=%f fd=%f" % (T, f_bs, f_mc, f_bt, f_fd))
    
def punto2_2_2(x_0, r, s, n, k, T, m, w):
    for r in np.linspace(0.005,r,4):
        f_bs, f_mc, f_bt, f_fd = punto2_1_3(x_0, r, s, n, k, T, m, w)
        print("2.2.2 r=%f bs=%f mc=%f bt=%f fd=%f" % (r, f_bs, f_mc, f_bt, f_fd))

def punto2_2_3(x_0, r, s, n, k, T, m, w):
    for s in np.linspace(0.05,s,4):
        f_bs, f_mc, f_bt, f_fd = punto2_1_3(x_0, r, s, n, k, T, m, w)
        print("2.2.3 s=%f bs=%f mc=%f bt=%f fd=%f" % (s, f_bs, f_mc, f_bt, f_fd))

def punto2_2_4(x_0, r, s, n, k, T, m, w):
    for k in np.linspace(5,k,4):
        f_bs, f_mc, f_bt, f_fd = punto2_1_3(x_0, r, s, n, k, T, m, w)
        print("2.2.4 k=%f bs=%f mc=%f bt=%f fd=%f" % (k, f_bs, f_mc, f_bt, f_fd))

def punto2_2_5(x_0, r, s, n, k, T, m, w):
    f_bs = black_scholes(x_0, r, s, k, T)
    print("2.2.5 f=%f" % f_bs)

    for new_m in np.linspace(10,m,4):
        f_mc, fcalls = montecarlo(x_0, r, s, n, T, int(new_m), w, k)
        print("2.2.5 m=%d f=%f" % (new_m, f_mc))

    for new_w in np.linspace(80,200,4):
        f_mc, fcalls = montecarlo(x_0, r, s, n, T, m, int(new_w), k)
        print("2.2.5 w=%d f=%f" % (new_w, f_mc))

    for new_n in np.linspace(100,n,4):
        f_mc, fcalls = montecarlo(x_0, r, s, int(new_n), T, m, w, k)
        print("2.2.5 n=%d f=%f" % (new_n, f_mc))

def punto2_2_6(x_0, r, s, k, T):
    f_bs = black_scholes(x_0, r, s, k, T)
    print("2.2.6 f=%f" % f_bs)

    for n in np.linspace(80,500,4):
        f_fd = finite_differences(s, r, k, T, int(n))
        print("2.2.6 n=%d f=%f" % (n, f_fd))

def punto2_2_7(x_0, r, s, k, T):
    f_bs = black_scholes(x_0, r, s, k, T)
    print("2.2.7 f=%f" % f_bs)

    for n in np.linspace(80,500,4):
        f_bt = binomial_tree(x_0, r, s, k, T, int(n))
        print("2.2.7 n=%d f=%f" % (n, f_bt))

if __name__ == '__main__':
    #s = 0.75
    #u, ub, um, ua = 1, -0.5, 1, -0.75
    #x_t = punto1_1_1(5, u, s)
    #x_t_b, x_t_m, x_t_a = punto1_1_2(x_t, ub, um, ua, s, plot=True)
    #punto1_1_3(x_t, x_t_b, x_t_m, x_t_a, u, ub, um, ua, s)

    info_activo = pd.read_csv("AABA_2019.csv")
    price = info_activo["price.adjusted"]
    returns = info_activo["ret.adjusted.prices"] 

    k = x_0 = price.iloc[-1]
    n, T = len(price), 1
    m, w = 1000, 100
    r, s = punto2_1_2(returns)

    f_bs, f_mc, f_bt, f_fd = punto2_1_3(x_0, r, s, n, k, T, m, w)
    print("2.1.3 bs=%f mc=%f bt=%f fd=%f" % (f_bs, f_mc, f_bt, f_fd))

#    punto2_2_1(x_0, r, s, n, k, T, m, w)
#    punto2_2_2(x_0, r, s, n, k, T, m, w)
#    punto2_2_3(x_0, r, s, n, k, T, m, w)
#    punto2_2_4(x_0, r, s, n, k, T, m, w)
#    punto2_2_5(x_0, r, s, n, k, T, m, w)
#    punto2_2_6(x_0, r, s, k, T)
#    punto2_2_7(x_0, r, s, k, T)
