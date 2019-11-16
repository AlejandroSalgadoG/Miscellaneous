import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

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

def punto1_1_3(x_t, x_t_f, u, u_f, s, n=365, T=1, plot=False):
    np.random.seed()
    ub, um, ua = u_f
    [x_t_b], [x_t_m], [x_t_a] = x_t_f

    k = 1000
    x_b, x_m, x_a = np.zeros([k, n]), np.zeros([k, n]), np.zeros([k, n])
    for i in range(k):
        x_b[i], x_m[i], x_a[i] = punto1_1_2(x_t, ub, um, ua, s)

    weights = np.linspace(1,0.6,n)

    x_b_u = np.mean(x_b, axis=0) + np.std(x_b, axis=0) * weights
    x_b_l = np.mean(x_b, axis=0) - np.std(x_b, axis=0) * weights
    inside = sum( [1 for i in range(n) if x_b_l[i] <= x_t_b[i] <= x_b_u[i]] )
    print("% aciertos bajista", inside/n*100)

    x_m_u = np.mean(x_m, axis=0) + np.std(x_m, axis=0) * weights
    x_m_l = np.mean(x_m, axis=0) - np.std(x_m, axis=0) * weights
    inside = sum( [1 for i in range(n) if x_m_l[i] <= x_t_m[i] <= x_m_u[i]] )
    print("% aciertos constante", inside/n*100)

    x_a_u = np.mean(x_a, axis=0) + np.std(x_a, axis=0) * weights
    x_a_l = np.mean(x_a, axis=0) - np.std(x_a, axis=0) * weights
    inside = sum( [1 for i in range(n) if x_a_l[i] <= x_t_a[i] <= x_a_u[i]] )
    print("% aciertos alcista", inside/n*100)

    if plot:
        plt.plot(x_t_b, color='y')
        plt.plot(x_t_m, color='g')
        plt.plot(x_t_a, color='r')

        plt.plot(x_b_u, color='b', linewidth=0.5)
        plt.plot(x_b_l, color='b', linewidth=0.5)

        plt.plot(x_m_u, color='b', linewidth=0.5)
        plt.plot(x_m_l, color='b', linewidth=0.5)

        plt.plot(x_a_u, color='b', linewidth=0.5)
        plt.plot(x_a_l, color='b', linewidth=0.5)
        plt.legend(["bajista", "constante", "alcista", "bandas"])
        plt.show()

def punto1_1_4(x_0, u, u_f, s, seed, n=365, T=1):
    np.random.seed(seed)
    print("1.1.4 s=%.2f" % s)
    ub, um, ua = u_f
    x_t = punto1_1_1(x_0, u, s, plot=False)
    x_t_f = punto1_1_2(x_t, ub, um, ua, s, plot=False)
    np.random.seed()
    punto1_1_3(x_t, x_t_f, u, u_f, s, plot=False)

def punto1_2_1(a, u, s, r, n, T, plot=False):
    dt = T/n
    sqrt_dt = np.sqrt(dt)
    x = np.zeros(n)
    normal = np.random.normal(0, 1, n-1)

    x[0] = u
    for i in range(1,n):
        x[i] = x[i-1] + a*(u-x[i-1])*dt + x[i-1]**r * s*sqrt_dt*normal[i-1]

    if plot:
        plt.plot(x)
        plt.show()
    return x

def punto1_2_3(x, n, T, r):
    dt = T/n

    A = sum([(x[i]*x[i-1])/(x[i-1]**(r*2)) for i in range(1, n)])
    B = sum([x[i-1]/(x[i-1]**(r*2)) for i in range(1, n)])
    C = sum([x[i]/(x[i-1]**(r*2)) for i in range(1, n)])
    D = sum([1/(x[i-1]**(r*2)) for i in range(1, n)])
    E = sum([(x[i-1]/(x[i-1]**r))**2 for i in range(1, n)])

    a = (E*D - B**2 - A*D + B*C) / ((E*D - B**2) * dt)
    u = (A - E*(1-a*dt))/ (a*B*dt)

    tmp = sum([((x[i] - x[i-1] - a*(u-x[i-1])*dt) / (x[i-1]**r) )**2 for i in range(1,n)])
    s = np.sqrt(tmp/T)

    return a, u, s

def punto1_2_4(a, u ,s, r, n, T):
    for a_tmp in [90, 100, 110]:
        x_t = punto1_2_1(a_tmp, u, s, r, n, T, plot=False)
        a_hat, u_hat, s_hat = punto1_2_3(x_t, n, T, r)
        print("1.2.4 a=%.2f a_hat=%f u_hat=%f s_hat=%f" % (a_tmp, a_hat, u_hat, s_hat))

    for u_tmp in [80, 90, 100]:
        x_t = punto1_2_1(a, u_tmp, s, r, n, T, plot=False)
        a_hat, u_hat, s_hat = punto1_2_3(x_t, n, T, r)
        print("1.2.4 u=%.2f a_hat=%f u_hat=%f s_hat=%f" % (u_tmp, a_hat, u_hat, s_hat))

    for s_tmp in [2, 2.5, 3]:
        x_t = punto1_2_1(a, u, s_tmp, r, n, T, plot=False)
        a_hat, u_hat, s_hat = punto1_2_3(x_t, n, T, r)
        print("1.2.4 s=%.2f a_hat=%f u_hat=%f s_hat=%f" % (s_tmp, a_hat, u_hat, s_hat))

def punto2_1_2(returns):
    r = 0.0269
    r_t = returns[1:]
    n = len(r_t)
    dt = 1/n
    s_hat = np.sqrt( np.var(r_t, ddof=0) / dt )
    return r, s_hat

def punto2_1_3(x_0, r, s, n, k, T, m, w):
    f_bs = black_scholes(x_0, r, s, k, T)
    f_mc, fcalls = montecarlo(x_0, r, s, n, T, m, w, k)
    f_bt = binomial_tree(x_0, r, s, k, T, 500)
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
    # First part
    x_0, u, s = 5, 0.5, 0.1
    u_f = ub, um, ua = -0.3, u, 1.3
    np.random.seed(344535096)
    x_t = punto1_1_1(x_0, u, s, plot=True)
    x_t_f = punto1_1_2(x_t, ub, um, ua, s, plot=True)
    punto1_1_3(x_t, x_t_f, u, u_f, s, plot=True)
    for seed, s in [(39114868, 0.2), (737532457, 0.3), (350989215, 0.4)]:
        punto1_1_4(x_0, u, u_f, s, seed)
    
    np.random.seed(2)
    a, u, s, r, n, T = 100, 90, 2.5, 0, 500, 1
    x_t = punto1_2_1(a, u, s, r, n, T, plot=True)
    a_hat, u_hat, s_hat = punto1_2_3(x_t, n, T, r)
    print("1.2.3 a=%.2f u=%.2f s=%.2f a_hat=%f u_hat=%f s_hat=%f" % (a, u, s, a_hat, u_hat, s_hat))
    punto1_2_4(a, u, s, r, n, T)

    # Second part
    np.random.seed()
    info_activo = pd.read_csv("AABA_2019.csv")
    price = info_activo["price.adjusted"]
    returns = info_activo["ret.adjusted.prices"]

    k = x_0 = price.iloc[-1]
    n, T = len(price), 1
    m, w = 1000, 100
    r, s = punto2_1_2(returns)
    print("2.1.2 r=%f s=%f" % (r,s))

    f_bs, f_mc, f_bt, f_fd = punto2_1_3(x_0, r, s, n, k, T, m, w)
    print("2.1.3 bs=%f mc=%f bt=%f fd=%f" % (f_bs, f_mc, f_bt, f_fd))

    punto2_2_1(x_0, r, s, n, k, T, m, w)
    punto2_2_2(x_0, r, s, n, k, T, m, w)
    punto2_2_3(x_0, r, s, n, k, T, m, w)
    punto2_2_4(x_0, r, s, n, k, T, m, w)
    punto2_2_5(x_0, r, s, n, k, T, m, w)
    punto2_2_6(x_0, r, s, k, T)
    punto2_2_7(x_0, r, s, k, T)
