import numpy as np
import matplotlib.pyplot as plt

def get_zero_mean_normal_vars(cov, shape):
    n_vars, _ = cov.shape 
    mean = np.zeros(n_vars)
    return np.random.multivariate_normal(mean, cov , shape)

T = 20

x_0 = np.array([3e4,10,5])
A = np.array([ [1,1,0.5], [0,1,1], [0,0,1] ])

F = A
Q = 100*np.array([[1,0,0], [0,1e-3,0], [0,0,1e-3]])

H = np.array([[1,0,0],[0,1,0],[0,0,1]])
R = 10000*np.array([[1,0,0], [0,1e-3,0], [0,0,1e-3]])

Xi = get_zero_mean_normal_vars(Q, T+1)
Zeta = get_zero_mean_normal_vars(R, T+1)

def real_model(A, x_0, T):
    _, n_vars = A.shape
    x = np.zeros([T+1, n_vars])
    x[0] = x_0
    for i in range(0,T):
        x[i+1] = np.matmul(A, x[i])
    return x

def simulate(A, x_0, Xi, T):
    _, n_vars = A.shape
    x = np.zeros([T+1, n_vars])
    x[0] = x_0
    for i in range(0,T):
        x[i+1] = np.matmul(A, x[i]) + Xi[i]
    return x

def perfect_observer(H, x, T):
    return np.array([ np.matmul(H, x[i]) for i in range(0,T+1)])

def noisy_observer(H, x, Zeta, T):
    return np.array([ np.matmul(H, x[i]) + Zeta[i] for i in range(0,T+1)])

def kalman_filter(F, H, R, Q, z_0, y, T):
    _, n_vars = F.shape
    z_0 = x_0
    s_0 = np.zeros([n_vars, n_vars])
    z_hat, z_pred = np.zeros([T+1, n_vars]), np.zeros([T+1, n_vars])
    s_hat, s_pred = np.zeros([T+1, n_vars, n_vars]), np.zeros([T+1, n_vars, n_vars])

    z_hat[0] = z_0
    s_hat[0] = s_0
    for i in range(0,T):
        z_pred[i+1] = np.matmul(F, z_hat[i])
        s_pred[i+1] = np.matmul(np.matmul(F, s_hat[i]), F. T) + Q
    
        M_opt = np.matmul(np.matmul(H, s_pred[i+1]), H.T) + R
        M_opt = np.linalg.inv(M_opt)
        M_opt = np.matmul(s_pred[i+1], np.matmul(H.T, M_opt))

        delta_y = y[i+1] - np.matmul(H, z_pred[i+1])
        z_hat[i+1] = z_pred[i+1] + np.matmul(M_opt, delta_y)
        s_hat[i+1] = s_pred[i+1] - np.matmul(M_opt, np.matmul(H, s_pred[i+1]))
    return z_hat

def plot_simulations(y_real, y, y_kalman, var, T):
    t = np.arange(T+1)
    plt.scatter(t, y[:,var])
    plt.plot(t, y[:,var])
    plt.scatter(t, y_real[:,var], color='g')
    plt.plot(t, y_real[:,var], color='g')
    plt.scatter(t, y_kalman[:,var], color="r")
    plt.plot(t, y_kalman[:,var], color="r")
    plt.legend(["Simulation", "Real", "Kalman"])
    plt.show()

x_real = real_model(A, x_0, T)
y_real = perfect_observer(H, x_real, T)

x = simulate(A, x_0, Xi, T)
y = noisy_observer(H, x, Zeta, T)

x_kalman = kalman_filter(A, H, R, Q, x_0, y, T)
y_kalman = perfect_observer(H, x_kalman, T)

plot_simulations(y_real, y, y_kalman, 0, T)
