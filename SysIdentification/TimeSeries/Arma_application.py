import numpy as np
import matplotlib.pyplot as plt

def norm_numbers(params, shape):
    mu, sigma = params
    return np.random.normal(mu, sigma, shape)

def t_numbers(params, shape):
    v = params
    return np.random.standard_t(v, shape)

def arma_series(T, r_0, phi, theta, distribution, params):
    r = np.zeros(T+1)
    a = distribution(params, T+2)

    phi_0, phi_1 = phi
    theta_1, theta_2 = theta
    
    r[0] = r_0
    
    for t in range(T):
        t_r = t+1 
        t_a = t+2
    
        r[t_r] = np.sum( phi_0 + phi_1*r[t_r-1] + a[t_a] - theta_1*a[t_a-1] - theta_2*a[t_a-2])

    return r

T  = 200
r_0 = 5.6

phi = [15.5, 0.779]
theta = [-0.0445, 0.382]

r_norm = arma_series(T, r_0, phi, theta, norm_numbers, [0, 1])
r_stud = arma_series(T, r_0, phi, theta, t_numbers, [2])

print("mean normal: %.2f" % np.mean(r_norm), "-", "variance normal: %.2f" % np.var(r_norm))
print("mean student: %.2f" % np.mean(r_stud), "-" , "variance student: %.2f" % np.var(r_stud))

plt.ylim(0,90)
plt.plot(np.arange(T+1), r_norm, 'C0', linewidth=1)
plt.plot(np.arange(T+1), r_stud, 'C1', linewidth=1,)
plt.show()
