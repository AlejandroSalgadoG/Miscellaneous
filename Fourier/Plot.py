import numpy as np
import matplotlib.pyplot as plt

def time_fun(t, w):
    return np.cos(2*np.pi*t*w)

f_sin = 2

f = 2
t = 4.5

t_r = np.arange(0, t, 0.001)
t_i = -2*np.pi*t_r*f

f_t = time_fun(t_r, f_sin)

t_x, t_y = np.cos(t_i), np.sin(t_i)
f_x, f_y = f_t*t_x, f_t*t_y

plt.subplot(211)
plt.scatter(t_r, f_t, s=1 )

plt.subplot(212)
plt.scatter(t_x, t_y, s=1)
plt.plot(f_x, f_y, c='r')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])

plt.show()
