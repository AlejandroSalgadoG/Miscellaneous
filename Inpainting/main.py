import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

L = 50
n = 30
T0 = 0
dx = L / n
alpha = 4
tf = 60
dt = 0.1

t1, t2 = 100, 90

x = np.linspace(0, L, n)
t = np.arange(0, tf, dt)
T = np.ones(n) * T0 

def dT_dt():
  deriv = np.zeros(n)
  for i in range(n):
    t_b = T[i-1] if i > 0 else t1
    t_f = T[i+1] if i < n-1 else t2
    deriv[i] = alpha * (t_b -2*T[i] + t_f) / dx**2
  return deriv 

def get_color_line(x, T):
  points = np.array([x, T]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  return LineCollection(segments, cmap='jet', norm=plt.Normalize(0, 100))

fig, axs = plt.subplots(1, 1)
fig.colorbar(get_color_line(x, T))

for ti in t:
  lc = get_color_line(x, T)
  lc.set_array(T)
  axs.add_collection(lc)

  axs.set_xlim(0, L)
  axs.set_ylim(0, 100) 
  plt.pause(0.01)

  T += dT_dt() * dt
