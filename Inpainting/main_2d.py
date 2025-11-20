import numpy as np
import matplotlib.pyplot as plt

L = 200
n = 50
T0 = 0
dx = L / n
dy = dx
alpha = 2.5
tf = 60
dt = 0.1

t = np.arange(0, tf, dt)
T = np.ones((n,n)) * T0

#   l1
# l4  l2
#   l3 
l1 = np.ones(n) * 100
l2 = np.ones(n) * 0
l3 = np.ones(n) * 0
l4 = np.ones(n) * 100

def dT_dt():
  d = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      t_d = T[i-1, j] if i > 0 else l3[j]
      t_u = T[i+1, j] if i < n-1 else l1[j]
      t_l = T[i, j-1] if j > 0 else l4[i]
      t_r = T[i, j+1] if j < n-1 else l2[i]
      d[i,j] = alpha * (t_u + t_d + t_r + t_l - 4*T[i,j])
  return d 

plt.ion()

for ti in t:
  plt.pcolormesh(T, cmap=plt.cm.jet, vmin=0, vmax=100)
  plt.colorbar()
  plt.pause(0.000001)
  plt.clf()

  T += dT_dt() * dt
