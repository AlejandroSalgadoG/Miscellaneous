import math
import numpy as np
import matplotlib.pyplot as plt

n = 100
dt = 1/n # inter sample time
fs = 1/dt # sample frequency
p = n*dt # period
df = fs/n # frequency resolution
print(df)

f = 6 # frequency
a = 3 # amplitud

t = np.arange(0,p,dt) # time
x = a * np.cos( 2*np.pi*f*t )

nyquist = math.floor( t.size/2 ) + 1
coef = np.fft.fft( x )[:nyquist] / n

plt.scatter( np.arange(coef.size), abs(coef)*2, s=7 )
plt.scatter( np.arange(coef.size), np.angle(coef), s=7 )
plt.show()
