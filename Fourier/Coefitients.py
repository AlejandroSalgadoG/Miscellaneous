import math
import numpy as np
import matplotlib.pyplot as plt

n = 100
dt = 1/n # inter sample time
p = n*dt # period
df = 1/p # frequency resolution
print(df)

f, f2 = 6, 32 # frequency
a, a2 = 3, 3 # amplitude
phi, phi2 = 0, 0 # phase

t = np.arange(0,p,dt) # time
x = a * np.cos( 2*np.pi*f*t + phi) + a2 * np.cos( 2*np.pi*f2*t + phi2 ) # sample function

nyquist = math.floor( t.size/2 ) + 1
coef = np.fft.fft( x )[:nyquist] / n

fig, (ax1, ax2) = plt.subplots(2)

ax1.scatter( np.arange(coef.size), abs(coef)*2, s=7 )
ax2.scatter( np.arange(coef.size), np.angle(coef), s=7 )

ax1.set_ylabel("Amplitude")
ax2.set_ylabel("Phase")
ax2.set_xlabel("Frequency")
plt.show()

waves = []
for idx, c in enumerate(coef):
    f, a, phi = idx, abs(c)*2, np.angle(c)
    x_p = a * np.cos( 2*np.pi*f*t + phi )
    waves.append( x_p )
    
waves = np.array( waves )
waves[0] /= 2
fourier_wave = np.sum( waves, axis=0 )

plt.plot( t, waves[6] + waves[0] )
plt.plot( t, x )
plt.show()

plt.plot( t, waves[32] + waves[0] )
plt.plot( t, x )
plt.show()

plt.plot( t, fourier_wave )
plt.plot( t, x )
plt.show()
