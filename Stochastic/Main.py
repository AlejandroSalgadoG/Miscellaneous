import matplotlib.pyplot as plt

from Brownian import standard_brownian, bridge_brownian, drift_brownian, geometric_brownian

#simulations, time = standard_brownian(n=100, k=1000, dt=0.01)
#simulations, time = bridge_brownian(n=100, k=1000, dt=0.01)
#simulations, time = drift_brownian(n=100, k=1000, dt=0.01, u=20, s=5)
simulations, time = geometric_brownian(n=100, k=1000, dt=0.01, a=0.9, l=0.2)

for simulation in simulations:
    plt.plot(time, simulation, linewidth=1)
plt.show()
