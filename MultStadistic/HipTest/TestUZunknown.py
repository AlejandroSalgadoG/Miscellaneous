import numpy as np
from scipy.stats import f
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

# table 3.3
X = [ [35, 3.5 , 2.80],  
      [35, 4.9 , 2.70],
      [40, 30.0, 4.38],
      [10, 2.8 , 3.21],
      [6 , 2.7 , 2.73],
      [20, 2.8 , 2.81],
      [35, 4.6 , 2.88],
      [35, 10.9, 2.90],
      [35, 8.0 , 3.28],
      [30, 1.6 , 3.20] ]

n = len(X)
u_0 = [15, 6, 2.85]
X_T = np.array(X).T

y_bar = np.mean( X, axis=0 )
s = np.cov( X_T )
s_1 = np.linalg.inv( s )

v, p = n-1, 3
t_2 = n * np.matmul( np.matmul( (y_bar-u_0).T, s_1 ), y_bar-u_0 )
#t_2 = n * mahalanobis( y_bar, u_0, s_1 ) ** 2
F_test = (v-p+1)/(v*p) * t_2
F = f.ppf(0.95, p, v-p+1)

x = np.linspace(f.ppf(0.01, p, v-p+1), f.ppf(0.99, p, v-p+1), 100)
plt.plot( x, f.pdf(x, p, v-p+1) )
plt.vlines( F, 0, f.pdf(F, p, v-p+1) )
plt.vlines( F_test, 0, f.pdf(F_test, p, v-p+1), color='k' )

plt.xlim( f.ppf(0.01, p, v-p+1), f.ppf(0.99, p, v-p+1) )
plt.ylim( 0, 0.7 )
plt.show()
