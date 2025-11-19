import numpy as np
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

# Table 3.1
X = [ [69, 153], 
      [74, 175],
      [68, 155],
      [70, 135],
      [72, 172],
      [67, 150],
      [66, 115],
      [70, 137],
      [76, 200],
      [68, 130],
      [72, 140],
      [79, 265],
      [74, 185],
      [67, 112],
      [66, 140],
      [71, 150],
      [74, 165],
      [75, 185],
      [75, 210],
      [76, 220] ]

sigma = [ [ 20,  100],
          [100, 1000] ]

n = len(X)
u_0 = [70, 170]
y_bar = np.mean( X, axis=0 )
sigma_1 = np.linalg.inv( sigma )

z_2 = n * np.matmul( np.matmul( (y_bar-u_0).T, sigma_1 ), y_bar-u_0 )
#z_2 = n * mahalanobis( y_bar, u_0, sigma_1 ) ** 2
chi_2 = chi2.ppf(0.95, 2)

x = np.linspace(chi2.ppf(0.01, 2), chi2.ppf(0.99, 2), 100)
plt.plot( x, chi2.pdf(x, 2) )
plt.vlines( chi_2, 0, chi2.pdf(chi_2, 2) )
plt.vlines( z_2, 0, chi2.pdf(z_2, 2), color='k' )

plt.xlim( chi2.ppf(0.01, 2), chi2.ppf(0.99, 2) )
plt.ylim( 0, chi2.pdf(0.01, 2) )
plt.show()
