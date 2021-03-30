import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

data = pd.read_csv( "pulpfiber_simp.csv" )
n,p = data.shape

u_bar = data.mean()
s = data.cov()
s_1 = np.linalg.inv( s )

dist = [ mahalanobis( x, u_bar, s_1 ) for _,x in data.iterrows() ]
threshold = np.sqrt( chi2.ppf(0.975, p) )

plt.scatter( np.arange( n ), dist )
plt.hlines(threshold, 0, n)
plt.ylim(0, 16)
plt.show()
