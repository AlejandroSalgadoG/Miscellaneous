import numpy as np
import pandas as pd
from math import floor
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis

from Auxiliar import random_subsets

data = pd.read_csv( "pulpfiber_simp.csv" )
n,p = data.shape

m = 4000 # number of combinations to be used

h = floor( (n+p+1)/ 2 )
c = np.sqrt( chi2.ppf(h/n, p) )

obj_fun = []

subsets = random_subsets( np.arange(n), m, p+1 )
for subset in subsets:
    sample = data.iloc[subset,:]
    u_bar = sample.mean()
    s = sample.cov(ddof=0)
    d_s = np.linalg.det(s)
    s_1 = np.linalg.inv( s )

    dist = np.sort([ mahalanobis( x, u_bar, s_1 ) for _,x in data.iterrows() ])
    obj = (dist[h] / c)**p * np.sqrt(d_s)
    obj_fun.append( obj )

best_subset = subsets[ np.argmin( obj_fun ) ]
best_data = data.iloc[best_subset,:]

best_u = best_data.mean()
best_s = best_data.cov()

print( best_u )
print( best_s )
