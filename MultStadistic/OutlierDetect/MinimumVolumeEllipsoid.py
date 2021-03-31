import numpy as np
import pandas as pd
from math import floor
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis

from Auxiliar import random_subsets

data = pd.read_csv( "pulpfiber_simp.csv" )
n,p = data.shape

m = 3000 # number of combinations to be used

h = floor( (n+p+1)/ 2 )
c = np.sqrt( chi2.ppf(h/n, p) )

def get_estimations( sample ):
    u_bar = sample.mean()
    s = sample.cov()
    d_s = np.linalg.det(s)
    s_1 = np.linalg.inv( s )
    return u_bar, s, s_1, d_s

obj_fun = []

print( "calculating subsets...", end="" )
subsets = random_subsets( np.arange(n), m, p+1 )
print( "done" )

for i, subset in enumerate(subsets):
    print( "%.2f%%" % (i/m*100), end="\r" ) # print progress
    sample = data.iloc[subset,:]
    u_bar, s, s_1, d_s = get_estimations( sample )

    dist = np.sort([ mahalanobis( x, u_bar, s_1 ) for _,x in data.iterrows() ])

    obj = (dist[h]/c)**p * np.sqrt(d_s)
    obj_fun.append( obj )

best_subset = subsets[ np.argmin( obj_fun ) ]
best_sample = data.iloc[best_subset,:]

best_u, best_s, best_s_1, d_s = get_estimations( best_sample )
#best_dist = np.sort([ mahalanobis( x, best_u, best_s_1 ) for _,x in data.iterrows() ])
#best_s *= best_dist[h]/c**2

print( best_u.values )
print( best_s.values )
