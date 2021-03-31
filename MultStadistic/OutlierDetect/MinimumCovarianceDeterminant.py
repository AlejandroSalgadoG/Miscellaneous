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
c_steps = 5

def get_estimations( sample ):
    u_bar = sample.mean()
    s = sample.cov()
    s_1 = np.linalg.inv( s )
    return u_bar, s, s_1

def make_iteration(data, subset):
    sample = data.iloc[subset,:]
    u_bar, s, s_1 = get_estimations( sample )
    obj_fun = np.linalg.det( s )
    return np.argsort([ mahalanobis( x, u_bar, s_1 ) for _,x in data.iterrows() ]), obj_fun

obj_funs = []

print( "calculating subsets...", end="" )
subsets = random_subsets( np.arange(n), m, h )
print( "done" )

for i, subset in enumerate(subsets):
    print( "%.2f%%" % (i/m*100), end="\r" ) # print progress
    sorted_idx, _ = make_iteration( data, subset )
    for i in range(c_steps): sorted_idx, obj_fun = make_iteration( data, sorted_idx[:h] )
    obj_funs.append( obj_fun ) 

best_subset = subsets[ np.argmin( obj_fun ) ]
best_sample = data.iloc[ best_subset, : ]
u_bar, s, _ = get_estimations( best_sample )

print( u_bar.values )
print( s.values )
