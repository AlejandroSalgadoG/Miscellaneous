import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

data = pd.read_csv( "data.csv" )
groups = [ group.reset_index(drop=True).drop(columns=["G"]) for _,group in data.groupby("G") ]
data = data.drop( columns=["G"] )

k = 3
n = 30
p = 6

group_means = [ group.mean(axis=0) for group in groups ]
group_cov = [ group.cov() for group in groups ]

E = np.zeros( (p,p) )
for group, mean in zip( groups, group_means ):
    center_samples = group - mean.values.T
    for _, center_sample in center_samples.iterrows():
        center_sample = center_sample.to_frame() 
        E += center_sample.dot( center_sample.T )

Sp = E / (n*3 - k)
Sp_1 = np.linalg.inv( Sp.values )

def lin_classif( y_bar, Sp_1, y ):
    b = y_bar.T.dot( Sp_1 )
    a = b.dot( y_bar )
    return np.log( 1/k ) - a / 2 + b.dot(y) 

def quad_classif( y_bar, S, y ):
    S_1, d_S = np.linalg.inv( S.values ), np.linalg.det( S.values )
    return np.log( 1/k ) - np.log( d_S ) / 2 + mahalanobis( y, y_bar, S_1 )**2 / 2

y = data.iloc[0,:]

lin_criteria = [ lin_classif( y_bar, Sp_1, y ) for y_bar in group_means ]
quad_criteria = [ quad_classif( y_bar, S, y ) for (y_bar, S) in zip(group_means, group_cov) ]

print( lin_criteria,  np.argmax( lin_criteria ) )
print( quad_criteria, np.argmax( quad_criteria ) )
