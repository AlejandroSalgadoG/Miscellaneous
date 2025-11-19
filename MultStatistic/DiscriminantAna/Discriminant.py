import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

data = pd.read_csv( "data.csv" )
groups = [ group.reset_index(drop=True).drop(columns=["G"]) for _,group in data.groupby("G") ]
g = data["G"]
data = data.drop( columns=["G"] )

k= 3
n = 30
p = 6

total_mean = data.sum() / (n*k)
group_means = [ group.mean(axis=0) for group in groups ]

H = np.zeros( (p,p) )
for group_mean in group_means:
    center_mean = group_mean - total_mean
    center_mean = center_mean.to_frame()
    H += center_mean.dot( center_mean.T )

E = np.zeros( (p,p) )
for group, mean in zip( groups, group_means ):
    center_samples = group - mean.values.T
    for _, center_sample in center_samples.iterrows():
        center_sample = center_sample.to_frame() 
        E += center_sample.dot( center_sample.T )

E_1 = np.linalg.inv( E.values )
EH = E_1.dot(H)

n_l = min(k-1, p)

l,A = np.linalg.eig( EH )
idx = np.argsort(l)[::-1]
l,A = np.real(l)[idx][:n_l], np.real(A)[:,idx][:,:n_l]

print(l)
print( A )

Z = data.dot(A)
print( Z )
