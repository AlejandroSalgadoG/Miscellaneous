import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv("Data.txt")
data = data[ data["Group"] > 1 ]
data = data[ ["WDIM", "CIRCUM", "FBEYE", "EYEHD", "EARHD", "JAW"]  ]
cov = data.cov()
print( cov )

l,A = np.linalg.eig( cov )
idx = np.argsort(l)[::-1]
l,A = l[idx], A[:,idx]

prop_var = l / np.sum( l )
Z = data.dot(A)

print( np.sqrt(l) )
print(prop_var)
print(A)
