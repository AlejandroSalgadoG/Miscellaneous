import numpy as np
from itertools import combinations

def random_subsets(data, n, k):
    subsets = tuple( combinations( data, k ) )
    return [ list(subsets[i]) for i in np.random.choice( len(subsets), n, replace=False )]
