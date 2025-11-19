import numpy as np

def check_duplicates(subsets, new_subset):
    for subset in subsets:
        if np.array_equal( subset, new_subset ): 
            return False
    return True

def random_subsets(data, n, k):
    subsets = []
    for i in range(n):
        unique_subset = False
        while not unique_subset:
            new_subset = np.random.choice( data, k, replace=False )
            unique_subset = check_duplicates( subsets, new_subset )
        subsets.append( new_subset )
    return subsets
