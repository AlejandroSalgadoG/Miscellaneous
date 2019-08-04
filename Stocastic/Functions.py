import numpy as np

def vector_times_scalars(vector, scalars):
    vectors = np.tile(vector, [len(scalars),1])
    return (vectors.T * scalars).T
