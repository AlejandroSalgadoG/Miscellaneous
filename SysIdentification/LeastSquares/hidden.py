import numpy as np

A = np.array([ [1,2], 
               [3,4] ])

def norm_numbers(mu, sigma, shape):
    return np.random.normal(mu, sigma, shape)

def secret_system(x):
  return np.matmul(A, x) + norm_numbers(0, 0.1, [2,1])