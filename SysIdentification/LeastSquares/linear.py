import numpy as np

def vector(values):
  return np.array( [[x] for x in values] )

def multiply(A,B):
  return np.matmul(A,B)

def invert(A):
  return np.linalg.inv(A)