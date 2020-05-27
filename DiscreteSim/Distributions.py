import numpy as np

def exponential(beta):
    return np.random.exponential(beta)

def gamma(shape, scale):
    return np.random.gamma(shape, scale)

def fixed(time):
    return time
