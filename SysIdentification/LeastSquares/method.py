import numpy as np

from linear import vector, invert, multiply

def result_matrix(v1, v2):
  n, _ = v1.shape
  m, _ = v2.shape
  return np.zeros([n,m])

def calc_v(inputs, outputs):
  vt = result_matrix(inputs[0], outputs[0])
  for x, y in zip(inputs, outputs):
    vt += multiply(x,y.T)
  return vt

def calc_g(inputs):
  gt = result_matrix(inputs[0], inputs[0])
  for x in inputs:
    gt += multiply(x, x.T)
  return gt

def least_squares(inputs, outputs):
  vt = calc_v(inputs, outputs)
  gt = calc_g(inputs)
  gt_1 = invert(gt)
  return multiply(vt.T, gt_1)