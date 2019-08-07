from hidden import secret_system
from method import least_squares
from linear import vector

x1 = vector([1,3])
x2 = vector([2,4])

y1 = secret_system(x1)
y2 = secret_system(x2)

inputs = [x1, x2]
outputs = [y1, y2]

A_hat = least_squares(inputs, outputs)

print(A_hat)