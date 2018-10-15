import sys
import numpy as np

d = np.loadtxt(sys.argv[1], dtype=int)
num_machines, num_jobs = d.shape

def get_b_i(i):
    if i == 0:
        return 0
    else:
        return min([np.sum(d[:i,j]) for j in range(num_jobs)])

def get_a_i(i):
    if i == num_machines-1:
        return 0
    else:
        return min([np.sum(d[i+1:,j]) for j in range(num_jobs)])

def get_t_i(i):
    return np.sum(d[i])

def first():
    return max([get_b_i(i)+get_t_i(i)+get_a_i(i) for i in range(num_machines)])

def second():
    return max(np.sum(d, axis=0))

def main():
    lb = max( first(), second() )
    print(lb)

main()
