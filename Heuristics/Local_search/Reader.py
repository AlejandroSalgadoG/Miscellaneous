import sys
import numpy as np

times = np.loadtxt(sys.argv[1], dtype=int)
num_jobs, num_machines = times.shape
bound = np.sum(times)
