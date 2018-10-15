import sys
import numpy as np

from Reader import num_jobs
from Calc_cost import calc_cost

def best_insertion():
    solution = np.array([], dtype=int)
    unallocated = list(range(num_jobs))

    for job_idx in range(num_jobs):
        best_cost = sys.maxsize
        for job in unallocated:
            for pos in range(job_idx+1):
                sol = np.insert(solution, pos, job)
                cost = calc_cost(sol)

                if cost < best_cost:
                    best_cost = cost
                    best_insertion = [job,pos]

        job, pos = best_insertion
        solution = np.insert(solution, pos, job)
        unallocated.remove(job)

    return solution
