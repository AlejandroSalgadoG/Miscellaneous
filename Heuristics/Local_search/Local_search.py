import numpy as np

from Calc_cost import calc_cost
from Reader import num_jobs
    
def local_search(solution):
    best_cost = calc_cost(solution)
    best_sol = solution
    for job_idx in range(num_jobs):
        sub_sol = np.delete(solution,job_idx)
        for idx in range(num_jobs):
            if job_idx != idx:
                sol = np.insert(sub_sol, idx, solution[job_idx])
                sol_cost = calc_cost(sol)
                if sol_cost < best_cost:
                    best_cost = sol_cost
                    best_sol = sol
    return best_sol
