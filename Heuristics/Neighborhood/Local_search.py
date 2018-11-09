import sys
import numpy as np

from Calc_cost import calc_cost
from Reader import num_jobs

def local_search_1(solution):
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
    return best_sol, best_cost

def swap(pos1, pos2, solution):
    sol = np.copy(solution)
    tmp = sol[pos1]
    sol[pos1] = sol[pos2]
    sol[pos2] = tmp
    return sol

def local_search_2(solution):
    best_cost = calc_cost(solution)
    best_sol = solution
    for idx1 in range(num_jobs):
        for idx2 in range(idx1+1, num_jobs):
            sol = swap(idx1, idx2, solution)
            sol_cost = calc_cost(sol)
            if sol_cost < best_cost:
                best_cost = sol_cost
                best_sol = sol
    return best_sol, best_cost
