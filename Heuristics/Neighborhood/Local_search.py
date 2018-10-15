import sys
import numpy as np

from Calc_cost import calc_cost
from Reader import num_jobs

def remove_element(array, pos):
    return np.delete(array, pos), array[pos]

def local_search_1(solution):
    best_cost = sys.maxsize
    best_sol = None
    for idx in range(num_jobs):
        sol, sol_cost = sub_search( *remove_element(solution, idx) )
        if sol_cost < best_cost:
            best_cost = sol_cost
            best_sol = sol
    return best_sol, best_cost
    
def local_search_2(solution):
    best_cost = sys.maxsize
    best_sol = None
    for idx1 in range(num_jobs):
        sub_sol1, job1 = remove_element(solution, idx1)   
        for idx2 in range(num_jobs-1):
            sub_sol2, _ = sub_search( *remove_element(sub_sol1, idx2) ) 
            sol, sol_cost = sub_search(sub_sol2, job1) 
            if sol_cost < best_cost:
                best_cost = sol_cost
                best_sol = sol
    return best_sol, best_cost
                
def sub_search(sub_sol, job):
    best_cost = sys.maxsize
    best_sol = None
    for idx in range(sub_sol.size+1):
        sol = np.insert(sub_sol, idx, job)
        sol_cost = calc_cost(sol)
        if sol_cost < best_cost:
            best_cost = sol_cost
            best_sol = sol
    return best_sol, best_cost
