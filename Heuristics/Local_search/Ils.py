import numpy as np

from Best_insertion import best_insertion
from Local_search import local_search
from Calc_cost import calc_cost

def accept_criteria(sol, best_cost):
    return min(calc_cost(sol), best_cost)

def swap(pos1, pos2, solution):
    sol = np.copy(solution)
    tmp = sol[pos1]
    sol[pos1] = sol[pos2]
    sol[pos2] = tmp
    return sol

def perturbate(solution):
    job1, job2 = np.random.choice(solution,2,replace=False)
    return swap(job1, job2, solution)

def ils(num_iter):
    initial_sol = best_insertion()
    solution = local_search(initial_sol)
    best_cost = calc_cost(solution)

    for i in range(num_iter):
        sol = perturbate(solution)
        sol = local_search(sol)
        sol_cost = calc_cost(sol)

        if sol_cost < best_cost:
            solution = sol
            best_cost = sol_cost

    return solution, best_cost
