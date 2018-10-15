from Best_insertion import best_insertion
from Local_search import local_search_1, local_search_2

def vnd():
    best_sol, best_cost = best_insertion()
    i = 1
    while i <= 2:
        if i == 1:
            sol, sol_cost = local_search_1(best_sol)
        else:
            sol, sol_cost = local_search_2(best_sol)

        if sol_cost < best_cost:
            best_cost = sol_cost
            best_sol = sol
            i = 1
        else:
            i += 1

    return best_sol, best_cost
