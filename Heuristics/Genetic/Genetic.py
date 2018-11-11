import numpy as np
from random import random

from Calc_cost import calc_cost
from Reader import num_jobs, num_machines

def build_initial_pop(num_sampl, num_jobs):
    return np.array([ np.random.permutation(num_jobs) for i in range(num_sampl)])

def delete_elements(array, elements):
    new_array = list(array)
    for element in elements:
        new_array.remove(element)
    return new_array

def crossover(parent1, parent2):
    result = np.copy(parent1)
    cut_point = np.random.randint(1, num_machines-1)
    result[cut_point:] = delete_elements(parent2, parent1[:cut_point])
    return result

def calc_costs(samples):
    return [ calc_cost(sample) for sample in samples ]

def calc_probs(costs):
    inv_costs = [1/cost for cost in costs]
    total_sum = sum(inv_costs)
    return [inv_cost/total_sum for inv_cost in inv_costs]

def select_couple(population, costs):
    probs = calc_probs(costs)
    num_elem, _ = population.shape
    idx_popul = np.arange(num_elem)
    sel_elem = np.random.choice(idx_popul, 2, replace=False, p=probs)
    return population[sel_elem]

def swap(pos1, pos2, solution):
    sol = np.copy(solution)
    tmp = sol[pos1]
    sol[pos1] = sol[pos2]
    sol[pos2] = tmp
    return sol

def mutate(solution):
    idx1, idx2 = np.random.choice(solution, 2, replace=False)
    return swap(idx1, idx2, solution)

def update_population(population, costs, new_population, new_costs):
    num_elem, _ = population.shape
    join_population = np.concatenate((population, new_population))
    join_costs = np.concatenate((costs, new_costs))

    sort_idx = np.argsort(join_costs)
    join_costs = join_costs[sort_idx]
    join_population = join_population[sort_idx]

    return join_population[:num_elem], join_costs[:num_elem]

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

def genetic(num_iter, num_sampl, mut_prob, local_prob):
    population = build_initial_pop(num_sampl, num_jobs)
    costs = calc_costs(population)

    new_population = np.zeros((num_sampl, num_jobs), dtype=int)
    new_costs = np.zeros(num_sampl, dtype=int)

    for i in range(num_iter):
        for j in range(num_sampl):
            parent1, parent2 = select_couple(population, costs)
            son = crossover(parent1, parent2)
            if random() < mut_prob:
                son = mutate(son)
            if random() < local_prob:
                son = local_search(son)

            new_costs[j] = calc_cost(son)
            new_population[j] = son
        population, costs = update_population(population, costs, new_population, new_costs)

    best_idx = np.argmin(costs)
    return population[best_idx], costs[best_idx]
