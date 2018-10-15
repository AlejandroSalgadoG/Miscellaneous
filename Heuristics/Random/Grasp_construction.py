import sys
from time import time
import numpy as np

from Reader import times, num_jobs, num_machines, bound

def calc_job_schedule(release, durations):
    schedule = [0 for i in durations]
    for idx, duration in enumerate(durations):
        schedule[idx] = max(schedule[idx-1], release[idx]) + duration
    return schedule

def get_score(unalloc_jobs, schedule):
    return np.array([ calc_job_schedule(schedule, job) for job in unalloc_jobs ])

def get_rcl(options, alpha):
    scores = options[:,-1]
    d_max, d_min = max(scores), min(scores)
    threshold = (d_max-d_min)*alpha + d_min
    return [idx for idx,score in enumerate(scores) if score <= threshold]

def grasp_construction(alpha):
    cost = 0
    solution = []
    unallocated = list(range(num_jobs))
    schedule = np.zeros(num_machines)

    for i in range(num_jobs):
        options = get_score(times[unallocated], schedule)
        rcl = get_rcl(options, alpha)

        neighbor_idx = np.random.choice(rcl)
        neighbor = unallocated[neighbor_idx]

        solution.append(neighbor)
        schedule = options[neighbor_idx]
        cost += schedule[-1]
        del unallocated[neighbor_idx]

    return solution, cost

def main(alpha):
    start = time()
    solution, cost = grasp_construction(float(alpha))
    end = time()

    print( "Solution:", solution )
    print( "Objective function: %.1f" % cost )
    print( "Bound: %.1f" % bound )
    print( "Gap: %f" % ((cost - bound) / bound * 100) )
    print( "Time elapsed: %.2f seconds" % (end - start) )

main(sys.argv[2])
