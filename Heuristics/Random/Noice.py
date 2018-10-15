import sys
from time import time
import numpy as np

from Reader import times, num_jobs, num_machines, bound

def calc_job_schedule(release, durations):
    schedule = [0 for i in durations]
    for idx, duration in enumerate(durations):
        schedule[idx] = max(schedule[idx-1], release[idx]) + duration
    return schedule

def calc_new_schedule(start_schedule, jobs):
    schedule = [ start_schedule ]
    for idx, job in enumerate(jobs):
        new_schedule = calc_job_schedule(schedule[idx], job)
        schedule.append(new_schedule)
    return schedule

def calc_cost(first_part, second_part):
    cost = 0
    for elem in first_part:
        cost += elem[-1]
    for elem in second_part:
        cost += elem[-1]
    return cost

def prepend(element, array):
    return np.insert(array, 0, element, axis=0)

def best_insertion(jobs):
    num_jobs, num_machines = jobs.shape

    solution = np.array([], dtype=np.int)
    schedule = [ [0] * num_machines ]
    unallocated = list(range(num_jobs))

    for job_idx in range(num_jobs):
        best_cost = sys.maxsize
        best_schedule = schedule
        for job in unallocated:
            for pos in range(job_idx+1):
                tmp_jobs = prepend(jobs[job], jobs[solution[pos:]])
                new_schedule = calc_new_schedule(schedule[pos], tmp_jobs)
                tmp_cost = calc_cost(schedule[:pos], new_schedule)

                if tmp_cost < best_cost:
                    best_cost = tmp_cost
                    insertion = pos, job
                    best_schedule = new_schedule

        pos, job = insertion
        solution = np.insert(solution, pos, job)
        schedule[pos:] = best_schedule
        unallocated.remove(job)

    return solution

def apply_noice(r):
    if r == 0:
        return times
    else:
        return times + np.random.randint(-r, r+1, size=times.shape)

def main(r):
    noise_instance = apply_noice(int(r))

    start = time()
    solution = best_insertion(noise_instance)
    end = time()

    null_schedule = [0 for i in times[0]]
    schedule = calc_new_schedule(null_schedule, times[solution])
    cost = calc_cost(schedule, [])

    print( "Solution:", solution )
    print( "Objective function: %.1f" % cost )
    print( "Bound: %.1f" % bound )
    print( "Gap: %f" % ((cost - bound) / bound * 100) )
    print( "Time elapsed: %.2f seconds" % (end - start) )

main(sys.argv[2])
