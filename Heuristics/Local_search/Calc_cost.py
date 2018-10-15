import sys
import numpy as np

from Reader import times, num_machines

def calc_cost(solution):
    return sum([ schedule[-1] for schedule in calc_schedule(solution)] )

def calc_schedule(solution):
    sol_schedule = [ [0]*num_machines ]
    for idx, time in enumerate(times[solution]):
        job_schedule = calc_job_schedule(sol_schedule[idx], time) 
        sol_schedule.append(job_schedule)
    return sol_schedule[1:]

def calc_job_schedule(prev_schedule, job_schedule):
    schedule = [0] * num_machines
    for idx, duration in enumerate(job_schedule):
        schedule[idx] = max(schedule[idx-1], prev_schedule[idx]) + duration
    return schedule
