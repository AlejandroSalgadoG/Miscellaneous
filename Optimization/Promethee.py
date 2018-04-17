#!/bin/env python

from Promethee_functions import *

def calc_pi_table(table, weights, method_func):
    height = len(table)
    width = len(table[0])
    
    pi_table  = [[ 0 for j in range(width)] for i in range(height)]

    for i in range(height):
        for j in range(width):
            if i == j:
                continue
            pi = 0
            for criteria in range(width):
                x = table[i][criteria] - table[j][criteria]
                pi += weights[criteria] * method_func(criteria, x)
            pi_table[i][j] = pi
    return pi_table

def calc_phi_plus(table, option):
    height = len(table)
    width = len(table[0])

    n_1 = height - 1

    total_pi = 0
    for j in range(width):
        if j == option:
            continue
        total_pi += table[option][j]

    return total_pi/n_1

def calc_phi_minus(table, option):
    height = len(table)
    width = len(table[0])

    n_1 = height - 1

    total_pi = 0
    for i in range(height):
        if i == option:
            continue
        total_pi += table[i][option]

    return total_pi/n_1

def main():
    table  = [[5, -9, 9, 6],
              [3, -7, 7, 4],
              [7, -3, 1, 7],
              [9, -6, 5, 3]]

    ahp_weights = [0.27, 0.29, 0.23, 0.21]
    goal_weights = [0.25, 0.3, 0.23, 0.21]

    pi_table = calc_pi_table(table, goal_weights, function2)

    options_num = len(table)

    phi_plus = [ calc_phi_plus(pi_table, option) for option in range(options_num) ]
    phi_minus = [ calc_phi_minus(pi_table, option) for option in range(options_num) ]
    phi_total = [ phi_plus[option] - phi_minus[option] for option in range(options_num) ]

    display_output(pi_table, phi_plus, phi_minus, phi_total)
    
if __name__ == '__main__':
   main()
