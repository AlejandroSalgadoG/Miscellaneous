#!/usr/bin/env python3

from Promethee_functions import *

def promethee(A, w, min_max, preference_fun):
    print_matrix("A", A)
    print_vector("w", w)
    print_vector("Intention (0 minimize, 1 maximize)", min_max)

    pi_matrix = calc_pi_table(A, w, min_max, preference_fun)
    print_matrix("Pi matrix", pi_matrix)

    alt_num = len(A)

    phi_plus = [ calc_phi_plus(pi_matrix, alt) for alt in range(alt_num) ]
    phi_minus = [ calc_phi_minus(pi_matrix, alt) for alt in range(alt_num) ]

    phi_combined = [phi_plus, phi_minus]
    phi_combined = get_transpose(phi_combined)
    print_matrix("Phi +  Phi -", phi_combined)

    phi_total = [ [ phi_plus[alt] - phi_minus[alt] ] for alt in range(alt_num) ]
    print_matrix("Phi total", phi_total)

def main():
    A  = [
           [5, 9, 9, 6],
           [3, 7, 7, 4],
           [7, 3, 1, 7],
           [9, 6, 5, 3]
         ]

    ahp_weights =  [ 0.27, 0.29, 0.23, 0.21 ]
    goal_weights = [ 0.25, 0.30, 0.23, 0.21 ]

    min_max = [ maximize, minimize, maximize, maximize ]

    promethee(A, goal_weights, min_max, function3)

if __name__ == '__main__':
   main()
