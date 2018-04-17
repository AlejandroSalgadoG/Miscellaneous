import sys

from Promethee_preference_functions import *

minimize = 0
maximize = 1

def get_0_matrix(height, width):
    return [ [ 0 for j in range(width) ] for i in range(height) ]

def print_matrix(title, A):
    height = len(A)
    width = len(A[0])
    
    print(title)
    for i in range(height):
        for j in range(width):
            sys.stdout.write( " %.2f " % A[i][j])
        print( "" )

def print_vector(title, b):
    length = len(b)

    print(title)
    for i in range(length):
        sys.stdout.write( " %.2f " % b[i])
    print( "" )

def get_transpose(A):
    alt_num = len(A)
    criteria_num = len(A[0])

    transpose = []

    for j in range(criteria_num):
        column = [ A[i][j] for i in range(alt_num) ]
        transpose.append(column)
    return transpose

def calculate_pi(i,j, A, w, min_max, preference_fun):
    criteria_num = len(A[0])

    pi = 0
    for k in range(criteria_num):
        if min_max[k] == maximize:
            x = A[i][k] - A[j][k]
        else:
            x = A[j][k] - A[i][k]
        pi += w[k] * preference_fun(k, x)
    return pi

def calc_pi_table(A, w, min_max, preference_fun):
    alt_num = len(A)
    criteria_num = len(A[0])
    
    pi_matrix  = get_0_matrix(alt_num, criteria_num)

    for i in range(alt_num):
        for j in range(criteria_num):
            pi_matrix[i][j] = calculate_pi(i,j, A, w, min_max, preference_fun)
    return pi_matrix

def calc_phi_plus(A, alt):
    alt_num = len(A)
    criteria_num = len(A[0])
    n_1 = alt_num - 1
    total_pi = 0

    for j in range(criteria_num):
        total_pi += A[alt][j]
    return total_pi/n_1

def calc_phi_minus(A, alt):
    alt_num = len(A)
    n_1 = alt_num - 1
    total_pi = 0

    for i in range(alt_num):
        total_pi += A[i][alt]
    return total_pi/n_1
