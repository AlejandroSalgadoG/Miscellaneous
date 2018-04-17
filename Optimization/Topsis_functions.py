import sys

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

def normalize(A):
    alt_num = len(A)
    criteria_num = len(A[0])

    N = get_0_matrix(alt_num, criteria_num)

    A_transpose = get_transpose(A)

    min_vals = []
    max_vals = []

    for j in range(criteria_num):
         min_vals.append( min(A_transpose[j]) )
         max_vals.append( max(A_transpose[j]) )

    for i in range(alt_num):
        for j in range(criteria_num):
            N[i][j] = (A[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j])

    return N

def ponderate(N, w):
    alt_num = len(N)
    criteria_num = len(N[0])

    N_pond = get_0_matrix(alt_num, criteria_num)

    for i in range(alt_num):
        for j in range(criteria_num):
            N_pond[i][j] = N[i][j] * w[j]

    return N_pond

def get_ideal_vec(A, min_max):
    criteria_num = len(A[0])

    A_transpose = get_transpose(A)
    ideal = []

    for j in range(criteria_num):
        if min_max[j] == maximize:
            ideal.append( max(A_transpose[j]) )
        else:
            ideal.append( min(A_transpose[j]) )
    return ideal

def get_nadir_vec(A, min_max):
    criteria_num = len(A[0])

    A_transpose = get_transpose(A)
    nadir = []

    for j in range(criteria_num):
        if min_max[j] == maximize:
            nadir.append( min(A_transpose[j]) )
        else:
            nadir.append( max(A_transpose[j]) )
    return nadir

def get_distance(a, vector, p):
    criteria_num = len(a)

    sum_total = 0
    for j in range(criteria_num):
        sum_total += (a[j] - vector[j]) ** p

    return sum_total ** (1/p)

def get_similarity(d_plus, d_minus):
    return d_minus / (d_plus + d_minus)
