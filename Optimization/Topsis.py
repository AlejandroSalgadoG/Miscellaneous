#!/usr/bin/env python3

from Topsis_functions import *

def topsis(A, w, min_max):
    print_matrix("A", A)
    print_vector("w", w)
    print_vector("Intention (0 minimize, 1 maximize)", min_max)

    N = normalize(A)
    print_matrix("N", N)

    N_pond = ponderate(N, w)
    print_matrix("N ponderated", N_pond)

    ideal = get_ideal_vec(N_pond, min_max)
    print_vector("Ideal", ideal)

    nadir = get_nadir_vec(N_pond, min_max)
    print_vector("Nadir", nadir)

    alt_num = len(A)

    d_plus = [ get_distance(N_pond[i], ideal, p=2) for i in range(alt_num) ]
    d_minus = [ get_distance(N_pond[i], nadir, p=2) for i in range(alt_num) ]

    distances = [d_plus, d_minus]
    distances = get_transpose(distances)
    print_matrix("dist + dist -", distances)

    similar_ratio = [ [ get_similarity(d_plus[i], d_minus[i]) ] for i in range(alt_num) ]
    print_matrix("Similarity ratio", similar_ratio)

def main():
    A = [
          [  1,  8,  1  ],
          [  2,  6,  2  ],
          [ 2.5, 7, 1.5 ],
          [  3,  4,  2  ],
          [  4,  2, 2.5 ]
        ]  

    w = [ 0.588, 0.294, 0.118 ]
    min_max = [ minimize, minimize, minimize ]

    topsis(A,w, min_max)

if __name__ == '__main__':
   main()
