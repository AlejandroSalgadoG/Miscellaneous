#!/bin/env python

from Electre_functions import *

def electre(A, w, min_max):
    print_matrix("A", A)

    concordance = get_concordance(A, w, min_max)
    print_matrix("Corcordance matrix", concordance)

    N = normalize(A)
    print_matrix("N", N)

    N_pond = ponderate(N, w)
    print_matrix("N ponderated", N_pond)

    discordance = get_discordance(N_pond, min_max)
    print_matrix("Discordance matrix", discordance)

    c = get_mean(concordance)
    d = get_mean(discordance)
    print("Umbrals c=%.2f, d=%.2f" % (c,d))

def main():
    A = [
          [  1,  8,  1  ],
          [  2,  6,  2  ],
          [ 2.5, 7, 1.5 ],
          [  3,  4,  2  ],
          [  4,  2, 2.5 ]
        ]  

    w = [ 0.588, 0.294, 0.118 ]
    min_max = [ "min", "min", "min" ]

    electre(A,w, min_max)

if __name__ == '__main__':
   main()
