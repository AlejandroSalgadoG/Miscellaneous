#!/usr/bin/env python3

from Electre_functions import *

def electre(A, w, min_max):
    print_matrix("A", A)
    print_vector("w", w)
    print_vector("Intention (0 minimize, 1 maximize)", min_max)

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

    dom_concordant = get_dom_concordant(c, concordance)
    print_matrix("Concordant dominance", dom_concordant)

    dom_discordant = get_dom_discordant(d, discordance)
    print_matrix("Discordant dominance", dom_discordant)

    dominance = get_dominance(dom_concordant, dom_discordant)
    print_matrix("Dominance matrix", dominance)


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

    electre(A,w, min_max)

if __name__ == '__main__':
   main()
