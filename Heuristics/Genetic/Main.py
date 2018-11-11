import sys
from time import time

from Genetic import genetic

def main(num_iter, num_sampl, mut_prob, local_prob):
    start = time()
    solution, cost = genetic(num_iter, num_sampl, mut_prob, local_prob)
    end = time()

    print( "Objective function: %d" % cost )
    print( "Time elapsed: %.2f seconds" % (end - start) )

main( int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]) )
