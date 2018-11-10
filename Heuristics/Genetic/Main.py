import sys
from time import time

from Genetic import genetic

def main(num_iter):
    start = time()
    solution, cost = genetic(num_iter, num_sampl)
    end = time()

#    print( "Solution:", solution )
    print( "Objective function: %d" % cost )
    print( "Time elapsed: %.2f seconds" % (end - start) )

main( int(sys.argv[2]), int(sys.argv[3]) )
