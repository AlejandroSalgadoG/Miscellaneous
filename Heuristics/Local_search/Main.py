import sys
import numpy as np
from time import time

from Reader import bound
from Ils import ils

def main(num_iter):
    start = time()
    solution, cost = ils(num_iter)
    end = time()

    print( "Solution:", solution )
    print( "Objective function: %.1f" % cost )
    print( "Bound: %.1f" % bound )
    print( "Gap: %f" % ((cost - bound) / bound * 100) )
    print( "Time elapsed: %.2f seconds" % (end - start) )

main( int(sys.argv[2]) )
