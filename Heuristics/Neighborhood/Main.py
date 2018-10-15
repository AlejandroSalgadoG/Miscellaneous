import sys
import numpy as np
from time import time

from Reader import bound
from Vnd import vnd

def main():
    start = time()
    solution, cost = vnd()
    end = time()

    print( "Solution:", solution )
    print( "Objective function: %.1f" % cost )
    print( "Bound: %.1f" % bound )
    print( "Gap: %f" % ((cost - bound) / bound * 100) )
    print( "Time elapsed: %.2f seconds" % (end - start) )

main()
