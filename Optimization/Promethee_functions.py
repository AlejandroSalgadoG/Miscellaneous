import sys

def function1(criteria, x):
    if x <= 0:
        return 0
    else:
        return 1 

def function2(criteria, x):
    if criteria == 3:
        l = 0.75
    else:
        l = 1.5

    if x <= l:
        return 0
    else:
        return 1

def function3(criteria, x):
    if criteria == 3:
        m = 0.75
    else:
        m = 1.5

    if x <= m:
        if x < 0:
            return 0
        return x/m
    else:
        return 1


def display_output(pi_table, phi_plus, phi_minus, phi_total):
    options_num = len(pi_table)
    criteria_num = len(pi_table[0])

    print("Pi table")
    for i in range(options_num):
        for j in range(criteria_num):
            if i == j:
                sys.stdout.write("  x  ")
                continue
            sys.stdout.write("%.2f " % pi_table[i][j])
        print()

    print()

    print("Promethee 1")
    for option in range(options_num):
        print("%.2f, %.2f" % (phi_plus[option], phi_minus[option]))
    
    print()

    print("Promethee 2")
    for option in range(options_num):
        print("%.2f" % phi_total[option])
