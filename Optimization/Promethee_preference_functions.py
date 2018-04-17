def function1(criteria, x):
    if x <= 0:
        return 0
    else:
        return 1 

def function2(criteria, x):
    if criteria == 3: # remembrer 0 index
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
