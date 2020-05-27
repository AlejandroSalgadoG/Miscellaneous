avocato_weights = {
                    12: 0.5575, 
                    14: 0.2975,  
                    16: 0.2650,  
                    18: 0.2365,  
                    20: 0.2115,  
                    22: 0.1890,  
                    24: 0.1715,  
                    26: 0.1565,  
                    28: 0.1435,  
                    30: 0.1330,  
                    32: 0.1135   
                  }

c_probs = {
              12: 0.001130,
              14: 0.013581,
              16: 0.046228,
              18: 0.076699,
              20: 0.128074,
              22: 0.159559,
              24: 0.117110,
              26: 0.143708,
              28: 0.075032,
              30: 0.075599,
              32: 0.103716,
              "extra": 0.003849,
              "primera": 0.007316,
              "segunda": 0.009419,
              "tercera": 0.014854,
              "otros": 0.024125
          }

def calculate_waste_probs():
    waste_probs = {}

    total_extra =  c_probs[12] + c_probs[14] + c_probs[16] + c_probs[18]
    for size in [12, 14, 16, 18]:
        waste_probs[size] = c_probs["extra"] * c_probs[size]/total_extra

    total_primera =  c_probs[20] + c_probs[22]
    for size in [20, 22]:
        waste_probs[size] = c_probs["primera"] * c_probs[size]/total_primera

    total_segunda =  c_probs[24] + c_probs[26]
    for size in [24, 26]:
        waste_probs[size] = c_probs["segunda"] * c_probs[size]/total_segunda

    total_tercera =  c_probs[28] + c_probs[30] + c_probs[32]
    for size in [28, 30, 32]:
        waste_probs[size] = c_probs["tercera"] * c_probs[size]/total_tercera

    return waste_probs

def calculate_machine_probs():
    transitions = []
    waste_probs = calculate_waste_probs()

    for size in [12, 14, 16, 18]:
        transitions.append( c_probs[size] + waste_probs[size] ) 

    for size in [20, 22]:
        transitions.append( c_probs[size] + waste_probs[size] ) 

    for size in [24, 26]:
        transitions.append( c_probs[size] + waste_probs[size] ) 

    for size in [28, 30, 32]:
        transitions.append( c_probs[size] + waste_probs[size] ) 

    transitions.append( c_probs["otros"] )

    return transitions
