sleepIn x y
    | x == False && y == False = True
    | x == False && y == True = True
    | x == True && y == False = False
    | x == True && y == True = True

main = print ( sleepIn False False )
